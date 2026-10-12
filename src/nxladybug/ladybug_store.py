"""LadybugDB storage provider for arrow-backed graphs.

Persists the per-type ``pyarrow.Table`` frames handed over by
:func:`nxarrow.backend.persist` into a LadybugDB database with one
unified node table and one unified relationship table::

    NxNode(node, <attrs...>, node_type, _nx_order)
    NxEdge(FROM NxNode TO NxNode, <attrs...>, edge_type, _nx_order)
    NxMeta(key, value)   # JSON-encoded graph-level attributes

Table names are configurable at construction. Every persist drops and
recreates the tables, so re-persisting a smaller graph leaves no stale
rows behind.

Each frame is bulk-loaded in two steps — Arrow memory registration into
a staging table, then a native bulk copy:

.. code-block:: python

    conn.create_arrow_table(stage, frame)          # register Arrow memory
    conn.execute(f"COPY NxNode FROM "             # bulk copy out of it
                 f"(MATCH (s:`{stage}`) RETURN s.`node`, ...)")

Column mapping is positional (staging ``RETURN`` order matches the
stored column order), mirroring the ``COPY <table> FROM $df`` convention
used by the LDBC bulk loader. Endpoints resolve through the node primary
key inside ``COPY``. Empty frames are skipped; a graph with no nodes or
no edges simply leaves that table out (loads treat a missing table as
empty).

Round-trip::

    provider = LadybugStorageProvider(":memory:")
    persist(graph, provider, graph_attrs=graph.graph)
    restored = provider.load_graph()

``load_graph`` reads the tables back with ``MATCH ... RETURN`` (ordered
by the internal sequence column, which is then dropped) and rebuilds an
``ArrowDiGraph`` via :meth:`ArrowDiGraph.from_arrow`, so the restored
graph compares equal to the persisted one.

Fidelity notes. First, Arrow types map to Ladybug types: all integers to
``INT64``, floats to ``DOUBLE``, booleans to ``BOOL``, strings to
``STRING``, dates to ``DATE`` and timestamps to ``TIMESTAMP``; anything
else (times, durations, binaries, nested types, ...) is cast to
``STRING`` before the copy, which stringifies time keys in particular.
Second, the node key column (``node``) becomes the Ladybug primary key,
so key values must be unique, as they always are in a graph. Third,
``graph_attrs`` must be JSON-serializable; it is forwarded through
:func:`persist` keyword arguments and stored as one JSON document in the
meta table. Fourth, ``_nx_order`` is a reserved column name used for the
internal insertion sequence; graphs using it as an attribute are refused
with a clear error.
"""

import json
import uuid

import pyarrow as pa

from nxarrow.graph import ArrowDiGraph

__all__ = ["LadybugStorageProvider"]

DEFAULT_NODE_TABLE = "NxNode"
DEFAULT_EDGE_TABLE = "NxEdge"
DEFAULT_META_TABLE = "NxMeta"

#: Reserved column holding the global insertion sequence. Lets loads
#: restore exact node/edge order via ``ORDER BY`` regardless of storage
#: order; it is stripped from the adopted tables so it never surfaces as
#: an attribute.
ORDER_COL = "_nx_order"


def _quote(identifier):
    """Quote an identifier with backticks (Ladybug escape: double them)."""
    return "`" + str(identifier).replace("`", "``") + "`"


def _literal(value):
    """Quote a string literal with single quotes (escape: double them)."""
    return "'" + str(value).replace("'", "''") + "'"


def _ladybug_type(dtype):
    """Map an Arrow type to a Ladybug column type (None: cast to STRING)."""
    if pa.types.is_dictionary(dtype):
        return _ladybug_type(dtype.value_type)
    if pa.types.is_integer(dtype) or pa.types.is_unsigned_integer(dtype):
        return "INT64"
    if pa.types.is_floating(dtype):
        return "DOUBLE"
    if pa.types.is_boolean(dtype):
        return "BOOL"
    if pa.types.is_string(dtype) or pa.types.is_large_string(dtype):
        return "STRING"
    if pa.types.is_date(dtype):
        return "DATE"
    if pa.types.is_timestamp(dtype):
        return "TIMESTAMP"
    return None


def _stage_frame(table):
    """Cast unmapped columns to STRING so every column has a Ladybug type."""
    names = table.column_names
    if ORDER_COL in names:
        raise ValueError(
            f"attribute name {ORDER_COL!r} is reserved for Ladybug persistence; "
            "rename the attribute before persisting"
        )
    out = table
    for field in table.schema:
        if _ladybug_type(field.type) is None:
            idx = out.schema.get_field_index(field.name)
            out = out.set_column(
                idx, field.name, out.column(field.name).cast(pa.string())
            )
    return out


def _unify(tables):
    """Concatenate tables with heterogeneous columns, filling gaps with nulls."""
    if len(tables) == 1:
        return tables[0]
    schema = pa.unify_schemas([t.schema for t in tables])
    return pa.concat_tables([t.cast(schema) for t in tables])


class LadybugStorageProvider:
    """Store per-type Arrow frames in LadybugDB node/relationship tables.

    Parameters
    ----------
    db_path:
        LadybugDB database path, or ``":memory:"`` (default) for a
        transient database that lives as long as this provider. File
        paths persist across processes; reopen with
        :meth:`LadybugStorageProvider.load`.
    node_table, edge_table, meta_table:
        Ladybug table names for nodes, edges and the graph-attrs sidecar.
    """

    def __init__(
        self,
        db_path=":memory:",
        node_table=DEFAULT_NODE_TABLE,
        edge_table=DEFAULT_EDGE_TABLE,
        meta_table=DEFAULT_META_TABLE,
    ):
        self.db_path = str(db_path)
        self.node_table = node_table
        self.edge_table = edge_table
        self.meta_table = meta_table
        self._db = None
        self._conn = None

    # -- connection lifecycle --

    @property
    def conn(self):
        """The live Ladybug connection, opened on first use."""
        if self._conn is None:
            import ladybug as lb

            self._db = lb.Database(self.db_path)
            self._conn = lb.Connection(self._db)
        return self._conn

    def close(self):
        """Close the connection (and database). The provider is unusable after."""
        if self._conn is not None:
            try:
                self._conn.close()
            finally:
                self._conn = None
        if self._db is not None:
            try:
                self._db.close()
            finally:
                self._db = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    # -- storage provider protocol (see nxarrow.backend.persist) --

    def store_nodes(self, frames, graph_attrs=None, **kwargs):
        """Bulk-load node frames into the node table. Returns ``{table: rows}``."""
        rows = [t.num_rows for t in frames.values()]
        total = sum(rows)
        conn = self.conn
        # Drop the edge table first: it references the node table, and both
        # are rebuilt on every persist so no stale rows survive re-persist.
        conn.execute(f"DROP TABLE IF EXISTS {_quote(self.edge_table)}")
        conn.execute(f"DROP TABLE IF EXISTS {_quote(self.node_table)}")
        if total:
            unified = _unify([_stage_frame(t) for t in frames.values()])
            self._create_node_table(unified)
            offset = 0
            for table in frames.values():
                offset = self._copy_node_frame(table, unified.schema, offset)
        if graph_attrs is not None:
            self._store_meta(dict(graph_attrs))
        return {self.node_table: total} if total else {}

    def store_edges(self, frames, graph_attrs=None, **kwargs):
        """Bulk-load edge frames into the relationship table. Returns ``{table: rows}``."""
        rows = [t.num_rows for t in frames.values()]
        total = sum(rows)
        conn = self.conn
        conn.execute(f"DROP TABLE IF EXISTS {_quote(self.edge_table)}")
        if total:
            self._ensure_node_table_for_edges(frames)
            unified = _unify([_stage_frame(t) for t in frames.values()])
            self._create_edge_table(unified)
            offset = 0
            for table in frames.values():
                offset = self._copy_edge_frame(table, unified.schema, offset)
        if graph_attrs is not None:
            self._store_meta(dict(graph_attrs))
        return {self.edge_table: total} if total else {}

    # -- loading --

    def load_graph(self):
        """Rebuild the persisted graph (tables, types, attrs, graph attrs)."""
        nodes = self._load_table(self.node_table, endpoint=False)
        edges = self._load_table(self.edge_table, endpoint=True)
        graph = ArrowDiGraph.from_arrow(nodes, edges)
        graph.graph.update(self._load_meta())
        return graph

    @classmethod
    def load(cls, db_path, **kwargs):
        """Open the database at ``db_path`` and rebuild the persisted graph."""
        return cls(db_path, **kwargs).load_graph()

    # -- DDL --

    def _create_node_table(self, unified):
        cols = [
            f"{_quote(f.name)} {_ladybug_type(f.type)}" for f in unified.schema
        ]
        cols.append(f"{_quote(ORDER_COL)} INT64")
        cols.append(f"PRIMARY KEY ({_quote('node')})")
        self.conn.execute(
            f"CREATE NODE TABLE {_quote(self.node_table)} ({', '.join(cols)})"
        )

    def _create_edge_table(self, unified):
        props = [
            f"{_quote(f.name)} {_ladybug_type(f.type)}"
            for f in unified.schema
            if f.name not in ("source", "target")
        ]
        props.append(f"{_quote(ORDER_COL)} INT64")
        self.conn.execute(
            f"CREATE REL TABLE {_quote(self.edge_table)} "
            f"(FROM {_quote(self.node_table)} TO {_quote(self.node_table)}"
            + (f", {', '.join(props)}" if props else "")
            + ")"
        )

    def _ensure_node_table_for_edges(self, frames):
        """Create a bare node table so a standalone edge store has a TO/FROM target.

        :func:`persist` always stores nodes first, so this only triggers on
        direct ``store_edges`` calls. The key type is inferred from the
        ``source`` column of the unified edge frames.
        """
        names = {row[0] for row in self._table_names()}
        if self.node_table in names:
            return
        unified = _unify([_stage_frame(t) for t in frames.values()])
        key_type = _ladybug_type(unified.schema.field("source").type) or "STRING"
        self.conn.execute(
            f"CREATE NODE TABLE {_quote(self.node_table)} "
            f"({_quote('node')} {key_type}, PRIMARY KEY ({_quote('node')}))"
        )

    # -- COPY --

    def _copy_frame(self, dest_table, dest_cols, frame):
        """Register ``frame`` as a staging table and bulk-copy it into ``dest_table``.

        Column mapping is positional: the ``RETURN`` list follows
        ``dest_cols`` order, matching the stored column order.
        """
        stage = f"_nx_stage_{uuid.uuid4().hex}"
        conn = self.conn
        conn.create_arrow_table(stage, frame)
        try:
            aliased = "s"
            returns = ", ".join(
                f"{_quote(aliased)}.{_quote(c)}" for c in dest_cols
            )
            conn.execute(
                f"COPY {_quote(dest_table)} FROM "
                f"(MATCH ({_quote(aliased)}:{_quote(stage)}) RETURN {returns})"
            )
        finally:
            try:
                conn.drop_arrow_table(stage)
            except Exception:
                pass

    def _copy_node_frame(self, table, schema, offset):
        if table.num_rows == 0:
            return offset
        staged = _stage_frame(table.cast(schema))
        order = pa.array(
            range(offset, offset + table.num_rows), type=pa.int64()
        )
        staged = staged.append_column(ORDER_COL, order)
        self._copy_frame(self.node_table, [*schema.names, ORDER_COL], staged)
        return offset + table.num_rows

    def _copy_edge_frame(self, table, schema, offset):
        if table.num_rows == 0:
            return offset
        staged = _stage_frame(table.cast(schema))
        order = pa.array(
            range(offset, offset + table.num_rows), type=pa.int64()
        )
        staged = staged.append_column(ORDER_COL, order)
        self._copy_frame(self.edge_table, [*schema.names, ORDER_COL], staged)
        return offset + table.num_rows

    # -- graph attrs sidecar --

    def _store_meta(self, graph_attrs):
        import pyarrow as pa

        conn = self.conn
        conn.execute(f"DROP TABLE IF EXISTS {_quote(self.meta_table)}")
        conn.execute(
            f"CREATE NODE TABLE {_quote(self.meta_table)} "
            f"({_quote('key')} STRING, {_quote('value')} STRING, "
            f"PRIMARY KEY ({_quote('key')}))"
        )
        frame = pa.table(
            {
                "key": pa.array(["graph"], type=pa.string()),
                "value": pa.array(
                    [json.dumps(graph_attrs)], type=pa.string()
                ),
            }
        )
        self._copy_frame(self.meta_table, ["key", "value"], frame)

    def _load_meta(self):
        if self.meta_table not in {row[0] for row in self._table_names()}:
            return {}
        result = self.conn.execute(
            f"MATCH (m:{_quote(self.meta_table)}) "
            f"WHERE m.{_quote('key')} = 'graph' "
            f"RETURN m.{_quote('value')} AS {_quote('value')}"
        ).get_as_arrow()
        if result.num_rows == 0:
            return {}
        return json.loads(result.column("value")[0].as_py())

    # -- reads --

    def _table_names(self):
        result = self.conn.execute("CALL show_tables() RETURN *").get_as_arrow()
        return list(
            zip(result.column("name").to_pylist(), result.column("type").to_pylist())
        )

    def _table_columns(self, table):
        """Stored column names of ``table`` in DDL order.

        Relationship endpoints (``FROM``/``TO``) are not listed by
        ``table_info`` — only real properties are — so no filtering for
        them is needed.
        """
        result = self.conn.execute(
            f"CALL table_info({_literal(table)}) RETURN *"
        ).get_as_arrow()
        return result.column("name").to_pylist()

    def _load_table(self, table, endpoint):
        if table not in {name for name, _ in self._table_names()}:
            return None
        columns = self._table_columns(table)
        ordered = ORDER_COL in columns
        if endpoint:
            node, edge = _quote(self.node_table), _quote(table)
            src, tgt = _quote("source"), _quote("target")
            returns = [
                f"a.{_quote('node')} AS {src}",
                f"b.{_quote('node')} AS {tgt}",
            ]
            returns += [
                f"e.{_quote(c)} AS {_quote(c)}"
                for c in columns
                if c != ORDER_COL
            ]
            query = (
                f"MATCH (a:{node})-[e:{edge}]->(b:{node}) "
                f"RETURN {', '.join(returns)}"
            )
            if ordered:
                query += f" ORDER BY e.{_quote(ORDER_COL)}"
        else:
            returns = [f"n.{_quote(c)} AS {_quote(c)}" for c in columns if c != ORDER_COL]
            query = f"MATCH (n:{_quote(table)}) RETURN {', '.join(returns)}"
            if ordered:
                query += f" ORDER BY n.{_quote(ORDER_COL)}"
        result = self.conn.execute(query).get_as_arrow()
        # get_as_arrow may chunk columns; combine for clean tables.
        return result.combine_chunks() if result.num_columns else None
