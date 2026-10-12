"""Persistence tests: same ``arrow``-backend graphs, parquet or Ladybug store."""

import datetime

import pytest

pa = pytest.importorskip("pyarrow")
pytest.importorskip("ladybug")

import networkx as nx
from nxarrow.backend import persist
from nxarrow.graph import ArrowDiGraph
from nxarrow.parquet import ParquetStorageProvider

from nxladybug import LadybugStorageProvider


@pytest.fixture
def typed_graph():
    g = nx.DiGraph(backend="arrow")
    g.add_node("alice", node_type="user", age=30)
    g.add_node("bob", node_type="user", age=25)
    g.add_node("post1", node_type="post", text="hello")
    g.add_edge("alice", "post1", edge_type="authored")
    g.add_edge("bob", "post1", edge_type="liked", weight=0.9)
    g.add_edge("alice", "bob", edge_type="follows")
    return g


def _assert_graphs_equal(original, restored):
    assert restored.node_types == original.node_types
    assert restored.edge_types == original.edge_types
    assert restored.nodes_table().num_rows == original.nodes_table().num_rows
    assert restored.edges_table().num_rows == original.edges_table().num_rows
    for node in original.nodes:
        assert restored.has_node(node)
        assert restored.node_attrs(node) == original.node_attrs(node)
    for u, v in original.edges:
        assert restored.has_edge(u, v)
        assert restored.edge_attrs(u, v) == original.edge_attrs(u, v)
    assert restored.graph == original.graph


def _row_count(provider, query):
    return provider.conn.execute(query).get_as_arrow().column(0)[0].as_py()


# -- Ladybug store --


def test_ladybug_provider_roundtrip(typed_graph):
    typed_graph.graph["name"] = "roundtrip"
    provider = LadybugStorageProvider()
    try:
        counts = persist(typed_graph, provider, graph_attrs=typed_graph.graph)
        assert counts == {provider.node_table: 3, provider.edge_table: 3}
        names = {name for name, _ in provider._table_names()}
        assert {provider.node_table, provider.edge_table, provider.meta_table} <= names

        restored = provider.load_graph()
        _assert_graphs_equal(typed_graph, restored)
        # insertion order survives the round-trip via the sequence column
        assert restored.nodes == typed_graph.nodes
        assert restored.edges == typed_graph.edges
    finally:
        provider.close()


def test_ladybug_provider_roundtrip_untyped_and_empty():
    provider = LadybugStorageProvider()
    try:
        g = nx.DiGraph(backend="arrow")
        g.add_edge("a", "b", weight=1.5)
        persist(g, provider)
        restored = provider.load_graph()
        _assert_graphs_equal(g, restored)
        assert restored.nodes_table().column_names == ["node"]

        persist(nx.DiGraph(backend="arrow"), provider)
        empty = provider.load_graph()
        assert empty.number_of_nodes() == 0
        assert empty.number_of_edges() == 0
    finally:
        provider.close()


def test_ladybug_provider_repersist_drops_stale_rows(typed_graph):
    provider = LadybugStorageProvider()
    try:
        persist(typed_graph, provider)
        assert _row_count(provider, "MATCH (n:NxNode) RETURN count(*)") == 3
        assert _row_count(provider, "MATCH ()-[e:NxEdge]->() RETURN count(*)") == 3

        typed_graph.remove_edge("alice", "bob")  # drops the only "follows" edge
        typed_graph.remove_edge("bob", "post1")  # drops the only "liked" edge
        persist(typed_graph, provider)
        assert _row_count(provider, "MATCH ()-[e:NxEdge]->() RETURN count(*)") == 1
        _assert_graphs_equal(typed_graph, provider.load_graph())
    finally:
        provider.close()


def test_ladybug_provider_file_backed_reopen(tmp_path, typed_graph):
    typed_graph.graph["name"] = "file-roundtrip"
    db_path = tmp_path / "graph.lbdb"
    provider = LadybugStorageProvider(db_path)
    persist(typed_graph, provider, graph_attrs=typed_graph.graph)
    provider.close()

    restored = LadybugStorageProvider.load(db_path)
    _assert_graphs_equal(typed_graph, restored)
    assert restored.nodes == typed_graph.nodes


def test_ladybug_roundtrip_preserves_int_keys():
    g = nx.DiGraph(backend="arrow")
    g.add_edge(0, 1, weight=2.5)
    g.add_edge(1, 2)
    provider = LadybugStorageProvider()
    try:
        persist(g, provider)
        restored = provider.load_graph()
        assert restored.nodes == [0, 1, 2]
        assert restored.edges == [(0, 1), (1, 2)]
        assert restored.edge_attrs(0, 1) == {"weight": 2.5}
        assert restored.nodes_table().schema.field("node").type == pa.int64()
    finally:
        provider.close()


def test_ladybug_roundtrip_native_key_types():
    cases = [
        ([0, 1, 2], pa.int64()),
        ([0.5, 1.5, 2.5], pa.float64()),
        (["a", "b", "c"], pa.string()),
        (
            [
                datetime.date(2024, 1, 1),
                datetime.date(2024, 1, 2),
                datetime.date(2024, 1, 3),
            ],
            pa.date32(),
        ),
        (
            [
                datetime.datetime(2024, 1, 1, 1, 2),
                datetime.datetime(2024, 1, 2),
                datetime.datetime(2024, 1, 3, 4, 5),
            ],
            pa.timestamp("us"),
        ),
    ]
    for keys, expected_type in cases:
        g = nx.DiGraph(backend="arrow")
        g.add_edge(keys[0], keys[1], weight=1.0)
        g.add_edge(keys[1], keys[2])
        provider = LadybugStorageProvider()
        try:
            persist(g, provider)
            restored = provider.load_graph()
            assert restored.nodes == keys
            assert restored.edges == [(keys[0], keys[1]), (keys[1], keys[2])]
            assert restored.edge_attrs(keys[0], keys[1]) == {"weight": 1.0}
            assert restored.nodes_table().schema.field("node").type == expected_type
        finally:
            provider.close()


def test_ladybug_lazy_persist_roundtrip(typed_graph):
    # index-free adopted tables persist without materializing the index
    nodes, edges = typed_graph.nodes_table(), typed_graph.edges_table()
    lazy = ArrowDiGraph.from_arrow(nodes, edges, index=False)
    provider = LadybugStorageProvider()
    try:
        counts = persist(lazy, provider)
        assert lazy._index_built is False
        assert sum(counts.values()) == 6
        _assert_graphs_equal(typed_graph, provider.load_graph())
    finally:
        provider.close()


# -- Parquet store (same graphs, alternative choice at persist() time) --


def test_parquet_provider_roundtrip(tmp_path, typed_graph):
    typed_graph.graph["name"] = "roundtrip"
    provider = ParquetStorageProvider(tmp_path / "store")
    counts = persist(typed_graph, provider, graph_attrs=typed_graph.graph)
    assert sum(counts.values()) == 6
    assert (tmp_path / "store" / "manifest.json").exists()

    restored = provider.load_graph()
    _assert_graphs_equal(typed_graph, restored)
    # per-type files: 2 node types + 3 edge types
    assert len(list((tmp_path / "store" / "nodes").glob("*.parquet"))) == 2
    assert len(list((tmp_path / "store" / "edges").glob("*.parquet"))) == 3


def test_parquet_provider_roundtrip_untyped_and_empty(tmp_path):
    provider = ParquetStorageProvider(tmp_path / "store")
    g = nx.DiGraph(backend="arrow")
    g.add_edge("a", "b", weight=1.5)
    persist(g, provider)
    restored = provider.load_graph()
    _assert_graphs_equal(g, restored)
    assert restored.nodes_table().column_names == ["node"]

    persist(nx.DiGraph(backend="arrow"), provider)
    empty = provider.load_graph()
    assert empty.number_of_nodes() == 0
    assert empty.number_of_edges() == 0


def test_parquet_provider_repersist_drops_stale_files(tmp_path, typed_graph):
    provider = ParquetStorageProvider(tmp_path / "store")
    persist(typed_graph, provider)
    assert len(list((tmp_path / "store" / "edges").glob("*.parquet"))) == 3

    typed_graph.remove_edge("alice", "bob")  # drops the only "follows" edge
    typed_graph.remove_edge("bob", "post1")  # drops the only "liked" edge
    persist(typed_graph, provider)
    assert len(list((tmp_path / "store" / "edges").glob("*.parquet"))) == 1
    _assert_graphs_equal(typed_graph, provider.load_graph())


def test_parquet_roundtrip_preserves_int_keys(tmp_path):
    g = nx.DiGraph(backend="arrow")
    g.add_edge(0, 1, weight=2.5)
    g.add_edge(1, 2)
    provider = ParquetStorageProvider(tmp_path / "store")
    persist(g, provider)
    restored = provider.load_graph()
    assert restored.nodes == [0, 1, 2]
    assert restored.edges == [(0, 1), (1, 2)]
    assert restored.edge_attrs(0, 1) == {"weight": 2.5}
    assert restored.nodes_table().schema.field("node").type == pa.int64()


# -- the choice itself: one graph, either store --


def test_same_graph_persists_to_both_stores(tmp_path, typed_graph):
    typed_graph.graph["name"] = "either-store"
    ladybug = LadybugStorageProvider()
    parquet = ParquetStorageProvider(tmp_path / "store")
    try:
        persist(typed_graph, ladybug, graph_attrs=typed_graph.graph)
        persist(typed_graph, parquet, graph_attrs=typed_graph.graph)
        _assert_graphs_equal(typed_graph, ladybug.load_graph())
        _assert_graphs_equal(typed_graph, parquet.load_graph())
    finally:
        ladybug.close()
