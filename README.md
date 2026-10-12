# nx-ladybug

LadybugDB storage provider for
[nx-arrow](https://github.com/Ladybug-Memory/nx-arrow) columnar graphs.
`nx-arrow` owns the Arrow-backed graph and the `arrow` NetworkX backend;
this package only adds persistence into
[LadybugDB](https://ladybugdb.com/). Build and analyze graphs with
`nx-arrow` everywhere — the only `nxladybug` import is the persist step.

## Install

```bash
pip install nx-arrow nx-ladybug
```

For development:

```bash
pip install -e ".[test]"
```

## Use

```python
import networkx as nx
from nxladybug import LadybugStorageProvider, persist

g = nx.DiGraph([(0, 1), (1, 2)], backend="arrow")

nx.out_degree_centrality(g, backend="arrow")
list(nx.topological_sort(g, backend="arrow"))
```

## Persist

`persist()` bulk-loads per-type frames through a storage provider — pick
LadybugDB or Parquet:

```python
from nxarrow.parquet import ParquetStorageProvider
from nxladybug import LadybugStorageProvider, persist

# LadybugDB: one unified node table (NxNode) + one relationship table
# (NxEdge) + a meta table (NxMeta). Each frame is registered as an Arrow
# staging table and bulk-copied natively — COPY <table> FROM (MATCH ...) —
# so endpoints resolve through the node primary key inside the copy.
provider = LadybugStorageProvider("graph.lbdb")  # or ":memory:"
persist(g, provider, graph_attrs=g.graph)
restored = provider.load_graph()  # full round-trip

# ... or Parquet: one file per type plus a manifest.json
parquet = ParquetStorageProvider("graph_store")
persist(g, parquet, graph_attrs=g.graph)
restored = parquet.load_graph()
```

The Ladybug provider drops and recreates its tables on every persist, so
re-persisting a smaller graph leaves no stale rows. `:memory:` databases
live as long as the provider; file paths reopen with
`LadybugStorageProvider.load(path)`.

Custom stores implement `store_nodes(frames)` / `store_edges(frames)` and
register with `nxarrow.register_storage_provider`.

## Develop

```bash
pytest
```
