"""LadybugDB storage provider for :mod:`nxarrow` columnar graphs.

``nx-arrow`` owns the Arrow-backed graph and the ``arrow`` NetworkX
backend; this package only adds persistence into LadybugDB. Build graphs
the usual way and choose a store at :func:`persist` time::

    import networkx as nx
    from nxarrow.parquet import ParquetStorageProvider
    from nxladybug import LadybugStorageProvider, persist

    g = nx.DiGraph([(0, 1), (1, 2)], backend="arrow")
    persist(g, LadybugStorageProvider("graph.lbdb"), graph_attrs=g.graph)
    # ... or persist(g, ParquetStorageProvider("store"), graph_attrs=g.graph)
"""

from nxarrow.backend import (
    get_storage_provider,
    list_storage_providers,
    persist,
    register_storage_provider,
    unregister_storage_provider,
)

from .ladybug_store import LadybugStorageProvider

__all__ = [
    "LadybugStorageProvider",
    "persist",
    "register_storage_provider",
    "unregister_storage_provider",
    "get_storage_provider",
    "list_storage_providers",
]
