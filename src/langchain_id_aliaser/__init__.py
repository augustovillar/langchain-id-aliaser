"""langchain-id-aliaser — reversible UUID/ID aliasing for LangChain & LangGraph.

The dependency-free core (:class:`IdAliaser`) is always importable. The adapters
(``wrap_model``, ``AliasMiddleware``, graph hooks) require ``langchain-core`` and
are imported lazily so the core works without it.
"""

from __future__ import annotations

from .core import IdAliaser, UnknownAliasError

__all__ = [
    "IdAliaser",
    "UnknownAliasError",
    "wrap_model",
    "AliasMiddleware",
    "make_alias_middleware",
    "alias_before_model",
    "restore_after_model",
    "make_model_node",
]

__version__ = "0.1.0"


def __getattr__(name: str):  # PEP 562 lazy adapter imports
    if name in ("wrap_model", "AliasedModel"):
        from . import model_wrapper

        return getattr(model_wrapper, name)
    if name in ("AliasMiddleware", "make_alias_middleware"):
        from . import middleware

        return getattr(middleware, name)
    if name in ("alias_before_model", "restore_after_model", "make_model_node"):
        from . import graph

        return getattr(graph, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
