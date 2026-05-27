"""Adapter 3 — pre/post hooks for raw ``StateGraph`` graphs.

These are plain functions operating on a messages list, so you can drop them
around your model node without depending on any particular graph-construction
API. Pair them so the model sees aliases and the rest of the graph keeps real
ids.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

from .core import IdAliaser
from .messages import aliasify_messages, restore_message


def alias_before_model(
    messages: List[Any], aliaser: Optional[IdAliaser] = None, **aliaser_kwargs: Any
) -> Tuple[List[Any], IdAliaser]:
    """Alias ids in ``messages`` before the model node runs.

    Returns the aliased messages and the aliaser used; keep the aliaser to pass
    to :func:`restore_after_model` so the mapping matches.
    """
    aliaser = aliaser or IdAliaser(**aliaser_kwargs)
    return aliasify_messages(aliaser, messages), aliaser


def restore_after_model(
    message: Any, aliaser: IdAliaser, *, strict: bool = False
) -> Any:
    """Restore ids in the model's output message (content + tool-call args)
    before downstream nodes/tools consume it."""
    return restore_message(aliaser, message, strict=strict)


def make_model_node(model: Any, aliaser: Optional[IdAliaser] = None, **kw: Any):
    """Build a ``StateGraph`` node that wraps ``model`` with aliasing.

    The returned callable expects state with a ``"messages"`` list and returns
    ``{"messages": [restored_response]}`` — the conventional message-reducer
    shape. A shared ``aliaser`` keeps the mapping stable across turns.
    """
    shared = aliaser

    def node(state: dict) -> dict:
        local = shared or IdAliaser(**kw)
        aliased, local = alias_before_model(state["messages"], local)
        response = model.invoke(aliased)
        return {"messages": [restore_after_model(response, local)]}

    return node
