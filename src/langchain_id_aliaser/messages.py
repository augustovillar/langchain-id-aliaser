"""Apply the aliaser to LangChain message objects.

These helpers duck-type ``BaseMessage`` (``.content`` plus an optional
``.tool_calls``) so they work across langchain-core versions without importing
specific message classes. They are the bridge every adapter shares.
"""

from __future__ import annotations

from typing import Any, Iterable, List

from .core import IdAliaser


def aliasify_message(aliaser: IdAliaser, message: Any) -> Any:
    """Return a copy of ``message`` with ids in its content and tool-call args
    replaced by aliases."""
    return _transform_message(aliaser, message, restore=False, strict=False)


def restore_message(aliaser: IdAliaser, message: Any, strict: bool = False) -> Any:
    """Return a copy of ``message`` with aliases in its content and tool-call
    args restored to the original ids."""
    return _transform_message(aliaser, message, restore=True, strict=strict)


def aliasify_messages(aliaser: IdAliaser, messages: Iterable[Any]) -> List[Any]:
    return [aliasify_message(aliaser, m) for m in messages]


def restore_messages(
    aliaser: IdAliaser, messages: Iterable[Any], strict: bool = False
) -> List[Any]:
    return [restore_message(aliaser, m, strict) for m in messages]


def _apply(aliaser: IdAliaser, obj: Any, restore: bool, strict: bool) -> Any:
    if restore:
        return aliaser.restore(obj, strict=strict)
    return aliaser.aliasify(obj)


def _transform_message(
    aliaser: IdAliaser, message: Any, restore: bool, strict: bool
) -> Any:
    update: dict = {}

    content = getattr(message, "content", None)
    if content is not None:
        update["content"] = _apply(aliaser, content, restore, strict)

    # AIMessage tool calls: list of {"name", "args", "id", ...}. Only args carry
    # user ids; names/ids are left alone.
    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        new_calls = []
        for call in tool_calls:
            new_call = dict(call)
            if "args" in new_call:
                new_call["args"] = _apply(aliaser, new_call["args"], restore, strict)
            new_calls.append(new_call)
        update["tool_calls"] = new_calls

    if not update:
        return message
    if hasattr(message, "model_copy"):  # pydantic v2 (langchain-core)
        return message.model_copy(update=update)
    if hasattr(message, "copy"):  # pydantic v1 fallback
        return message.copy(update=update)
    for key, val in update.items():  # last-resort plain object
        setattr(message, key, val)
    return message
