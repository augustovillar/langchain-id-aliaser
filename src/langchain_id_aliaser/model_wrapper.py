"""Adapter 1 — wrap a chat model for single ``invoke`` / ``ainvoke`` calls."""

from __future__ import annotations

from typing import Any, List, Optional

from .core import IdAliaser
from .messages import aliasify_messages, restore_message


def wrap_model(
    model: Any,
    aliaser: Optional[IdAliaser] = None,
    *,
    strict: bool = False,
    **aliaser_kwargs: Any,
) -> "AliasedModel":
    """Wrap ``model`` so ids are aliased on the way to the provider and restored
    in the response (including outgoing tool-call args).

    Pass an existing ``aliaser`` to share a mapping across calls, or let one be
    created from ``aliaser_kwargs`` (e.g. ``mode="ordinal"``).
    """
    return AliasedModel(model, aliaser or IdAliaser(**aliaser_kwargs), strict=strict)


class AliasedModel:
    """A thin proxy around a chat model. Delegates everything except
    ``invoke``/``ainvoke``, which round-trip the aliaser."""

    def __init__(self, model: Any, aliaser: IdAliaser, *, strict: bool = False):
        self._model = model
        self.aliaser = aliaser
        self.strict = strict

    def _prep(self, input: Any) -> Any:
        if isinstance(input, list):
            return aliasify_messages(self.aliaser, input)
        # A bare string prompt or PromptValue — alias whatever text we can reach.
        if isinstance(input, str):
            return self.aliaser.aliasify(input)
        return input

    def invoke(self, input: Any, *args: Any, **kwargs: Any) -> Any:
        result = self._model.invoke(self._prep(input), *args, **kwargs)
        return restore_message(self.aliaser, result, strict=self.strict)

    async def ainvoke(self, input: Any, *args: Any, **kwargs: Any) -> Any:
        result = await self._model.ainvoke(self._prep(input), *args, **kwargs)
        return restore_message(self.aliaser, result, strict=self.strict)

    # Keep the wrapper usable wherever the raw model was (bind_tools, etc.).
    def bind_tools(self, *args: Any, **kwargs: Any) -> "AliasedModel":
        return AliasedModel(
            self._model.bind_tools(*args, **kwargs), self.aliaser, strict=self.strict
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._model, name)
