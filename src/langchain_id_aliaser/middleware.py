"""Adapter 2 — agent middleware for LangChain's ``wrap_model_call`` API.

Targets the agent-middleware interface in recent ``langchain`` releases. The
base class is imported lazily so this module is importable even when the running
version predates it; in that case :func:`make_alias_middleware` raises a clear
error.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Callable, Optional

from .core import IdAliaser
from .messages import aliasify_messages, restore_message, restore_messages


def _copy_with(obj: Any, **update: Any) -> Any:
    """Return a copy of ``obj`` with fields updated, supporting pydantic models,
    dataclasses (incl. frozen), and plain mutable objects."""
    if hasattr(obj, "model_copy"):
        return obj.model_copy(update=update)
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.replace(obj, **update)
    for key, val in update.items():
        setattr(obj, key, val)
    return obj


def _base_middleware_class() -> Any:
    try:
        from langchain.agents.middleware import AgentMiddleware  # type: ignore

        return AgentMiddleware
    except Exception:  # pragma: no cover - depends on installed version
        try:
            from langchain.agents.middleware.types import (  # type: ignore
                AgentMiddleware,
            )

            return AgentMiddleware
        except Exception:
            return object


_AgentMiddleware = _base_middleware_class()


class AliasMiddleware(_AgentMiddleware):  # type: ignore[misc, valid-type]
    """Aliases ids in the messages sent to the model and restores them in the
    model's response (content and outgoing tool-call args).

    Attach to an agent like any other middleware. Uses a fresh
    :class:`IdAliaser` per call by default (stateless, deterministic-hash). Pass
    a shared ``aliaser`` to keep one mapping across the whole run.
    """

    def __init__(
        self,
        aliaser: Optional[IdAliaser] = None,
        *,
        strict: bool = False,
        **aliaser_kwargs: Any,
    ) -> None:
        if _AgentMiddleware is not object:
            super().__init__()
        self._shared_aliaser = aliaser
        self._aliaser_kwargs = aliaser_kwargs
        self.strict = strict

    def _aliaser(self) -> IdAliaser:
        return self._shared_aliaser or IdAliaser(**self._aliaser_kwargs)

    def _alias_request(self, request: Any, aliaser: IdAliaser) -> Any:
        messages = getattr(request, "messages", None)
        if messages is None:
            return request
        return _copy_with(request, messages=aliasify_messages(aliaser, messages))

    def _restore_response(self, response: Any, aliaser: IdAliaser) -> Any:
        # A response may be a bare AIMessage, or a ModelResponse with .result
        # (a list of messages).
        result = getattr(response, "result", None)
        if result is not None:
            restored = (
                restore_messages(aliaser, result, self.strict)
                if isinstance(result, list)
                else restore_message(aliaser, result, self.strict)
            )
            return _copy_with(response, result=restored)
        if hasattr(response, "tool_calls") or hasattr(response, "content"):
            return restore_message(aliaser, response, strict=self.strict)
        return response

    def wrap_model_call(self, request: Any, handler: Callable[[Any], Any]) -> Any:
        aliaser = self._aliaser()
        response = handler(self._alias_request(request, aliaser))
        return self._restore_response(response, aliaser)

    async def awrap_model_call(
        self, request: Any, handler: Callable[[Any], Any]
    ) -> Any:
        aliaser = self._aliaser()
        response = await handler(self._alias_request(request, aliaser))
        return self._restore_response(response, aliaser)


def make_alias_middleware(*args: Any, **kwargs: Any) -> AliasMiddleware:
    """Build an :class:`AliasMiddleware`, erroring clearly if the installed
    ``langchain`` lacks the agent-middleware API."""
    if _AgentMiddleware is object:
        raise ImportError(
            "AliasMiddleware needs the agent-middleware API from a recent "
            "`langchain` release (langchain.agents.middleware.AgentMiddleware). "
            "Upgrade langchain, or use wrap_model / the graph hooks instead."
        )
    return AliasMiddleware(*args, **kwargs)
