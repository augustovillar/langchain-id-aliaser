import dataclasses

import pytest

# AliasMiddleware needs the agent-middleware API from langchain v1+.
pytest.importorskip("langchain.agents.middleware")

from langchain.agents.middleware import AgentMiddleware  # noqa: E402
from langchain.agents.middleware.types import (  # noqa: E402
    ModelRequest,
    ModelResponse,
)
from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402

from langchain_id_aliaser import AliasMiddleware, IdAliaser  # noqa: E402

UUID = "f47ac10b-58cc-4372-a567-0e02b2c3d479"


def _make_request(messages):
    fields = {f.name: None for f in dataclasses.fields(ModelRequest)}
    fields["messages"] = messages
    return ModelRequest(**fields)


def test_is_real_agent_middleware_subclass():
    assert isinstance(AliasMiddleware(), AgentMiddleware)


def test_middleware_roundtrip_with_real_model_types():
    aliaser = IdAliaser()
    mw = AliasMiddleware(aliaser)
    seen = {}

    def handler(request):
        seen["content"] = request.messages[0].content
        alias = request.messages[0].content.split()[-1]
        return ModelResponse(
            result=[
                AIMessage(
                    content="",
                    tool_calls=[{"name": "f", "args": {"id": alias}, "id": "c1"}],
                )
            ],
            structured_response=None,
        )

    request = _make_request([HumanMessage(content=f"user {UUID}")])
    response = mw.wrap_model_call(request, handler)

    assert UUID not in seen["content"]  # model never saw the UUID
    assert response.result[0].tool_calls[0]["args"]["id"] == UUID  # restored
