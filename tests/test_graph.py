import pytest

pytest.importorskip("langchain_core")

from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402

from langchain_id_aliaser import (  # noqa: E402
    IdAliaser,
    alias_before_model,
    make_model_node,
    restore_after_model,
)

UUID = "f47ac10b-58cc-4372-a567-0e02b2c3d479"


def test_before_after_hooks_roundtrip():
    messages = [HumanMessage(content=f"user {UUID}")]
    aliased, aliaser = alias_before_model(messages)
    assert UUID not in aliased[0].content

    alias = aliased[0].content.split()[-1]
    model_out = AIMessage(
        content="", tool_calls=[{"name": "f", "args": {"id": alias}, "id": "c1"}]
    )
    restored = restore_after_model(model_out, aliaser)
    assert restored.tool_calls[0]["args"]["id"] == UUID


class FakeModel:
    def invoke(self, messages, *a, **k):
        self.seen = messages
        alias = messages[0].content.split()[-1]
        return AIMessage(
            content="", tool_calls=[{"name": "f", "args": {"id": alias}, "id": "c1"}]
        )


def test_make_model_node():
    fake = FakeModel()
    node = make_model_node(fake, IdAliaser())
    out = node({"messages": [HumanMessage(content=f"user {UUID}")]})
    assert UUID not in fake.seen[0].content
    assert out["messages"][0].tool_calls[0]["args"]["id"] == UUID
