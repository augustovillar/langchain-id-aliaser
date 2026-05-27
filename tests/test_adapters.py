import pytest

pytest.importorskip("langchain_core")

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage  # noqa: E402

from langchain_id_aliaser import IdAliaser  # noqa: E402
from langchain_id_aliaser.messages import (  # noqa: E402
    aliasify_message,
    restore_message,
)
from langchain_id_aliaser.model_wrapper import wrap_model  # noqa: E402

UUID = "f47ac10b-58cc-4372-a567-0e02b2c3d479"


def test_aliasify_restore_human_message_content():
    a = IdAliaser()
    msg = HumanMessage(content=f"look up {UUID}")
    aliased = aliasify_message(a, msg)
    assert UUID not in aliased.content
    assert restore_message(a, aliased).content == msg.content


def test_tool_call_args_aliased_and_restored():
    a = IdAliaser()
    msg = AIMessage(
        content="",
        tool_calls=[{"name": "get_user", "args": {"user_id": UUID}, "id": "call_1"}],
    )
    aliased = aliasify_message(a, msg)
    assert aliased.tool_calls[0]["args"]["user_id"] != UUID
    restored = restore_message(a, aliased)
    assert restored.tool_calls[0]["args"]["user_id"] == UUID


def test_tool_message_content_roundtrip():
    a = IdAliaser()
    msg = ToolMessage(content=f"found {UUID}", tool_call_id="call_1")
    aliased = aliasify_message(a, msg)
    assert UUID not in aliased.content
    assert restore_message(a, aliased).content == msg.content


class FakeModel:
    """Echoes the aliased id back inside a tool call, as a real model would."""

    def __init__(self):
        self.seen = None

    def invoke(self, messages, *a, **k):
        self.seen = messages
        # The model only ever saw the alias; echo it into a tool call.
        alias = messages[0].content.split()[-1]
        return AIMessage(
            content=f"calling tool for {alias}",
            tool_calls=[{"name": "fetch", "args": {"id": alias}, "id": "c1"}],
        )

    async def ainvoke(self, messages, *a, **k):
        return self.invoke(messages, *a, **k)


def test_wrap_model_model_never_sees_uuid_but_caller_gets_it_back():
    fake = FakeModel()
    aliaser = IdAliaser()
    model = wrap_model(fake, aliaser)

    result = model.invoke([HumanMessage(content=f"user {UUID}")])

    # The underlying model saw only the alias.
    assert UUID not in fake.seen[0].content
    # The caller gets the real UUID back in content and tool-call args.
    assert UUID in result.content
    assert result.tool_calls[0]["args"]["id"] == UUID


def test_wrap_model_async():
    import asyncio

    fake = FakeModel()
    model = wrap_model(fake, IdAliaser())

    async def run():
        return await model.ainvoke([HumanMessage(content=f"user {UUID}")])

    result = asyncio.run(run())
    assert result.tool_calls[0]["args"]["id"] == UUID
