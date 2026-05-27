"""End-to-end tests inside a real ``create_agent`` loop.

These drive a full agent run with a scripted tool-calling chat model and a real
tool. The assertions prove the contract end to end:

* the chat model only ever sees short aliases, never the raw UUIDs;
* the tool, when executed by the agent, receives the *real* UUID;
* the final answer returned to the caller contains the real UUID.
"""

import re
from typing import Any, List, Optional

import pytest

pytest.importorskip("langchain.agents")

from langchain.agents import create_agent  # noqa: E402
from langchain_core.callbacks import CallbackManagerForLLMRun  # noqa: E402
from langchain_core.language_models import BaseChatModel  # noqa: E402
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage  # noqa: E402
from langchain_core.outputs import ChatGeneration, ChatResult  # noqa: E402
from langchain_core.tools import tool  # noqa: E402

from langchain_id_aliaser import AliasMiddleware, IdAliaser  # noqa: E402

USER_UUID = "f47ac10b-58cc-4372-a567-0e02b2c3d479"
ACCOUNT_UUID = "550e8400-e29b-41d4-a716-446655440000"
UUID_RE = re.compile(
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
)


class ScriptedToolCaller(BaseChatModel):
    """A chat model that:

    1. on first turn, emits a tool call echoing the id token it sees in the last
       human message;
    2. on the second turn, returns a final answer echoing the id token it sees in
       the tool result.

    It records every message content it observes in ``seen_contents`` so tests
    can assert the model never saw a raw UUID.
    """

    seen_contents: List[str] = []
    tool_name: str = "lookup_account"

    model_config = {"arbitrary_types_allowed": True}

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        for m in messages:
            if isinstance(m.content, str):
                self.seen_contents.append(m.content)

        last = messages[-1]
        token = _last_token(last.content)

        if isinstance(last, ToolMessage):
            msg = AIMessage(content=f"The account for {token} is active.")
        else:
            msg = AIMessage(
                content="",
                tool_calls=[
                    {"name": self.tool_name, "args": {"user_id": token}, "id": "c1"}
                ],
            )
        return ChatResult(generations=[ChatGeneration(message=msg)])

    def bind_tools(self, tools: Any, **kwargs: Any) -> "ScriptedToolCaller":
        return self  # scripted; tool schema is irrelevant

    @property
    def _llm_type(self) -> str:
        return "scripted-tool-caller"


def _last_token(content: Any) -> str:
    text = content if isinstance(content, str) else str(content)
    return text.split()[-1].rstrip(".") if text.split() else ""


def _build_agent(model, tool_seen):
    @tool
    def lookup_account(user_id: str) -> str:
        """Look up an account by user id."""
        tool_seen.append(user_id)
        return f"account {ACCOUNT_UUID} owned by {user_id}"

    return create_agent(
        model, [lookup_account], middleware=[AliasMiddleware(IdAliaser())]
    )


def test_agent_tool_receives_real_uuid_model_sees_only_alias():
    model = ScriptedToolCaller(seen_contents=[])
    tool_seen: List[str] = []
    agent = _build_agent(model, tool_seen)

    result = agent.invoke(
        {"messages": [("user", f"Check the account for user {USER_UUID}")]}
    )

    # 1. The model never saw a raw UUID in any message it processed.
    assert model.seen_contents, "model was never called"
    assert not any(UUID_RE.search(c) for c in model.seen_contents), model.seen_contents

    # 2. The tool actually executed and received the real UUID.
    assert tool_seen == [USER_UUID]

    # 3. The final message back to the caller contains the real UUID.
    final = result["messages"][-1]
    assert USER_UUID in final.content


def test_agent_handles_uuid_in_tool_result_on_second_turn():
    # The tool returns a *different* UUID; the model must see it aliased too.
    model = ScriptedToolCaller(seen_contents=[])
    tool_seen: List[str] = []
    agent = _build_agent(model, tool_seen)

    agent.invoke({"messages": [("user", f"Check account for {USER_UUID}")]})

    # The ToolMessage carried ACCOUNT_UUID; the model must never have seen it raw.
    assert not any(UUID_RE.search(c) for c in model.seen_contents)
    # And the model saw *some* tool-result content on its second turn.
    assert len(model.seen_contents) > 2


def test_agent_without_middleware_leaks_uuid_to_model():
    # Control: prove the middleware is what hides the UUID, not the harness.
    model = ScriptedToolCaller(seen_contents=[])
    tool_seen: List[str] = []

    @tool
    def lookup_account(user_id: str) -> str:
        """Look up an account by user id."""
        tool_seen.append(user_id)
        return "ok"

    agent = create_agent(model, [lookup_account])  # no middleware
    agent.invoke({"messages": [("user", f"Check {USER_UUID}")]})

    assert any(UUID_RE.search(c) for c in model.seen_contents)
    assert tool_seen == [USER_UUID]
