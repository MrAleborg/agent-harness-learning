from langchain_core.messages import AIMessage, ToolMessage

from harness.loop import run
from harness.testing import ScriptedModel



def call(name, args):
    return AIMessage("", tool_calls=[{"name": name, "args": args, "id": "c1"}])


def last_tool_message(model):
    msg = model.seen[1][-1]
    assert isinstance(msg, ToolMessage)
    return msg


def test_unknown_tool_is_reported_to_the_model():
    model = ScriptedModel(call("teleport", {}), AIMessage("sorry"))
    assert run(model, "go") == "sorry"
    msg = last_tool_message(model)
    assert msg.status == "error"
    assert "teleport" in msg.content


def test_failing_tool_is_reported_to_the_model():
    model = ScriptedModel(call("add", {"a": "not a number", "b": 1}), AIMessage("retrying"))
    assert run(model, "add") == "retrying"
    assert last_tool_message(model).status == "error"
