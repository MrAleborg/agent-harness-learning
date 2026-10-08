from langchain_core.messages import AIMessage

from harness.loop import run
from harness.testing import ScriptedModel



def test_loop_runs_tool_then_answers():
    model = ScriptedModel(
        AIMessage("", tool_calls=[{"name": "add", "args": {"a": 2, "b": 3}, "id": "c1"}]),
        AIMessage("5"),
    )
    assert run(model, "2+3?") == "5"
    assert model.seen[1][-1].content == "5"  # tool result was fed back
