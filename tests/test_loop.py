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


def test_stream_reports_events_in_order():
    model = ScriptedModel(
        AIMessage("", tool_calls=[{"name": "add", "args": {"a": 2, "b": 3}, "id": "c1"}]),
        AIMessage("5"),
    )
    events = []
    assert run(model, "2+3?", on_event=events.append) == "5"
    kinds = [e["type"] for e in events]
    assert kinds == ["model_reply", "tool_result", "token", "model_reply"]
    assert events[1] == {"type": "tool_result", "name": "add", "content": "5", "status": "success"}
    assert "".join(e["text"] for e in events if e["type"] == "token") == "5"
