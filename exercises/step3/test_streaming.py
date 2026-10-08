import json

from langchain_core.messages import AIMessage

from harness.loop import run
from harness.testing import ScriptedModel
from harness.tracing import JsonlTracer


def add_call():
    return AIMessage("Let me add.", tool_calls=[{"name": "add", "args": {"a": 2, "b": 3}, "id": "c1"}])


def test_tokens_are_streamed():
    events = []
    assert run(ScriptedModel(AIMessage("Hello there world")), "hi", on_event=events.append) == "Hello there world"
    tokens = [e["text"] for e in events if e["type"] == "token"]
    assert len(tokens) == 3, "expected one token event per streamed chunk"
    assert "".join(tokens) == "Hello there world"


def test_streamed_tool_call_is_rebuilt_and_run():
    model = ScriptedModel(add_call(), AIMessage("5"))
    assert run(model, "2+3?") == "5"
    assert model.seen[1][-1].content == "5"


def test_events_tell_the_story_of_the_run():
    events = []
    run(ScriptedModel(add_call(), AIMessage("5")), "2+3?", on_event=events.append)
    story = [e for e in events if e["type"] != "token"]
    assert [e["type"] for e in story] == ["model_reply", "tool_result", "model_reply"]
    assert story[0]["content"] == "Let me add."
    assert story[0]["tool_calls"][0]["args"] == {"a": 2, "b": 3}
    assert story[1] == {"type": "tool_result", "name": "add", "content": "5", "status": "success"}
    assert story[2]["content"] == "5" and not story[2]["tool_calls"]


def test_tool_errors_are_traced():
    events = []
    bad = AIMessage("", tool_calls=[{"name": "teleport", "args": {}, "id": "c1"}])
    run(ScriptedModel(bad, AIMessage("sorry")), "go", on_event=events.append)
    result = next(e for e in events if e["type"] == "tool_result")
    assert result["name"] == "teleport" and result["status"] == "error"


def test_jsonl_tracer_writes_one_line_per_event(tmp_path):
    path = tmp_path / "trace.jsonl"
    events = []
    tracer = JsonlTracer(path)
    run(ScriptedModel(add_call(), AIMessage("5")), "2+3?", on_event=lambda e: (events.append(e), tracer(e)))
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    assert [line["type"] for line in lines] == [e["type"] for e in events]
    assert all(isinstance(line["time"], float) for line in lines)
