from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from harness.context import estimate_tokens, trim_history, truncate_tool_output
from harness.loop import run
from harness.testing import ScriptedModel


def test_estimate_tokens():
    assert estimate_tokens([]) == 0
    assert estimate_tokens([HumanMessage("a" * 8), AIMessage("b" * 9)]) == 2 + 3


def test_short_tool_output_is_unchanged():
    assert truncate_tool_output("hello", max_chars=10) == "hello"


def test_long_tool_output_keeps_head_and_tail():
    text = "H" * 500 + "x" * 9000 + "T" * 500
    out = truncate_tool_output(text, max_chars=1000)
    assert out.startswith("H" * 500) and out.endswith("T" * 500)
    assert "9000" in out and "truncated" in out
    assert len(out) < 1100


def tool_round(i, size):
    call = AIMessage("", tool_calls=[{"name": "read_file", "args": {"path": f"f{i}"}, "id": f"c{i}"}])
    return [call, ToolMessage("x" * size, tool_call_id=f"c{i}")]


def history(rounds, size=100):
    msgs = [HumanMessage("do the task")]
    for i in range(rounds):
        msgs += tool_round(i, size)
    return msgs


def test_history_under_budget_is_unchanged():
    msgs = history(2)
    assert trim_history(msgs, 1000) == msgs


def test_trim_drops_oldest_and_keeps_task_and_latest():
    msgs = history(5)  # 1 + 10 messages, 5 x 25 tokens of tool output
    trimmed = trim_history(msgs, 60)
    assert len(msgs) == 11, "the input list must not change"
    assert trimmed[0] is msgs[0] and trimmed[-1] is msgs[-1]
    assert estimate_tokens(trimmed) <= 60
    assert trimmed[1:] == msgs[-len(trimmed) + 1:], "keep a contiguous run of the most recent messages"


def test_trim_never_leaves_orphan_tool_results():
    for budget in range(0, 140, 5):
        trimmed = trim_history(history(5), budget)
        requested = {c["id"] for m in trimmed if isinstance(m, AIMessage) for c in m.tool_calls}
        for m in trimmed:
            if isinstance(m, ToolMessage):
                assert m.tool_call_id in requested, f"orphan tool result at budget {budget}"


def test_trim_keeps_first_and_last_even_over_budget():
    msgs = [HumanMessage("a" * 400), AIMessage("b" * 400)]
    assert trim_history(msgs, 10) == msgs


def test_loop_truncates_big_tool_outputs(tmp_path):
    big = tmp_path / "big.txt"
    big.write_text("y" * 50_000)
    call = AIMessage("", tool_calls=[{"name": "read_file", "args": {"path": str(big)}, "id": "c1"}])
    model = ScriptedModel(call, AIMessage("done"))
    run(model, "read it")
    sent = model.seen[1][-1]
    assert isinstance(sent, ToolMessage) and len(sent.content) < 2100 and "truncated" in sent.content


def test_loop_trims_what_the_model_sees(tmp_path):
    files = []
    for i in range(6):
        f = tmp_path / f"f{i}.txt"
        f.write_text(str(i) * 400)  # 100 tokens each
        files.append(f)
    calls = [AIMessage("", tool_calls=[{"name": "read_file", "args": {"path": str(f)}, "id": f"c{i}"}]) for i, f in enumerate(files)]
    model = ScriptedModel(*calls, AIMessage("done"))
    assert run(model, "read them all", max_context_tokens=250) == "done"
    for seen in model.seen:
        assert isinstance(seen[0], HumanMessage)
        assert estimate_tokens(seen) <= 250
    assert "5" * 400 in model.seen[-1][-1].content, "the latest tool result must reach the model"
