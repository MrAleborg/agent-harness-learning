# agent-harness-learning

Learning to build a full AI agent harness in Python, one layer at a time, on top of LangChain.

## Roadmap

Each step ends with something you can run and a test.

1. **Agent loop with tools** (`src/harness/loop.py`). Call the model, execute the tool calls it returns, feed results back, stop when it answers in plain text. Written by hand so the loop is visible; LangChain only supplies the Ollama model client and `@tool`.
2. **Real tools and errors.** File read/write, shell, web fetch. Tool errors go back to the model as messages instead of crashing; a turn limit and a token budget stop runaway loops.
3. **Streaming and tracing.** Stream tokens and tool events to the terminal; log every turn so a run can be replayed and debugged.
4. **Context management.** Count tokens, trim or summarize old turns, truncate large tool outputs, use prompt caching.
5. **Memory.** Short-term (the conversation) vs long-term (facts saved to disk and recalled into the prompt), then retrieval over documents.
6. **Planning and permissions.** A todo list the agent maintains, plus approval prompts before risky tools run.
7. **Multi-agent orchestration.** A coordinator that spawns sub-agents with their own context, then compare with LangGraph's version of the same pattern.

## Step 1: run it

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e '.[dev]'
pytest                                   # runs the loop against a scripted fake model
ollama pull llama3.1                     # any tool-calling model works
OLLAMA_MODEL=llama3.1 python -m harness.loop "What is 1234 + 5678?"
```

Exercise: add a second tool (for example `multiply`) and ask a question that needs both.

## Step 2: real tools and errors

Your turn to write the code. Two parts, each checked by tests in `exercises/step2`:

1. **Tools** (`src/harness/tools.py`): implement `read_file`, `write_file`, `run_shell` and `fetch_url`. Use only the standard library.
2. **Errors** (`src/harness/loop.py`, see the comment in `run`): an unknown tool name or a tool that raises must not crash the run. Send the error back to the model as a `ToolMessage` with `status="error"` and keep looping.

```bash
pytest exercises/step2                   # red until you're done
python -m harness.loop "Create notes.txt containing the output of 'uname -a'"
```

Things to notice when you run it for real: what the model does after a tool error, and why `run_shell` with no confirmation step is dangerous (step 6 fixes that).

## Step 3: streaming and tracing

Two more pieces to write, checked by `pytest exercises/step3`:

1. **Streaming and events** (`run` in `src/harness/loop.py`, see its docstring): use `model.stream` so text arrives as it is generated, and report each token, model reply and tool result through `on_event`.
2. **Tracing** (`src/harness/tracing.py`): `JsonlTracer` appends every event to a `.jsonl` file, one JSON object per line.

Then make `python -m harness.loop` print tokens live and save a trace, run a multi-tool question, and read the trace back with `jq` to see every step the agent took.

`harness.testing.ScriptedModel` is the fake model all the tests use; it can `invoke` and `stream`.
