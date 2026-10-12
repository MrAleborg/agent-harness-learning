"""The agent loop, written by hand so every turn is visible."""
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool

from harness.tools import fetch_url, read_file, run_shell, write_file


@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b

@tool
def multiply(a:int, b:int) -> int:
    """Multiply two integers"""
    return a * b

TOOLS = {t.name: t for t in [add, multiply, read_file, write_file, run_shell, fetch_url]}


def run(model, prompt: str, max_turns: int = 10, on_event=lambda event: None, max_context_tokens=None) -> str:
    """Call the model, run any tools it asks for, feed results back, repeat.

    Step 4 exercise: keep the context small (helpers in harness/context.py).
    - Pass every tool result through truncate_tool_output before it becomes a ToolMessage.
    - When max_context_tokens is set, send the model trim_history(messages, max_context_tokens)
      instead of the full history. Keep the full history in `messages` itself.

    Step 3 exercise: stream the model's reply and report what happens through `on_event`.
    - Use model.stream(messages) instead of model.invoke(messages). It yields AIMessageChunks;
      add them together (chunk1 + chunk2 + ...) to rebuild the full reply, tool calls included.
    - Call on_event with a dict for each of these, in order:
        {"type": "token", "text": <chunk text>}                   for every non-empty streamed chunk
        {"type": "model_reply", "content": ..., "tool_calls": ...} once the reply is complete
        {"type": "tool_result", "name": ..., "content": ..., "status": "success" or "error"}
    """
    messages = [HumanMessage(prompt)]
    for _ in range(max_turns):
        reply = None
        for chunk in model.stream(messages):
            if chunk.content:
                on_event({"type": "token", "text": chunk.content})
            reply = chunk if reply is None else reply + chunk
        messages.append(reply)
        on_event({"type": "model_reply", "content": reply.content, "tool_calls": reply.tool_calls})
        if not reply.tool_calls:
            return reply.content
        for call in reply.tool_calls:
            # step 2 exercise: an unknown tool name or a tool that raises crashes the run here.
            # Catch it and append ToolMessage(<what went wrong>, tool_call_id=..., status="error")
            # so the model sees the error and can try something else.
            # Outside the try so an on_event failure is not reported as a tool error.
            try:
                content, status = str(TOOLS[call["name"]].invoke(call["args"])), "success"
            except Exception as e:
                content, status = str(e), "error"
            messages.append(ToolMessage(content, tool_call_id=call["id"], status=status))
            on_event({"type": "tool_result", "name": call["name"], "content": content, "status": status})
    raise RuntimeError(f"no final answer after {max_turns} turns")


if __name__ == "__main__":
    import os
    import sys
    from langchain_openai import ChatOpenAI

    # OpenRouter speaks the OpenAI API; "openrouter/free" routes to a free model, which may not support tool calling
    model = ChatOpenAI(
        model=os.environ.get("OPENROUTER_MODEL", "openrouter/free"),
        base_url="http://localhost:20128/v1",
        api_key=os.environ["OMNIROUTE_KEY"],
    ).bind_tools(list(TOOLS.values()))
    print(run(model, " ".join(sys.argv[1:]) or "What is 1234 + 5678?"))
