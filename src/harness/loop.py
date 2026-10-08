"""Step 1: the agent loop, written by hand so every turn is visible."""
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


def run(model, prompt: str, max_turns: int = 10) -> str:
    """Call the model, run any tools it asks for, feed results back, repeat."""
    messages = [HumanMessage(prompt)]
    for _ in range(max_turns):
        reply = model.invoke(messages)
        messages.append(reply)
        if not reply.tool_calls:
            return reply.content
        for call in reply.tool_calls:
            # step 2 exercise: an unknown tool name or a tool that raises crashes the run here.
            # Catch it and append ToolMessage(<what went wrong>, tool_call_id=..., status="error")
            # so the model sees the error and can try something else.
            try:
                result = TOOLS[call["name"]].invoke(call["args"])
                messages.append(ToolMessage(str(result), tool_call_id=call["id"]))
            except Exception as e:
                messages.append(ToolMessage(str(e), tool_call_id=call["id"], status="error"))
    raise RuntimeError(f"no final answer after {max_turns} turns")


if __name__ == "__main__":
    import os
    import sys
    from langchain_ollama import ChatOllama

    # needs a model that supports tool calling, e.g. llama3.1 or qwen3
    model = ChatOllama(model=os.environ.get("OLLAMA_MODEL", "gemma4:12b")).bind_tools(list(TOOLS.values()))
    print(run(model, " ".join(sys.argv[1:]) or "What is 1234 + 5678?"))
