"""Step 1: the agent loop, written by hand so every turn is visible."""
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool


@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


TOOLS = {t.name: t for t in [add]}


def run(model, prompt: str, max_turns: int = 10) -> str:
    """Call the model, run any tools it asks for, feed results back, repeat."""
    messages = [HumanMessage(prompt)]
    for _ in range(max_turns):
        reply = model.invoke(messages)
        messages.append(reply)
        if not reply.tool_calls:
            return reply.content
        for call in reply.tool_calls:
            result = TOOLS[call["name"]].invoke(call["args"])
            messages.append(ToolMessage(str(result), tool_call_id=call["id"]))
    raise RuntimeError(f"no final answer after {max_turns} turns")


if __name__ == "__main__":
    import os
    import sys
    from langchain_ollama import ChatOllama

    # needs a model that supports tool calling, e.g. llama3.1 or qwen3
    model = ChatOllama(model=os.environ.get("OLLAMA_MODEL", "llama3.1")).bind_tools(list(TOOLS.values()))
    print(run(model, " ".join(sys.argv[1:]) or "What is 1234 + 5678?"))
