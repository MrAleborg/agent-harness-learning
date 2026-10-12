"""Step 4 exercise: keep what the model sees small enough to fit, and cheap."""


def estimate_tokens(messages) -> int:
    """Rough token count: for each message, len(message.content) / 4 rounded up, summed.

    Real tokenizers differ by model; ~4 characters per token is a common rule of thumb.
    """
    raise NotImplementedError("step 4: implement estimate_tokens")


def truncate_tool_output(text: str, max_chars: int = 2000) -> str:
    """Return `text` unchanged if it fits in `max_chars`.

    Otherwise keep the first and last max_chars // 2 characters and put a marker between them
    saying how many characters were cut, e.g. "\\n[... 8000 chars truncated ...]\\n".
    """
    raise NotImplementedError("step 4: implement truncate_tool_output")


def trim_history(messages: list, max_tokens: int) -> list:
    """Return a new list that fits in `max_tokens` (per estimate_tokens) by dropping old messages.

    Rules:
    - Always keep messages[0] (the user's task) and messages[-1] (the latest), even if they alone
      exceed the budget. If the latest is a tool result, also keep the AIMessage that requested it
      (and that AIMessage's other tool results).
    - Drop the oldest messages after messages[0] first.
    - Never keep a ToolMessage whose AIMessage (the one that requested it) was dropped:
      model APIs reject a tool result with no matching tool call.
    - Don't change the input list.
    """
    raise NotImplementedError("step 4: implement trim_history")
