"""Step 4 exercise: keep what the model sees small enough to fit, and cheap."""

import math

from langchain_core.messages import ToolMessage

def estimate_tokens(messages) -> int:
    """Rough token count: for each message, len(message.content) / 4 rounded up, summed.

    Real tokenizers differ by model; ~4 characters per token is a common rule of thumb.
    """
    return sum(math.ceil(len(m.content) / 4) for m in messages)


def truncate_tool_output(text: str, max_chars: int = 2000) -> str:
    """Return `text` unchanged if it fits in `max_chars`.

    Otherwise keep the first and last max_chars // 2 characters and put a marker between them
    saying how many characters were cut, e.g. "\\n[... 8000 chars truncated ...]\\n".
    """
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    cut = len(text) - 2 * half
    return f"{text[:half]}\n[... {cut} chars truncated ...]\n{text[len(text) - half:]}"


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
    if estimate_tokens(messages) <= max_tokens:
        return list(messages)
    # An AI tool request and its tool results are one unit: dropping either alone breaks the API.
    units = []
    for m in messages[1:]:
        if isinstance(m, ToolMessage) and units and getattr(units[-1][0], "tool_calls", None):
            units[-1].append(m)
        else:
            units.append([m])
    kept = []
    used = estimate_tokens(messages[:1])
    for unit in reversed(units):
        cost = estimate_tokens(unit)
        if kept and used + cost > max_tokens:
            break  # the latest unit is always kept, even over budget
        kept.append(unit)
        used += cost
    return [messages[0]] + [m for unit in reversed(kept) for m in unit]
