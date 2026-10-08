"""A fake model for tests: replays canned AIMessages through invoke() or stream()."""
import json
import re

from langchain_core.messages import AIMessageChunk


class ScriptedModel:
    def __init__(self, *replies):
        self.replies = list(replies)
        self.seen = []  # the messages the model was given, one list per call

    def invoke(self, messages):
        self.seen.append(list(messages))
        return self.replies.pop(0)

    def stream(self, messages):
        """Yield the next reply in pieces, like a real model: word by word, tool args split in two."""
        reply = self.invoke(messages)
        for word in re.findall(r"\S+\s*", reply.content):
            yield AIMessageChunk(content=word)
        for i, call in enumerate(reply.tool_calls):
            args = json.dumps(call["args"])
            half = len(args) // 2
            yield AIMessageChunk(content="", tool_call_chunks=[{"name": call["name"], "args": args[:half], "id": call["id"], "index": i}])
            yield AIMessageChunk(content="", tool_call_chunks=[{"name": None, "args": args[half:], "id": None, "index": i}])
