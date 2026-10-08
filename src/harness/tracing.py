"""Step 3 exercise: record every event of a run so it can be read back and debugged."""


class JsonlTracer:
    """Pass an instance as run(..., on_event=JsonlTracer("trace.jsonl")).

    Each call appends the event to the file as one JSON line, with an added "time" key
    (seconds since the epoch, from time.time()).
    """

    def __init__(self, path):
        self.path = path

    def __call__(self, event: dict) -> None:
        raise NotImplementedError("step 3: implement JsonlTracer")
