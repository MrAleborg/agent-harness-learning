"""Step 2 exercise: real tools. Replace each NotImplementedError with a working body.

Tools may raise on failure: the loop (not the tool) turns exceptions into
messages the model can read. Run `pytest exercises/step2` to check your work.
"""
from langchain_core.tools import tool


@tool
def read_file(path: str) -> str:
    """Return the text content of the file at `path`."""
    raise NotImplementedError("step 2: implement read_file")


@tool
def write_file(path: str, content: str) -> str:
    """Write `content` to the file at `path`, creating parent folders, and confirm."""
    raise NotImplementedError("step 2: implement write_file")


@tool
def run_shell(command: str, timeout: int = 30) -> str:
    """Run a shell command. Return its exit code, stdout and stderr.

    If it runs longer than `timeout` seconds, stop it and say it timed out.
    """
    raise NotImplementedError("step 2: implement run_shell (hint: subprocess.run)")


@tool
def fetch_url(url: str) -> str:
    """Fetch `url` over HTTP and return the response body as text (first 10,000 chars)."""
    raise NotImplementedError("step 2: implement fetch_url (hint: urllib.request)")
