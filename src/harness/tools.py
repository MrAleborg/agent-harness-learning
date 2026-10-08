"""Step 2 exercise: real tools. Replace each NotImplementedError with a working body.

Tools may raise on failure: the loop (not the tool) turns exceptions into
messages the model can read. Run `pytest exercises/step2` to check your work.
"""
import os
import subprocess
import urllib.request
from langchain_core.tools import tool


@tool
def read_file(path: str) -> str:
    """Return the text content of the file at `path`."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, 'r') as file:
        return "".join(file.readlines())


@tool
def write_file(path: str, content: str) -> str:
    """Write `content` to the file at `path`, creating parent folders, and confirm."""
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    with open(path, 'w') as file:
        file.write(content)
        return "File has been written."


@tool
def run_shell(command: str, timeout: int = 30) -> str:
    """Run a shell command. Return its exit code, stdout and stderr.

    If it runs longer than `timeout` seconds, stop it and say it timed out.
    """
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=timeout)
        return f"exit code: {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    except subprocess.TimeoutExpired:
        return f"Command timed out after {timeout} seconds"


@tool
def fetch_url(url: str) -> str:
    """Fetch `url` over HTTP and return the response body as text (first 10,000 chars)."""
    with urllib.request.urlopen(url) as response:
        body = response.read().decode('utf-8')
        return body[:10000]
