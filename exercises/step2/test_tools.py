import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from harness.tools import fetch_url, read_file, run_shell, write_file


def test_write_then_read(tmp_path):
    path = tmp_path / "sub" / "note.txt"
    write_file.invoke({"path": str(path), "content": "hello"})
    assert read_file.invoke({"path": str(path)}) == "hello"


def test_shell_reports_output_and_exit_code():
    assert "hi" in run_shell.invoke({"command": "echo hi"})
    assert "3" in run_shell.invoke({"command": "exit 3"})


def test_shell_times_out():
    assert "timed out" in run_shell.invoke({"command": "sleep 5", "timeout": 1}).lower()


def test_fetch_url():
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"pong")

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        assert fetch_url.invoke({"url": f"http://127.0.0.1:{server.server_port}/"}) == "pong"
    finally:
        server.shutdown()
