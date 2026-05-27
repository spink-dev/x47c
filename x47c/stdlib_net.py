import socket
import threading
from typing import Callable

def connect(host: str, port: int) -> socket.socket:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((host, port))
    return s

def listen(port: int, handler: Callable):
    def _run():
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("", port))
        server.listen(5)
        while True:
            conn, addr = server.accept()
            threading.Thread(target=handler, args=(conn, addr), daemon=True).start()
    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return t

def http_get(url: str, timeout: float = 10.0) -> dict:
    import urllib.request
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            return {"status": resp.status, "body": body, "headers": dict(resp.headers)}
    except Exception as e:
        return {"error": str(e), "status": 0}
