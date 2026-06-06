"""Launch the PyCopter Web UI from the repository root."""

from __future__ import annotations

import socket
import sys
from pathlib import Path

import panel as pn


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gui.app import create_app  # noqa: E402


def _first_free_port(start: int = 5006, attempts: int = 20) -> int:
    for port in range(start, start + attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise RuntimeError(f"No free localhost port found in range {start}-{start + attempts - 1}.")


def _websocket_origins(port: int) -> list[str]:
    return [f"127.0.0.1:{port}", f"localhost:{port}"]


def main() -> None:
    port = _first_free_port()
    print(f"Starting PyCopter Web UI at http://127.0.0.1:{port}/dashboard")
    print("Close this terminal window or press Ctrl+C to stop the Web UI.")
    pn.serve(
        {"/dashboard": create_app},
        address="127.0.0.1",
        port=port,
        websocket_origin=_websocket_origins(port),
        show=True,
        title="pycopter Web UI",
    )


if __name__ == "__main__":
    main()
