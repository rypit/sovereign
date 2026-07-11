"""Fake engine adapter for exercising the real ``engine_worker`` entrypoint
in tests, with no real engine bindings installed. Serves a real
``http.server`` HTTP health endpoint and emits a scripted
``state_change(serving)`` telemetry event.

Set env var ``SOVEREIGN_FAKE_ADAPTER_CRASH=1`` to make ``run()`` raise
instead of serving, so tests can exercise the crash path.
"""

from __future__ import annotations

import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

from sovereign.workers.protocol import EventType


def _make_handler(health_path: str) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - stdlib API name
            if self.path == health_path:
                body = b'{"status": "ok"}'
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_response(404)
                self.end_headers()

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            pass  # keep test output quiet

    return Handler


def run(cfg: Any, telemetry: Any, controller: Any) -> None:
    if os.environ.get("SOVEREIGN_FAKE_ADAPTER_CRASH") == "1":
        raise RuntimeError("fake adapter simulated crash")

    handler_cls = _make_handler(cfg.health_path)
    httpd = HTTPServer((cfg.host, cfg.port), handler_cls)
    httpd.timeout = 0.2

    telemetry.emit(EventType.STATE_CHANGE, {"state": "serving"})

    def _shutdown() -> None:
        httpd.shutdown()

    controller.shutdown_callback = _shutdown

    serve_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    serve_thread.start()
    controller.stop_event.wait()
    httpd.shutdown()
    serve_thread.join(timeout=5.0)
    httpd.server_close()
