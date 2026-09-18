"""Opt-in, source-only runtime REPL for inspecting a live FaceTrackVR process.

The application loads this file dynamically only in a source checkout. Enable
it persistently with ``~/.config/EyeTrackVR/runtime_repl.json`` or per launch
with ``FACETRACKVR_RUNTIME_REPL=1``. Keeping it under scripts/ and out of the
normal import graph means PyInstaller does not bundle the arbitrary-code
endpoint.

Requests and responses are one JSON object per line. The small command-line
client at the bottom is the intended way to query the running process.
"""

from __future__ import annotations

import argparse
import hmac
import json
import os
import secrets
import socket
import socketserver
import threading
import traceback
from typing import Any


_MAX_REQUEST_BYTES = 256 * 1024
_MAX_RESULT_CHARS = 1024 * 1024


def default_config_path() -> str:
    base = os.environ.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config")
    return os.path.join(base, "EyeTrackVR", "runtime_repl.json")


def enabled_from_config(path: str | None = None) -> bool:
    """Return whether this source checkout should expose its local REPL.

    The environment variable is an explicit per-launch override: common true
    spellings enable it and every other provided value disables it. Without an
    override, the separate development config persists independently of the
    application's frequently rewritten tracking settings.
    """
    override = os.environ.get("FACETRACKVR_RUNTIME_REPL")
    if override is not None:
        return override.strip().lower() in {"1", "true", "yes", "on"}
    try:
        with open(path or default_config_path(), encoding="utf-8") as config_file:
            config = json.load(config_file)
    except (OSError, json.JSONDecodeError):
        return False
    return config.get("enabled") is True


class _LoopbackServer(socketserver.TCPServer):
    allow_reuse_address = True


class _RequestHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        raw = self.rfile.readline(_MAX_REQUEST_BYTES + 1)
        if len(raw) > _MAX_REQUEST_BYTES:
            response = {"ok": False, "error": "request is too large"}
        else:
            try:
                request_data = json.loads(raw)
                response = self.server.runtime_repl.execute(request_data)
            except Exception:
                response = {"ok": False, "error": traceback.format_exc()}
        self.wfile.write(json.dumps(response).encode("utf-8") + b"\n")


class RuntimeReplServer:
    """Single-process Python evaluation endpoint bound strictly to loopback."""

    def __init__(
        self,
        namespace: dict[str, Any],
        *,
        port: int = 5678,
        token: str | None = None,
    ) -> None:
        self.namespace = dict(namespace)
        self.namespace.setdefault("__builtins__", __builtins__)
        self.token = token or secrets.token_urlsafe(24)
        self._execution_lock = threading.Lock()
        self._server = _LoopbackServer(("127.0.0.1", port), _RequestHandler)
        self._server.runtime_repl = self
        self.host, self.port = self._server.server_address
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            daemon=True,
            name="RuntimeREPL",
        )

    def start(self) -> "RuntimeReplServer":
        self._thread.start()
        return self

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        if self._thread is not threading.current_thread():
            self._thread.join(timeout=1.0)

    def execute(self, request_data: dict[str, Any]) -> dict[str, Any]:
        supplied_token = str(request_data.get("token", ""))
        if not hmac.compare_digest(supplied_token, self.token):
            return {"ok": False, "error": "authentication failed"}

        operation = request_data.get("op", "eval")
        code = request_data.get("code")
        if operation not in {"eval", "exec"} or not isinstance(code, str):
            return {"ok": False, "error": "expected op=eval|exec and string code"}

        try:
            with self._execution_lock:
                if operation == "eval":
                    result = eval(compile(code, "<runtime-repl>", "eval"), self.namespace)
                else:
                    exec(compile(code, "<runtime-repl>", "exec"), self.namespace)
                    result = self.namespace.get("_")
            rendered = repr(result)
            if len(rendered) > _MAX_RESULT_CHARS:
                rendered = rendered[:_MAX_RESULT_CHARS] + "...<truncated>"
            return {"ok": True, "result": rendered}
        except BaseException:
            return {"ok": False, "error": traceback.format_exc()}


def request(
    code: str,
    *,
    operation: str = "eval",
    port: int = 5678,
    token: str,
    timeout: float = 5.0,
) -> dict[str, Any]:
    payload = json.dumps({"token": token, "op": operation, "code": code})
    with socket.create_connection(("127.0.0.1", port), timeout=timeout) as connection:
        connection.sendall(payload.encode("utf-8") + b"\n")
        response = connection.makefile("rb").readline(_MAX_RESULT_CHARS + 4096)
    return json.loads(response)


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    operation = parser.add_mutually_exclusive_group(required=True)
    operation.add_argument("--eval", dest="eval_code")
    operation.add_argument("--exec", dest="exec_code")
    parser.add_argument("--port", type=int, default=5678)
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument(
        "--token",
        default=os.environ.get("FACETRACKVR_RUNTIME_REPL_TOKEN", ""),
    )
    args = parser.parse_args()
    if not args.token:
        parser.error("pass --token or set FACETRACKVR_RUNTIME_REPL_TOKEN")
    op = "eval" if args.eval_code is not None else "exec"
    code = args.eval_code if args.eval_code is not None else args.exec_code
    response = request(
        code,
        operation=op,
        port=args.port,
        token=args.token,
        timeout=args.timeout,
    )
    if response.get("ok"):
        print(response["result"])
        return 0
    print(response.get("error", "unknown runtime REPL error"))
    return 1


if __name__ == "__main__":
    raise SystemExit(_main())
