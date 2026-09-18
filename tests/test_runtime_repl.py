import json

from scripts.runtime_repl import RuntimeReplServer, enabled_from_config, request


def test_runtime_repl_persistent_config_and_environment_override(tmp_path, monkeypatch):
    config_path = tmp_path / "runtime_repl.json"
    config_path.write_text(json.dumps({"enabled": True}), encoding="utf-8")
    assert enabled_from_config(str(config_path))

    monkeypatch.setenv("FACETRACKVR_RUNTIME_REPL", "0")
    assert not enabled_from_config(str(config_path))
    monkeypatch.setenv("FACETRACKVR_RUNTIME_REPL", "yes")
    assert enabled_from_config(str(config_path))


def test_runtime_repl_authenticates_and_keeps_a_persistent_namespace():
    server = RuntimeReplServer({"value": 2}, port=0, token="test-token").start()
    try:
        denied = request("value", port=server.port, token="wrong-token")
        assert denied == {"ok": False, "error": "authentication failed"}

        assigned = request(
            "value = value + 3",
            operation="exec",
            port=server.port,
            token="test-token",
        )
        assert assigned == {"ok": True, "result": "None"}

        evaluated = request("value * 2", port=server.port, token="test-token")
        assert evaluated == {"ok": True, "result": "10"}
    finally:
        server.close()


def test_runtime_repl_returns_evaluation_tracebacks():
    server = RuntimeReplServer({}, port=0, token="test-token").start()
    try:
        response = request("1 / 0", port=server.port, token="test-token")
        assert response["ok"] is False
        assert "ZeroDivisionError" in response["error"]
    finally:
        server.close()
