# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Regressions for the second review pass on the local STT dictation feature:

1. scripts/build_whisper_cpp.sh must not rm -rf a whisper.cpp/src tree under a
   custom Studio home unless Studio itself created it (ownership marker), the
   same policy studio/setup.sh applies before its destructive replacements.
2. _snapshot_is_complete must validate every shard of a sharded PyTorch
   (pytorch_model.bin.index.json) checkpoint, like the safetensors path.
3. _snapshot_is_complete must require tokenizer assets (tokenizer.json or
   vocab.json + merges.txt); weights + config alone decode to blank text.
4. Custom-repo downloads must pin the revision validated beforehand and
   restrict snapshot_download to the model/tokenizer/config/preprocessor file
   classes (TOCTOU + unbounded-download hardening).
5. The GGML sidecar's readiness probe must not treat an arbitrary local HTTP
   responder as whisper-server (mic audio would be posted to it), and the port
   reservation must stay held until just before spawn.
"""

from __future__ import annotations

import http.server
import json
import os
import socket
import stat
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

import core.inference.stt_ggml_sidecar as ggml_module
import core.inference.stt_sidecar as stt_sidecar_module
from core.inference.stt_ggml_sidecar import GgmlSttSidecar, SttEngineUnavailableError
from core.inference.stt_sidecar import validate_remote_model

_BUILD_SCRIPT = _BACKEND_ROOT.parents[1] / "scripts" / "build_whisper_cpp.sh"


# 1. build_whisper_cpp.sh ownership gate ----------------------------------------


def _stub_tools(tmp_path: Path) -> dict:
    """PATH with git/cmake stubs so the script never reaches a real build."""
    bin_dir = tmp_path / "stub-bin"
    bin_dir.mkdir(exist_ok = True)
    for tool in ("git", "cmake"):
        stub = bin_dir / tool
        stub.write_text("#!/bin/sh\necho stub-%s-invoked >&2\nexit 1\n" % tool)
        stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    return env


def _run_build_script(env: dict) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["sh", str(_BUILD_SCRIPT)],
        env = env,
        capture_output = True,
        text = True,
        timeout = 60,
    )


def test_build_script_refuses_unowned_dir_in_custom_studio_home(tmp_path):
    home = tmp_path / "studio-home"
    src = home / "whisper.cpp" / "src"
    src.mkdir(parents = True)
    user_file = src / "user-data.txt"
    user_file.write_text("precious")

    env = _stub_tools(tmp_path)
    env["UNSLOTH_STUDIO_HOME"] = str(home)
    result = _run_build_script(env)

    assert result.returncode != 0
    assert "not marked as an Unsloth-owned" in result.stderr
    # The unowned tree, and the user's file inside it, survived untouched.
    assert user_file.read_text() == "precious"


def test_build_script_proceeds_when_marker_present(tmp_path):
    home = tmp_path / "studio-home"
    install = home / "whisper.cpp"
    (install / "src").mkdir(parents = True)
    (install / ".unsloth-studio-owned").write_text("")

    env = _stub_tools(tmp_path)
    env["UNSLOTH_STUDIO_HOME"] = str(home)
    result = _run_build_script(env)

    # Past the guard: it fails later at the stubbed git clone, not the gate.
    assert "not marked as an Unsloth-owned" not in result.stderr
    assert "stub-git-invoked" in result.stderr


def test_build_script_marks_fresh_custom_install_dir(tmp_path):
    home = tmp_path / "studio-home"
    home.mkdir()

    env = _stub_tools(tmp_path)
    env["UNSLOTH_STUDIO_HOME"] = str(home)
    _run_build_script(env)

    # A directory the script creates is marked so re-runs stay allowed.
    assert (home / "whisper.cpp" / ".unsloth-studio-owned").is_file()


def test_build_script_keeps_legacy_home_behavior(tmp_path):
    fake_home = tmp_path / "user-home"
    src = fake_home / ".unsloth" / "whisper.cpp" / "src"
    src.mkdir(parents = True)

    env = _stub_tools(tmp_path)
    env.pop("UNSLOTH_STUDIO_HOME", None)
    env.pop("STUDIO_HOME", None)
    env["HOME"] = str(fake_home)
    result = _run_build_script(env)

    # The legacy managed dir is always Studio-owned; no gate, straight to git.
    assert "not marked as an Unsloth-owned" not in result.stderr
    assert "stub-git-invoked" in result.stderr


# 2 + 3. _snapshot_is_complete --------------------------------------------------


def _base_snapshot(tmp_path: Path) -> Path:
    snap = tmp_path / "snap"
    snap.mkdir()
    (snap / "config.json").write_text("{}")
    (snap / "preprocessor_config.json").write_text("{}")
    (snap / "tokenizer.json").write_text("{}")
    return snap


def test_sharded_pytorch_snapshot_requires_every_shard(tmp_path):
    snap = _base_snapshot(tmp_path)
    index = {
        "weight_map": {
            "a": "pytorch_model-00001-of-00002.bin",
            "b": "pytorch_model-00002-of-00002.bin",
        }
    }
    (snap / "pytorch_model.bin.index.json").write_text(json.dumps(index))
    (snap / "pytorch_model-00001-of-00002.bin").write_bytes(b"w" * 8)

    # One missing .bin shard must read as incomplete, like the safetensors path.
    assert stt_sidecar_module._snapshot_is_complete(snap) is False

    (snap / "pytorch_model-00002-of-00002.bin").write_bytes(b"w" * 8)
    assert stt_sidecar_module._snapshot_is_complete(snap) is True


def test_snapshot_without_tokenizer_assets_is_incomplete(tmp_path):
    snap = _base_snapshot(tmp_path)
    (snap / "model.safetensors").write_bytes(b"w" * 8)
    assert stt_sidecar_module._snapshot_is_complete(snap) is True

    # Weights + config but no tokenizer decodes to blank text; not complete.
    (snap / "tokenizer.json").unlink()
    assert stt_sidecar_module._snapshot_is_complete(snap) is False

    # The slow vocab.json + merges.txt pair is an accepted alternative.
    (snap / "vocab.json").write_text("{}")
    assert stt_sidecar_module._snapshot_is_complete(snap) is False
    (snap / "merges.txt").write_text("")
    assert stt_sidecar_module._snapshot_is_complete(snap) is True


# 4. Revision pinning and allow_patterns ----------------------------------------


def test_validate_remote_model_returns_the_validated_revision(monkeypatch):
    class _FakeApi:
        def __init__(self, token = None):
            pass

        def model_info(
            self,
            repo,
            expand = None,
            timeout = None,
        ):
            return SimpleNamespace(config = {"model_type": "whisper"}, sha = "abc123")

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "HfApi", _FakeApi)
    result = validate_remote_model("someone/custom-whisper")
    assert result["revision"] == "abc123"


def test_download_pins_revision_and_limits_patterns(monkeypatch):
    captured = {}

    def fake_snapshot_download(**kwargs):
        captured.update(kwargs)
        return "/cached"

    class _FakeApi:
        def __init__(self, token = None):
            pass

        def model_info(
            self,
            repo,
            files_metadata = None,
            timeout = None,
        ):
            return SimpleNamespace(siblings = [], sha = "head456")

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "HfApi", _FakeApi)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)

    state = stt_sidecar_module._SnapshotDownloadState()
    # The revision resolved at validation time wins over the current head.
    state._run("someone/custom-whisper", None, revision = "abc123")
    assert captured["revision"] == "abc123"
    patterns = captured["allow_patterns"]
    assert "*.safetensors" in patterns and "tokenizer.json" in patterns
    # No wildcard that would admit arbitrary repo contents.
    assert "*" not in patterns

    # Without a validated revision (curated repos), pin to the metadata head.
    captured.clear()
    state._run("someone/custom-whisper", None)
    assert captured["revision"] == "head456"
    assert captured["allow_patterns"]


# 5. GGML readiness must identify whisper-server --------------------------------


class _CannedHandler(http.server.BaseHTTPRequestHandler):
    body = b""

    def do_GET(self):  # noqa: N802
        payload = type(self).body
        self.send_response(200)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass


def _serve(body: bytes):
    handler = type("Handler", (_CannedHandler,), {"body": body})
    server = http.server.HTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target = server.serve_forever, daemon = True)
    thread.start()
    return server, server.server_address[1]


def _fake_alive_process():
    return SimpleNamespace(poll = lambda: None, pid = 999999)


def test_wait_for_server_rejects_a_foreign_http_responder(monkeypatch):
    server, port = _serve(b"<html>hello from some other local app</html>")
    try:
        monkeypatch.setattr(ggml_module, "_SERVER_START_TIMEOUT_SECONDS", 1.0)
        with pytest.raises(SttEngineUnavailableError, match = "did not start in time"):
            GgmlSttSidecar._wait_for_server(_fake_alive_process(), port)
    finally:
        server.shutdown()


def test_wait_for_server_accepts_the_whisper_server_page(monkeypatch):
    server, port = _serve(b"<html><title>Whisper.cpp Server</title></html>")
    try:
        monkeypatch.setattr(ggml_module, "_SERVER_START_TIMEOUT_SECONDS", 5.0)
        GgmlSttSidecar._wait_for_server(_fake_alive_process(), port)
    finally:
        server.shutdown()


def test_probe_requires_the_managed_child_to_be_alive():
    server, port = _serve(b"whisper")
    try:
        dead = SimpleNamespace(poll = lambda: 0, pid = 999999)
        assert GgmlSttSidecar._probe_is_whisper_server(dead, port) is False
        assert GgmlSttSidecar._probe_is_whisper_server(_fake_alive_process(), port) is True
    finally:
        server.shutdown()


def test_port_reservation_is_held_until_released():
    reservation, port = GgmlSttSidecar._reserve_free_port()
    try:
        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            with pytest.raises(OSError):
                probe.bind(("127.0.0.1", port))
        finally:
            probe.close()
    finally:
        reservation.close()
    # Released right before spawn: the port becomes bindable for the child.
    child = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    child.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        child.bind(("127.0.0.1", port))
    finally:
        child.close()
