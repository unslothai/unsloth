# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "tests" / "studio" / "run_real_mlx_smoke.py"

# The fake llama.cpp binaries are shebang scripts.
pytestmark = pytest.mark.skipif(sys.platform == "win32", reason = "needs shebang executables")

FAKE_LLAMA_CLI = """#!{python}
print({prompt!r} + "Unsloth!")
"""

FAKE_LLAMA_SERVER = """#!{python}
import json, sys
from http.server import BaseHTTPRequestHandler, HTTPServer

class Handler(BaseHTTPRequestHandler):
    def _send(self, payload):
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._send({{"status": "ok"}})

    def do_POST(self):
        sent = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self._send({{"content": "Unsloth!" if sent["prompt"] == {prompt!r} else ""}})

    def log_message(self, *args):
        pass

HTTPServer(("127.0.0.1", int(sys.argv[sys.argv.index("--port") + 1])), Handler).serve_forever()
"""


@pytest.fixture(scope = "module")
def smoke():
    spec = importlib.util.spec_from_file_location("run_real_mlx_smoke", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def reload_gguf(smoke, tmp_path, monkeypatch):
    llama_dir = tmp_path / "llama.cpp-install"
    (llama_dir / "build" / "bin").mkdir(parents = True)
    save_dir = tmp_path / "workdir" / "gguf"
    save_dir.mkdir(parents = True)
    (save_dir / "model.Q8_0.gguf").write_bytes(b"GGUF")

    monkeypatch.setenv("UNSLOTH_LLAMA_CPP_PATH", str(llama_dir))
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setitem(sys.modules, "unsloth_zoo.llama_cpp", None)
    monkeypatch.chdir(tmp_path)

    def run(**binaries):
        for name, source in binaries.items():
            binary = llama_dir / "build" / "bin" / name.replace("_", "-")
            binary.write_text(source.format(python = sys.executable, prompt = smoke.PROMPT))
            binary.chmod(0o755)
        assert smoke._reload_gguf(save_dir, {"phases": {}}) == 0
        return json.loads((save_dir.parent / "gguf_reload_metrics.json").read_text())

    return run


def test_gguf_reload_uses_llama_server_when_the_install_has_no_llama_cli(reload_gguf):
    metrics = reload_gguf(llama_server = FAKE_LLAMA_SERVER)
    assert metrics["generation"] == "Unsloth!"
    assert metrics["gguf_has_expected"] is True


def test_gguf_reload_still_prefers_llama_cli_when_both_are_installed(reload_gguf):
    metrics = reload_gguf(llama_cli = FAKE_LLAMA_CLI, llama_server = "#!/bin/sh\nexit 1\n")
    assert metrics["llama_cli_returncode"] == 0
    assert metrics["gguf_has_expected"] is True


def test_gguf_reload_fails_when_neither_binary_is_installed(reload_gguf):
    with pytest.raises(SystemExit, match = "not found"):
        reload_gguf()
