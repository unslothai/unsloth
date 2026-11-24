# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""FastFlowLM without Lemonade: flm_host's lemond-compatible API, the direct install and the launcher."""

from __future__ import annotations

import hashlib
import io
import json
import sys
import textwrap
import threading
import zipfile
from http.server import ThreadingHTTPServer
from pathlib import Path

import httpx
import pytest

from core.inference import flm_host as fh

CATALOG = {
    "models": [
        {
            "name": "qwen3:0.6b",
            "label": ["reasoning"],
            "footprint": 0.65,
            "installed": True,
            "default_context_length": 32768,
        },
        {"name": "d3:edge", "label": ["decision"], "footprint": 0.58, "installed": False},
        {
            "name": "embed-gemma:300m",
            "label": ["embeddings"],
            "footprint": 0.62,
            "installed": False,
        },
    ]
}

# A stand-in for `flm serve`: answers /v1/models, echoes the model a chat named, then streams.
FAKE_SERVE = textwrap.dedent(
    """
    import json, sys
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    args = sys.argv[1:]
    if "fail" in args[1]:
        print("[ERROR]  NPU is busy"); sys.exit(3)
    port = int(args[args.index("--port") + 1])
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a): pass
        def do_GET(self):
            body = json.dumps({"data": [{"id": args[1]}]}).encode()
            self.send_response(200); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
        def do_POST(self):
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            body = ("data: " + json.dumps({"model": req["model"], "args": args}) + "\\n\\ndata: [DONE]\\n\\n").encode()
            self.send_response(200); self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
    print("serving", flush = True)
    ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
    """
)


class FakeFlm:
    def __init__(self, tmp_path):
        self.binary = tmp_path / "fake_serve.py"
        self.binary.write_text(FAKE_SERVE)
        self.calls, self.fail = [], set()

    def run(
        self,
        *args,
        timeout = 600.0,
        stdin = "",
    ):
        self.calls.append(args)
        if args[0] in self.fail:
            return 1, f"[ERROR]  {args[0]} failed"
        return 0, "{}" if args[0] == "version" else "ok"

    def catalog(self):
        return [fh.catalog_row(entry) for entry in CATALOG["models"]]


@pytest.fixture
def host(tmp_path, monkeypatch):
    real = fh.subprocess.Popen
    # Windows cannot exec a .py; run the fake serve on this interpreter.
    monkeypatch.setattr(fh.subprocess, "Popen", lambda cmd, **k: real([sys.executable, *cmd], **k))
    flm, serving = FakeFlm(tmp_path), fh.Serving()
    server = ThreadingHTTPServer(("127.0.0.1", 0), fh.make_handler(flm, serving, "key"))
    threading.Thread(target = server.serve_forever, daemon = True).start()
    client = httpx.Client(
        base_url = f"http://127.0.0.1:{server.server_port}",
        headers = {"Authorization": "Bearer key"},
        timeout = 60,
    )
    yield client, flm, serving
    serving.stop()
    server.shutdown()


def test_catalog_rows_use_lemonades_ids_and_chat_labels():
    rows = {r["id"]: r for r in (fh.catalog_row(e) for e in CATALOG["models"])}
    assert rows["qwen3-0.6b-FLM"] == {
        "id": "qwen3-0.6b-FLM", "checkpoint": "qwen3:0.6b", "labels": ["reasoning", "chat"], "size": 0.65,
        "downloaded": True, "recipe": "flm", "max_context_window": None,
    }  # fmt: skip
    assert rows["d3-edge-FLM"]["labels"] == ["decision", "chat"]
    assert rows["embed-gemma-300m-FLM"]["labels"] == ["embeddings"]


def test_context_window_is_the_downloaded_configs_like_lemond(tmp_path):
    entry = {"name": "qwen3:0.6b", "installed": True, "default_context_length": 32768,
             "url": "https://huggingface.co/FastFlowLM/Qwen3-0.6B-NPU2"}  # fmt: skip
    (tmp_path / "Qwen3-0.6B-NPU2").mkdir()
    (tmp_path / "Qwen3-0.6B-NPU2" / "config.json").write_text('{"max_position_embeddings": 40960}')
    assert fh.catalog_row(entry, tmp_path)["max_context_window"] == 40960
    assert fh.catalog_row({**entry, "installed": False}, tmp_path)["max_context_window"] is None
    assert fh.catalog_row({**entry, "url": "x/Other"}, tmp_path)["max_context_window"] is None


def test_flm_output_is_tagged_process_so_studio_drops_prompts_outside_a_load(capsys):
    from core.inference.lemonade_server import _LEMOND_LINE

    fh.log("prompt text", "Process")
    fh.log("ready")
    lines = capsys.readouterr().out.splitlines()
    assert [_LEMOND_LINE.match(line).groups() for line in lines] == [
        ("Info", "Process"),
        ("Info", "FlmHost"),
    ]


def test_the_key_is_required_except_for_live(host):
    client, _, _ = host
    assert httpx.get(f"{client.base_url}/live").status_code == 200
    assert httpx.get(f"{client.base_url}/v1/health").status_code == 401
    assert client.get("/v1/health").json() == {"status": "ok", "all_models_loaded": []}
    assert client.post("/internal/set", json = {"log_level": "debug"}).status_code == 200


def test_models_install_and_options(host):
    client, flm, _ = host
    assert [r["id"] for r in client.get("/v1/models?show_all=true").json()["data"]] == [
        "qwen3-0.6b-FLM",
        "d3-edge-FLM",
        "embed-gemma-300m-FLM",
    ]
    downloaded = [r["id"] for r in client.get("/v1/models").json()["data"]]
    assert downloaded == [r["id"] for r in flm.catalog() if r["downloaded"]] != []
    assert len(downloaded) < len(client.get("/v1/models?show_all=1").json()["data"])
    assert client.get("/v1/models/qwen3-0.6b-FLM/options").json()["effective"]["flm_args"] == ""
    assert client.post("/v1/install", json = {"recipe": "flm"}).status_code == 200
    flm.fail.add("version")
    assert client.post("/v1/install", json = {"recipe": "flm"}).status_code == 500


def test_pull_ends_with_complete_or_error_events(host):
    client, flm, _ = host
    text = client.post("/v1/pull", json = {"model_name": "qwen3-0.6b-FLM", "stream": True}).text
    assert text.startswith("event: complete") and flm.calls[-1] == ("pull", "qwen3:0.6b")
    flm.fail.add("pull")
    assert client.post("/v1/pull", json = {"model_name": "qwen3-0.6b-FLM"}).text.startswith(
        "event: error"
    )
    assert client.post("/v1/pull", json = {"model_name": "nope"}).status_code == 404


def test_load_serves_proxies_chat_by_tag_and_unloads(host):
    client, _, serving = host
    r = client.post(
        "/v1/load",
        json = {"model_name": "qwen3-0.6b-FLM", "ctx_size": 4096, "flm_args": "--pmode turbo"},
    )
    assert r.status_code == 200, r.text
    health = client.get("/v1/health").json()["all_models_loaded"]
    assert health == [{"model_name": "qwen3-0.6b-FLM", "checkpoint": "qwen3:0.6b", "device": "npu",
                       "recipe": "flm", "loaded": True, "recipe_options": {"ctx_size": 4096}}]  # fmt: skip
    chat = client.post(
        "/v1/chat/completions", json = {"model": "qwen3-0.6b-FLM", "stream": True, "messages": []}
    )
    data = json.loads(chat.text.split("data: ", 2)[1].split("\n")[0])
    assert data["model"] == "qwen3:0.6b"
    assert data["args"][-4:] == ["--ctx-len", "4096", "--pmode", "turbo"]
    assert client.post("/v1/unload", json = {"model_name": "qwen3-0.6b-FLM"}).status_code == 200
    assert not serving.alive() and client.get("/v1/health").json()["all_models_loaded"] == []
    assert client.post("/v1/chat/completions", json = {"messages": []}).status_code == 503


def test_a_configured_proxy_never_sees_the_loopback_calls(host, monkeypatch):
    client, _, _ = host
    for name in ("HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy"):
        monkeypatch.setenv(name, "http://127.0.0.1:9")
    for name in ("NO_PROXY", "no_proxy"):
        monkeypatch.delenv(name, raising = False)
    r = client.post("/v1/load", json = {"model_name": "qwen3-0.6b-FLM", "ctx_size": 4096})
    assert r.status_code == 200, r.text
    chat = client.post("/v1/chat/completions", json = {"model": "x", "stream": True, "messages": []})
    assert chat.status_code == 200 and "qwen3:0.6b" in chat.text


def test_a_failed_load_names_flms_reason(host):
    client, _, serving = host
    CATALOG["models"].append({"name": "fail:1b", "label": [], "footprint": 1, "installed": True})
    try:
        r = client.post("/v1/load", json = {"model_name": "fail-1b-FLM"})
    finally:
        CATALOG["models"].pop()
    assert r.status_code == 500
    assert (
        "FastFlowLM failed to start" in r.text and "NPU is busy" in r.text and not serving.alive()
    )


def test_a_refused_flm_exe_answers_with_the_windows_error(host, monkeypatch):
    client, flm, _ = host

    def refuse(*args, **kwargs):
        raise OSError(4551, "An Application Control policy has blocked this file")

    monkeypatch.setattr(fh.subprocess, "Popen", refuse)
    r = client.post("/v1/load", json = {"model_name": "qwen3-0.6b-FLM"})
    assert r.status_code == 500 and "Application Control policy" in r.text, r.text
    monkeypatch.setattr(fh.subprocess, "run", refuse)
    code, output = fh.Flm.run(flm, "pull", "qwen3:0.6b")
    assert code == 1 and "Application Control policy" in output


def test_delete_stops_a_serving_model_then_removes(host):
    client, flm, serving = host
    client.post("/v1/load", json = {"model_name": "qwen3-0.6b-FLM"})
    assert client.post("/v1/delete", json = {"model_name": "qwen3-0.6b-FLM"}).status_code == 200
    assert not serving.alive() and flm.calls[-1] == ("remove", "qwen3:0.6b")
    flm.fail.add("remove")
    assert client.post("/v1/delete", json = {"model_name": "qwen3-0.6b-FLM"}).status_code == 500


def test_a_silent_refused_dll_exit_names_its_status(tmp_path, monkeypatch):
    npu = tmp_path / "bin" / "flm" / "npu"
    npu.mkdir(parents = True)
    (npu / ("flm.exe" if sys.platform == "win32" else "flm")).write_text("")
    flm = fh.Flm(tmp_path)

    class Done:
        returncode, stdout, stderr = 0xC0E90002, "", ""

    monkeypatch.setattr(fh.subprocess, "run", lambda *a, **k: Done())
    code, output = flm.run("version", "--json")
    assert code and "0xC0E90002" in output


def test_flm_errors_printed_with_exit_zero_count_as_failures(tmp_path, monkeypatch):
    npu = tmp_path / "bin" / "flm" / "npu"
    npu.mkdir(parents = True)
    (npu / ("flm.exe" if sys.platform == "win32" else "flm")).write_text("")
    flm = fh.Flm(tmp_path)

    class Done:
        returncode, stdout, stderr = 0, "[ERROR]  Model not found: x", ""

    monkeypatch.setattr(fh.subprocess, "run", lambda *a, **k: Done())
    assert flm.run("remove", "x")[0] == 1
    Done.stdout = json.dumps(CATALOG)
    assert [r["id"] for r in flm.catalog()][0] == "qwen3-0.6b-FLM"


@pytest.mark.parametrize(
    "stdout, stderr",
    [
        # flm prints the list with dump(4); a line after it used to read as "Extra data".
        (json.dumps(CATALOG, indent = 4) + "\n", "[XRT] WARNING: device {0} busy\n"),
        (json.dumps(CATALOG, indent = 4) + "\n[FLM]  done {ok}\n", ""),
        ("[FLM]  NPU {dev04}\n" + json.dumps(CATALOG, indent = 4) + "\n", ""),
    ],
)
def test_the_model_list_ignores_text_flm_prints_around_it(tmp_path, monkeypatch, stdout, stderr):
    npu = tmp_path / "bin" / "flm" / "npu"
    npu.mkdir(parents = True)
    (npu / ("flm.exe" if sys.platform == "win32" else "flm")).write_text("")
    flm = fh.Flm(tmp_path)

    class Done:
        returncode = 0

    Done.stdout, Done.stderr = stdout, stderr
    monkeypatch.setattr(fh.subprocess, "run", lambda *a, **k: Done())
    assert [r["id"] for r in flm.catalog()] == [fh.catalog_row(e)["id"] for e in CATALOG["models"]]


def test_a_model_list_without_json_names_flms_output(tmp_path, monkeypatch):
    npu = tmp_path / "bin" / "flm" / "npu"
    npu.mkdir(parents = True)
    (npu / ("flm.exe" if sys.platform == "win32" else "flm")).write_text("")
    flm = fh.Flm(tmp_path)

    class Done:
        returncode, stdout, stderr = 0, '{"models": [', "terminate called after throwing\n"

    monkeypatch.setattr(fh.subprocess, "run", lambda *a, **k: Done())
    with pytest.raises(RuntimeError, match = "flm list failed: .*terminate called"):
        flm.catalog()


# Direct install and launcher.


def _zip() -> bytes:
    """The pinned asset's format: a zip on Windows, a tar.gz on Linux."""
    buf = io.BytesIO()
    files = {"flm.exe" if sys.platform == "win32" else "flm": b"MZ", "llama_npu.dll": b"MZ"}
    if sys.platform == "win32":
        with zipfile.ZipFile(buf, "w") as zf:
            for name, data in files.items():
                zf.writestr(name, data)
    else:
        import tarfile
        with tarfile.open(fileobj = buf, mode = "w:gz") as tf:
            for name, data in files.items():
                info = tarfile.TarInfo(name)
                info.size, info.mode = len(data), 0o755
                tf.addfile(info, io.BytesIO(data))
    return buf.getvalue()


@pytest.fixture
def installer(tmp_path, monkeypatch):
    studio = Path(__file__).resolve().parents[2]
    if str(studio) not in sys.path:
        sys.path.insert(0, str(studio))
    import install_lemonade_prebuilt as inst

    blob = _zip()
    pins = json.loads((studio / "lemonade_prebuilt_pins.json").read_text(encoding = "utf-8"))
    key = "windows-x64" if sys.platform == "win32" else "linux-x64"
    pins["fastflowlm"]["assets"][key]["sha256"] = hashlib.sha256(blob).hexdigest()
    pins_path = tmp_path / "pins.json"
    pins_path.write_text(json.dumps(pins))
    downloads = []

    def fake_download(url, destination, *, expected_sha256, progress, cancel):
        downloads.append(url)
        assert hashlib.sha256(blob).hexdigest() == expected_sha256
        Path(destination).write_bytes(blob)

    monkeypatch.setattr(inst, "_download", fake_download)
    monkeypatch.setattr(inst, "direct_flm", lambda: True)
    monkeypatch.setattr(inst, "host_asset_key", lambda: key)
    inst.downloads, inst.test_pins = downloads, pins_path
    return inst


def test_the_direct_install_puts_flm_where_lemond_would_and_is_reused(installer, tmp_path):
    root = tmp_path / "lemonade"
    assert installer.installed_lemond(root, installer.load_pins(installer.test_pins)) is None
    host = installer.install(root, pins_path = installer.test_pins)
    assert host == installer.FLM_HOST and host.is_file()
    flm_dir = installer.flm_dir(root)
    assert (flm_dir / "llama_npu.dll").is_file() and (flm_dir / installer.FLM_MARKER_NAME).is_file()
    assert "ROCm/FastFlowLM/releases/download/v1.0.13/" in installer.downloads[0]
    assert (
        installer.install(root, pins_path = installer.test_pins) == host
        and len(installer.downloads) == 1
    )
    assert not (root / "2026.44.1").exists()  # no Lemonade was installed


def test_the_direct_install_removes_old_lemond_and_keeps_models(installer, tmp_path):
    root = tmp_path / "lemonade"
    old = root / "2026.44.1"
    old.mkdir(parents = True)
    (old / installer.MARKER_NAME).write_text("{}")
    (old / installer.lemond_name()).write_bytes(b"MZ")
    model = root / "flm" / "models" / "Qwen3-0.6B-NPU2" / "model.q4nx"
    model.parent.mkdir(parents = True)
    model.write_bytes(b"weights")
    (root / "config").mkdir()
    (root / ".lemond-remove-2026.40.0-1").mkdir()  # an earlier interrupted removal
    installer.install(root, pins_path = installer.test_pins)
    assert not old.exists() and not (root / ".lemond-remove-2026.40.0-1").exists()
    assert model.read_bytes() == b"weights" and (root / "config").is_dir()
    assert (installer.flm_dir(root) / installer.FLM_MARKER_NAME).is_file()
    installer.install(root, pins_path = installer.test_pins)  # again: nothing left to remove
    assert sorted(p.name for p in root.iterdir()) == ["cache", "config", "flm"]


def test_a_lemond_that_cannot_be_moved_is_left_whole(installer, tmp_path, monkeypatch):
    root = tmp_path / "lemonade"
    old = root / "2026.44.1"
    old.mkdir(parents = True)
    (old / installer.MARKER_NAME).write_text("{}")

    def locked(src, dst):
        raise PermissionError(5, "Access is denied")

    monkeypatch.setattr(installer.os, "replace", locked)
    assert installer.remove_lemond_installs(root) == []
    assert (old / installer.MARKER_NAME).is_file()


def test_a_fastflowlm_pin_change_reads_as_not_installed(installer, tmp_path):
    root = tmp_path / "lemonade"
    installer.install(root, pins_path = installer.test_pins)
    pins = installer.load_pins(installer.test_pins)
    pins["fastflowlm"]["assets"][installer.host_asset_key()]["sha256"] = "0" * 64
    assert installer.installed_flm_host(root, pins) is None


@pytest.mark.parametrize(
    "platform, env, expect",
    [("win32", "", True), ("win32", "1", False), ("linux", "", True), ("linux", "1", False),
     ("darwin", "", False)],
)  # fmt: skip
def test_lemonade_runs_only_on_request(monkeypatch, platform, env, expect):
    studio = Path(__file__).resolve().parents[2]
    if str(studio) not in sys.path:
        sys.path.insert(0, str(studio))
    import install_lemonade_prebuilt as inst

    monkeypatch.setattr(inst.sys, "platform", platform)
    monkeypatch.setattr(inst.platform, "machine", lambda: "x86_64")
    monkeypatch.setenv("UNSLOTH_NPU_LEMONADE", env)
    assert inst.direct_flm() is expect


def test_lemonade_server_runs_a_py_stand_in_on_studios_python(tmp_path, monkeypatch):
    from core.inference import lemonade_server as ls

    seen = []

    def fake_popen(cmd, **kwargs):
        seen.append(cmd)
        raise FileNotFoundError("stop here")

    monkeypatch.setattr(ls.subprocess, "Popen", fake_popen)
    host = tmp_path / "flm_host.py"
    host.write_text("")
    server = ls.LemonadeServer(
        host, cache_dir = tmp_path / "c", config_dir = tmp_path / "k", flm_model_dir = tmp_path / "f"
    )
    with pytest.raises(FileNotFoundError):
        server.start(timeout = 5)
    assert seen[0][:3] == [sys.executable, str(host), str(tmp_path / "c")]


def test_the_output_relay_keeps_draining_when_printing_fails(monkeypatch):
    class Proc:
        stdout = iter(["\U0001f517 Linking model\n", "Model loaded\n", "ready\n"])

    def refuse(*args, **kwargs):
        raise UnicodeEncodeError("charmap", "x", 0, 1, "no")

    monkeypatch.setattr("builtins.print", refuse)
    tail = []
    fh._relay(Proc(), tail)
    assert tail == ["\U0001f517 Linking model", "Model loaded", "ready"]
