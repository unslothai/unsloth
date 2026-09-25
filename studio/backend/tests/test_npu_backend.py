# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Exercise NPU lifecycle and API calls against a fake lemond child process.

POSIX only: the stub executable is a shell script.
"""

from __future__ import annotations

import json
import os
import sys
import textwrap
import time
from pathlib import Path

import pytest

from core.inference import npu_backend as nb
from core.inference.lemonade_server import LemonadeServer, LemonadeUnavailable

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason = "shell-script lemond stand-in")

_FAKE = Path(__file__).parent / "fixtures" / "fake_lemond.py"


def _binary(tmp_path: Path) -> Path:
    binary = tmp_path / "bin" / "lemond"
    binary.parent.mkdir(parents = True)
    binary.write_text(
        textwrap.dedent(
            f"""\
            #!/bin/sh
            exec "{sys.executable}" "{_FAKE}" "$@"
            """
        ),
        encoding = "utf-8",
    )
    binary.chmod(0o755)
    return binary


class _Installer:
    def __init__(self, binary: Path) -> None:
        self.binary = binary
        self.installs = 0

    def installed_lemond(self, root):
        return self.binary

    def install(
        self,
        root,
        cancel = None,
    ):
        self.installs += 1
        return self.binary


@pytest.fixture
def npu(tmp_path, monkeypatch):
    installer = _Installer(_binary(tmp_path))
    monkeypatch.setattr(nb, "_installer_module", lambda: installer)
    monkeypatch.setattr(
        nb,
        "detect_amd_npu",
        lambda: {"present": True, "supported": True, "family": "XDNA2", "name": "NPU Strix Halo"},
    )
    backend = nb.LemonadeNpuBackend(root = tmp_path / "lemonade")
    backend.installer = installer
    yield backend
    backend.shutdown()


def _requests(backend) -> list[dict]:
    path = backend.root / "cache" / "requests.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines()]


def _wait_until(condition, tries: int = 300) -> None:
    """Poll a condition a test thread brings about. Counts polls: no clock in the assertion."""
    for _ in range(tries):
        if condition():
            return
        time.sleep(0.05)
    pytest.fail("the condition never held")


def _returns_promptly(call, within: float = 30.0) -> None:
    """Run ``call`` on a thread and require it back well inside the 60 s stall it must cut short."""
    import threading

    thread = threading.Thread(target = call, daemon = True)
    thread.start()
    thread.join(timeout = within)
    assert not thread.is_alive(), f"{call} waited out the stall"


def test_enable_installs_starts_and_validates(npu):
    status = npu.enable()
    assert status["state"] == "ready"
    assert status["runtime_running"] is True
    assert status["validation"]["ready"] is True
    assert npu.installer.installs == 1
    assert [r["body"] for r in _requests(npu) if r["path"] == "/v1/install"] == [
        {"recipe": "flm", "backend": "npu", "stream": False}
    ]


def test_config_is_private_and_quiet(npu):
    npu.enable()
    config = json.loads((npu.root / "config" / "config.json").read_text())
    assert config["broadcast"] is False
    assert config["auto_check_model_updates"] is False
    assert config["host"] == "127.0.0.1"
    # Not "auto", which scans the user's Hugging Face cache for GGUFs.
    assert config["models_dir"] == str(npu.root / "cache" / "models")
    assert config["flm"]["prefer_system"] is False


def test_validation_failure_names_the_fix(npu, monkeypatch):
    monkeypatch.setenv(
        "FAKE_FLM_VALIDATE", '{"ready": false, "memlock_ok": false, "all_fw_ok": true}'
    )
    with pytest.raises(nb.NpuError, match = "locked-memory limit"):
        npu.enable()
    status = npu.status()
    assert status["state"] == "failed"
    assert "memlock" in status["error"]


def test_unsupported_npu_is_refused_before_any_download(tmp_path, monkeypatch):
    installer = _Installer(_binary(tmp_path))
    monkeypatch.setattr(nb, "_installer_module", lambda: installer)
    monkeypatch.setattr(
        nb, "detect_amd_npu", lambda: {"present": True, "supported": False, "family": "XDNA1"}
    )
    backend = nb.LemonadeNpuBackend(root = tmp_path / "lemonade")
    with pytest.raises(nb.NpuError, match = "XDNA 2"):
        backend.enable()
    assert installer.installs == 0


def test_catalog_keeps_flm_chat_models_only(npu):
    npu.enable()
    ids = [model.id for model in npu.catalog()]
    assert ids == sorted(["gemma3-4b-FLM", "qwen3-0.6b-FLM", "qwen3-it-4b-FLM"])
    by_id = {model.id: model for model in npu.catalog()}
    assert by_id["gemma3-4b-FLM"].vision and not by_id["gemma3-4b-FLM"].tools
    assert by_id["qwen3-it-4b-FLM"].tools
    assert by_id["qwen3-0.6b-FLM"].reasoning
    assert by_id["qwen3-0.6b-FLM"].model_path == "lemonade:qwen3-0.6b-FLM"


def test_download_relays_progress_then_completes(npu):
    npu.enable()
    events = list(npu.download("qwen3-0.6b-FLM"))
    assert events[0]["percent"] == 40
    assert events[-1]["event"] == "complete"
    assert {m.id: m.downloaded for m in npu.catalog()}["qwen3-0.6b-FLM"] is True


@pytest.mark.parametrize(
    "pull, message", [("truncated", "ended before it completed"), ("error", "disk full")]
)
def test_a_download_without_lemonds_complete_event_fails(npu, monkeypatch, pull, message):
    monkeypatch.setenv("FAKE_LEMOND_PULL", pull)
    npu.enable()
    with pytest.raises(nb.NpuError, match = message):
        list(npu.download("qwen3-0.6b-FLM"))
    assert {m.id: m.downloaded for m in npu.catalog()}["qwen3-0.6b-FLM"] is False


def test_the_validator_is_tracked_while_it_runs(npu, monkeypatch):
    adopted: list[int] = []
    forgotten: list[int] = []
    monkeypatch.setattr(nb, "adopt_pid", adopted.append)
    monkeypatch.setattr(nb, "forget_pid", forgotten.append)
    npu.enable()
    assert len(adopted) == 1 and forgotten == adopted


def test_load_requires_a_download(npu):
    npu.enable()
    with pytest.raises(nb.NpuError, match = "not downloaded"):
        npu.load("qwen3-0.6b-FLM")
    assert not npu.is_loaded


def test_load_reports_the_context_lemond_started_with(npu, monkeypatch):
    npu.enable()
    list(npu.download("qwen3-0.6b-FLM"))
    npu.load("qwen3-0.6b-FLM")
    assert npu.is_loaded
    # Not the model's 40960 maximum: an unset length loads at the default.
    assert npu.loaded_context_length == nb.DEFAULT_CONTEXT_LENGTH
    load = [r["body"] for r in _requests(npu) if r["path"] == "/v1/load"][-1]
    assert load == {"model_name": "qwen3-0.6b-FLM", "ctx_size": 8192, "save_options": False}

    npu.load("qwen3-0.6b-FLM", 100_000)
    assert npu.loaded_context_length == 40960
    upstream = npu.upstream()
    assert upstream.model == "qwen3-0.6b-FLM"
    assert upstream.public_model == "lemonade:qwen3-0.6b-FLM"
    assert upstream.base_url.startswith("http://127.0.0.1:")
    assert upstream.api_key


def test_the_resident_reports_what_its_load_asked_for(npu):
    npu.enable()
    list(npu.download("qwen3-0.6b-FLM"))
    npu.load("qwen3-0.6b-FLM")
    assert npu.resident().context_length == nb.DEFAULT_CONTEXT_LENGTH
    assert npu.resident().requested_context_length is None
    npu.load("qwen3-0.6b-FLM", 16384)
    assert npu.resident().requested_context_length == 16384


def test_a_reader_racing_an_unload_does_not_crash(npu):
    npu.enable()
    list(npu.download("qwen3-0.6b-FLM"))
    npu.load("qwen3-0.6b-FLM")
    server = npu._server
    alive = server.is_alive

    def _unloaded_meanwhile():
        # unload() clears the record between a reader's liveness check and its read.
        npu._loaded = None
        return alive()

    server.is_alive = _unloaded_meanwhile
    try:
        assert npu.loaded_model.id == "qwen3-0.6b-FLM"
        assert npu.loaded_model is None
    finally:
        del server.is_alive


def test_a_catalog_error_body_is_an_error_not_an_empty_catalog(npu, monkeypatch):
    monkeypatch.setenv("FAKE_LEMOND_MODELS_FAILS", "1")
    npu.enable()
    with pytest.raises(nb.NpuError, match = "catalog unavailable"):
        npu.catalog()


def test_unload_then_delete(npu):
    npu.enable()
    list(npu.download("qwen3-0.6b-FLM"))
    npu.load("qwen3-0.6b-FLM")
    with pytest.raises(nb.NpuError, match = "Unload"):
        npu.delete("qwen3-0.6b-FLM")
    assert npu.unload() == "lemonade:qwen3-0.6b-FLM"
    assert not npu.is_loaded
    assert npu.unload() is None
    npu.delete("qwen3-0.6b-FLM")
    assert {m.id: m.downloaded for m in npu.catalog()}["qwen3-0.6b-FLM"] is False


@pytest.mark.parametrize("failure", ["1", "200"])
def test_a_rejected_unload_stops_lemond(npu, monkeypatch, failure):
    # "200": lemond's HTTP 200 answer with an error body.
    monkeypatch.setenv("FAKE_LEMOND_UNLOAD_FAILS", failure)
    monkeypatch.setenv("FAKE_LEMOND_DOWNLOADED", '["qwen3-0.6b-FLM"]')
    npu.enable()
    npu.load("qwen3-0.6b-FLM")
    process = npu._server._process
    assert npu.unload() == "lemonade:qwen3-0.6b-FLM"
    assert process.poll() is not None
    assert not npu.is_loaded
    npu.load("qwen3-0.6b-FLM")
    assert npu.is_loaded


@pytest.mark.parametrize("stop", ["cancel_load", "shutdown"])
def test_a_slow_load_can_be_stopped(npu, monkeypatch, stop):
    import threading

    monkeypatch.setenv("FAKE_LEMOND_LOAD_SECONDS", "60")
    npu.enable()
    list(npu.download("qwen3-0.6b-FLM"))
    errors: list[BaseException] = []

    def _load():
        try:
            npu.load("qwen3-0.6b-FLM")
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    thread = threading.Thread(target = _load)
    thread.start()
    _wait_until(lambda: any(r["path"] == "/v1/load" for r in _requests(npu)))
    assert npu.cancel_load("other-FLM") is False
    if stop == "cancel_load":
        _returns_promptly(lambda: npu.cancel_load("qwen3-0.6b-FLM"))
    else:
        _returns_promptly(npu.shutdown)
    thread.join(timeout = 30)
    assert not thread.is_alive()
    assert len(errors) == 1 and isinstance(errors[0], nb.NpuLoadCancelled)
    assert not npu.is_loaded
    assert npu.loading_model is None


@pytest.mark.parametrize("phase", ["download", "installing_flm", "validating"])
def test_shutdown_does_not_wait_for_enable(npu, monkeypatch, phase):
    import threading

    if phase == "download":
        original = npu.installer.install

        def _slow_download(root, cancel = None):
            assert cancel is not None and cancel.wait(60)
            raise RuntimeError("Lemonade download cancelled.")

        npu.installer.install = _slow_download
    elif phase == "installing_flm":
        monkeypatch.setenv("FAKE_LEMOND_INSTALL_SECONDS", "60")
    else:
        monkeypatch.setenv("FAKE_FLM_VALIDATE_SECONDS", "60")
    errors: list[BaseException] = []

    def _enable():
        try:
            npu.enable()
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    thread = threading.Thread(target = _enable)
    thread.start()
    target = "installing" if phase == "download" else phase
    _wait_until(
        lambda: npu.status()["state"] == target
        and (phase != "validating" or npu._validate_process is not None)
    )
    _returns_promptly(npu.shutdown)
    thread.join(timeout = 30)
    assert not thread.is_alive()
    assert len(errors) == 1 and isinstance(errors[0], nb.NpuError)
    assert npu.status()["runtime_running"] is False
    assert npu.status()["ready"] is False
    if phase == "download":
        npu.installer.install = original


def test_readiness_survives_a_restart_only_after_validation(npu, monkeypatch):
    npu.enable()
    assert npu.status()["ready"] is True
    npu.shutdown()
    restarted = nb.LemonadeNpuBackend(root = npu.root)
    assert restarted.status()["state"] == "idle"
    assert restarted.status()["ready"] is True

    monkeypatch.setenv("FAKE_FLM_VALIDATE", '{"ready": false, "memlock_ok": false}')
    with pytest.raises(nb.NpuError):
        restarted.enable()
    restarted.shutdown()
    assert nb.LemonadeNpuBackend(root = npu.root).status()["ready"] is False


def test_a_failed_replacement_leaves_nothing_on_the_npu(npu, monkeypatch):
    monkeypatch.setenv("FAKE_LEMOND_DOWNLOADED", '["qwen3-0.6b-FLM", "gemma3-4b-FLM"]')
    monkeypatch.setenv("FAKE_LEMOND_LOAD_FAILS_FOR", "gemma3-4b-FLM")
    npu.enable()
    npu.load("qwen3-0.6b-FLM")
    process = npu._server._process
    with pytest.raises(nb.NpuError, match = "flm failed to start"):
        npu.load("gemma3-4b-FLM")
    assert not npu.is_loaded
    assert process.poll() is not None
    npu.load("qwen3-0.6b-FLM")
    assert npu.is_loaded


def test_a_load_refused_before_touching_the_npu_keeps_the_resident(npu):
    npu.enable()
    list(npu.download("qwen3-0.6b-FLM"))
    npu.load("qwen3-0.6b-FLM")
    with pytest.raises(nb.NpuError, match = "not downloaded"):
        npu.load("gemma3-4b-FLM")
    assert npu.loaded_model.id == "qwen3-0.6b-FLM"


def test_a_cancel_during_the_final_health_check_wins(npu, monkeypatch):
    # The cancel stops lemond; the download must outlive the restart, as files on disk do.
    monkeypatch.setenv("FAKE_LEMOND_DOWNLOADED", '["qwen3-0.6b-FLM"]')
    npu.enable()
    resident_context = npu._resident_context

    def _cancelled_meanwhile(server, model_id):
        ctx = resident_context(server, model_id)
        assert npu.cancel_load(model_id) is True
        return ctx

    npu._resident_context = _cancelled_meanwhile
    with pytest.raises(nb.NpuLoadCancelled):
        npu.load("qwen3-0.6b-FLM")
    assert npu._loaded is None
    assert npu.loading_model is None
    del npu._resident_context
    npu.load("qwen3-0.6b-FLM")
    assert npu.cancel_load("qwen3-0.6b-FLM") is False
    assert npu.is_loaded


def test_a_cancel_landing_as_the_load_starts_is_kept(npu, monkeypatch):
    import threading

    monkeypatch.setenv("FAKE_LEMOND_DOWNLOADED", '["qwen3-0.6b-FLM"]')
    npu.enable()
    event = npu._load_cancelled
    results: list[bool] = []

    class _CancelledAsTheLoadStarts:
        def clear(self):
            # A cancel from another thread lands while the load is publishing itself.
            thread = threading.Thread(
                target = lambda: results.append(npu.cancel_load("qwen3-0.6b-FLM"))
            )
            thread.start()
            thread.join(timeout = 0.5)
            event.clear()

        def __getattr__(self, name):
            return getattr(event, name)

    npu._load_cancelled = _CancelledAsTheLoadStarts()
    with pytest.raises(nb.NpuLoadCancelled):
        npu.load("qwen3-0.6b-FLM")
    assert results == [True]
    assert npu._loaded is None


def test_a_refused_delete_is_reported(npu, monkeypatch):
    monkeypatch.setenv("FAKE_LEMOND_DELETE_FAILS", "1")
    npu.enable()
    list(npu.download("qwen3-0.6b-FLM"))
    with pytest.raises(nb.NpuError, match = "file in use"):
        npu.delete("qwen3-0.6b-FLM")
    assert {m.id: m.downloaded for m in npu.catalog()}["qwen3-0.6b-FLM"] is True


def test_a_restarted_runtime_reports_nothing_loaded(npu):
    npu.enable()
    list(npu.download("qwen3-0.6b-FLM"))
    npu.load("qwen3-0.6b-FLM")
    npu._server._process.kill()
    npu._server._process.wait()
    assert not npu.is_loaded
    assert npu.catalog()
    assert npu.status()["loaded_model"] is None


def test_requests_without_the_key_are_refused(npu):
    import httpx

    npu.enable()
    base = npu._server.base_url
    assert httpx.get(f"{base}/v1/health", trust_env = False).status_code == 401
    assert httpx.get(f"{base}/live", trust_env = False).status_code == 200


def test_stop_ends_the_process(npu):
    npu.enable()
    process = npu._server._process
    npu.shutdown()
    assert process.poll() is not None
    assert npu.status()["runtime_running"] is False


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason = "PR_SET_PDEATHSIG is Linux")
def test_lemond_dies_with_a_killed_owner_even_if_it_ignores_sigterm(tmp_path):
    """A SIGKILLed Unsloth must not leave lemond (and the model FastFlowLM holds) behind.

    The death signal is SIGKILL, not SIGTERM: this lemond ignores SIGTERM outright, and the real one
    took about 9 s over it with a model loaded."""
    import signal
    import subprocess
    import time

    binary = _binary(tmp_path)
    owner_src = textwrap.dedent(
        f"""\
        import sys, time
        from pathlib import Path
        sys.path.insert(0, {str(Path(__file__).resolve().parents[1])!r})
        from core.inference.lemonade_server import LemonadeServer
        root = Path({str(tmp_path)!r})
        server = LemonadeServer(
            Path({str(binary)!r}),
            cache_dir = root / "cache",
            config_dir = root / "config",
            flm_model_dir = root / "flm",
        )
        server.start(timeout = 30)
        print(server.pid, flush = True)
        time.sleep(600)
        """
    )
    env = dict(os.environ, FAKE_LEMOND_IGNORE_SIGTERM = "1")
    owner = subprocess.Popen(
        [sys.executable, "-c", owner_src], stdout = subprocess.PIPE, text = True, env = env
    )
    try:
        lemond_pid = int(owner.stdout.readline())
        owner.send_signal(signal.SIGKILL)
        owner.wait(timeout = 10)
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            try:
                os.kill(lemond_pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.1)
        else:
            os.kill(lemond_pid, signal.SIGKILL)
            pytest.fail("lemond outlived its SIGKILLed owner")
    finally:
        if owner.poll() is None:
            owner.kill()


def test_a_server_that_never_answers_is_reported(tmp_path):
    binary = tmp_path / "lemond"
    binary.write_text("#!/bin/sh\necho 'bind failed'\nexit 3\n", encoding = "utf-8")
    binary.chmod(0o755)
    server = LemonadeServer(
        binary,
        cache_dir = tmp_path / "cache",
        config_dir = tmp_path / "config",
        flm_model_dir = tmp_path / "flm",
    )
    with pytest.raises(LemonadeUnavailable, match = "bind failed"):
        server.start(timeout = 10)
    assert not server.is_alive()


def test_stop_interrupts_a_start_that_never_becomes_ready(tmp_path):
    import threading

    binary = tmp_path / "lemond"
    binary.write_text("#!/bin/sh\nexec sleep 60\n", encoding = "utf-8")
    binary.chmod(0o755)
    server = LemonadeServer(
        binary,
        cache_dir = tmp_path / "cache",
        config_dir = tmp_path / "config",
        flm_model_dir = tmp_path / "flm",
    )
    errors: list[BaseException] = []

    def _start():
        try:
            server.start(timeout = 60)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    thread = threading.Thread(target = _start)
    thread.start()
    _wait_until(server.is_alive)
    _returns_promptly(server.stop)
    thread.join(timeout = 30)
    assert len(errors) == 1 and isinstance(errors[0], LemonadeUnavailable)
    assert not server.is_alive()
    server.close()


@pytest.mark.parametrize(
    "path, expected",
    [
        ("lemonade:qwen3-0.6b-FLM", "qwen3-0.6b-FLM"),
        ("lemonade:  gemma3-4b-FLM ", "gemma3-4b-FLM"),
    ],
)
def test_model_id_from_path(path, expected):
    assert nb.is_npu_model_path(path)
    assert nb.model_id_from_path(path) == expected


@pytest.mark.parametrize("path", ["lemonade:", "lemonade:../x", "lemonade:a/b"])
def test_model_id_from_path_rejects_non_ids(path):
    with pytest.raises(nb.NpuError):
        nb.model_id_from_path(path)


def test_other_model_paths_are_not_npu():
    assert not nb.is_npu_model_path("unsloth/Qwen3-0.6B-GGUF")
    assert not nb.is_npu_model_path(None)
