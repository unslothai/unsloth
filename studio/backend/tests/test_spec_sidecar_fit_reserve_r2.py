# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Second-round cells for the DSpark/DFlash sidecar reserve and the Metal OOM recovery.

Each cell pins one path where the reserve or the retirement disagreed with what the
child really loads: the target-only bound on Metal, a drafter-free startup retry, a
user-owned bare --model-draft, request text echoed into the server log, the --fit on
startup retry, a user --fit-target in the extras, a draft pinned to another device, and
a terminal second OOM.
"""

from __future__ import annotations

import inspect
import subprocess
import sys
import threading
from pathlib import Path
from unittest.mock import patch

import pytest

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

import test_llama_cpp_metal_poisoned_respawn as _respawn  # noqa: E402
import test_llama_cpp_placement as _placement  # noqa: E402
import test_metal_explicit_context_guard as _metal  # noqa: E402
from core.inference.llama_cpp import GgufLoadIntent, LlamaCppBackend  # noqa: E402

harness = _respawn.harness  # the respawn fixture, reused as-is

_MIB = 1024 * 1024
_GIB = 1024 * _MIB
_CAPS = {
    "supports_dspark": True,
    "supports_dflash": True,
    "supports_fit_target": True,
    "supports_fit_ctx": True,
    "supports_kv_unified": True,
    "spec_draft_n_max_flag": "--spec-draft-n-max",
}


@pytest.fixture(autouse = True)
def _studio_home(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio_home"))


def _last(cmd, name):
    """Last-wins value of ``name``, in either the spaced or the ``=`` spelling."""
    value = None
    for i, tok in enumerate(cmd):
        if tok == name and i + 1 < len(cmd):
            value = cmd[i + 1]
        elif tok.startswith(name + "="):
            value = tok.split("=", 1)[1]
    return value


def _metal_cmd(tmp_path, monkeypatch, mode, *, budget_gib, extra_args = None, sub = None):
    sub = tmp_path / (sub or mode)
    sub.mkdir()
    sidecar = sub / "dspark-model-Q8_0.gguf"
    sidecar.write_bytes(b"draft")
    real_init = GgufLoadIntent.__init__

    def init(self, *args, **kwargs):
        kwargs.setdefault("dspark_draft_path", str(sidecar))
        kwargs.setdefault("speculative_type", mode)
        real_init(self, *args, **kwargs)

    monkeypatch.setattr(GgufLoadIntent, "__init__", init)
    monkeypatch.setattr(
        LlamaCppBackend,
        "probe_server_capabilities",
        classmethod(lambda cls, binary = None: dict(_CAPS)),
    )
    backend = LlamaCppBackend()
    cmd = _metal._launch(
        sub,
        monkeypatch,
        n_ctx = 0,
        backend = backend,
        real_fit = True,
        budget_bytes = int(budget_gib * _GIB),
        wired_bytes = int(budget_gib * _GIB),
        weights_bytes = 4 * _GIB,
        extra_args = extra_args,
    )["cmd"]
    monkeypatch.undo()
    return cmd, backend


def test_metal_target_only_bound_leaves_out_the_drafter_compute(tmp_path, monkeypatch):
    """Within 224 MiB of the target-only footprint the drafter's compute charge must not
    keep the sidecar: Auto then has to match a load with speculation off."""
    auto, _ = _metal_cmd(tmp_path, monkeypatch, "auto", budget_gib = 9.1)
    off, _ = _metal_cmd(tmp_path, monkeypatch, "off", budget_gib = 9.1)
    assert "--model-draft" not in auto
    assert _last(auto, "-c") == _last(off, "-c")


def test_metal_auto_keeps_a_user_owned_bare_model_draft(tmp_path, monkeypatch):
    """A bare --model-draft in the extras still loads (has_dft()), so Auto must not mark
    it dropped and release its reserve while the child loads it anyway."""
    custom = tmp_path / "custom-dspark-Q8_0.gguf"
    custom.write_bytes(b"draft")
    cmd, backend = _metal_cmd(
        tmp_path, monkeypatch, "auto", budget_gib = 12, extra_args = ["--model-draft", str(custom)]
    )
    assert str(custom) in cmd
    assert backend.spec_fallback_reason != "drafter_no_vram"


def _cuda_backend(tmp_path, memory = ((0, 24_000, 24_000),), vulkan = False):
    backend, gguf = _placement._backend(tmp_path, vulkan = vulkan, memory = list(memory))
    backend._select_gpus = lambda *a, **k: (None, True)
    backend.probe_server_capabilities = lambda _binary = None: dict(_CAPS)
    backend._get_gguf_size_bytes = lambda path: 8 * _GIB if "dspark" in str(path) else 1024
    return backend, gguf


def test_the_drafter_free_startup_retry_drops_the_reserve(tmp_path):
    backend, gguf = _cuda_backend(tmp_path)
    sidecar = tmp_path / "dspark-model-Q8_0.gguf"
    sidecar.write_bytes(b"draft")
    cmds = []

    def fake_popen(cmd, **kwargs):
        if not cmd or str(cmd[0]) != "/fake/llama-server":
            return _placement._REAL_POPEN(cmd, **kwargs)
        cmds.append(list(cmd))
        return type(
            "Process",
            (),
            {
                "pid": 123,
                "stdout": (),
                "poll": lambda self: None,
                "terminate": lambda self: None,
                "wait": lambda self, timeout = None: 0,
                "kill": lambda self: None,
            },
        )()

    def fake_health(timeout = None, **_kw):
        backend._stdout_lines = []
        return "draft-dspark" not in cmds[-1]

    backend._wait_for_health = fake_health
    with patch.object(subprocess, "Popen", side_effect = fake_popen):
        assert backend.load_model(
            GgufLoadIntent(
                gguf_path = str(gguf),
                model_identifier = "test",
                n_ctx = 4096,
                speculative_type = "dspark",
                dspark_draft_path = str(sidecar),
            )
        )
    first, retry = cmds[0], cmds[-1]
    assert int(_last(first, "--fit-target")) >= 1024 + 8 * 1024
    assert "--model-draft" not in retry
    assert int(_last(retry, "--fit-target") or 1024) < 1024 + 8 * 1024


@pytest.mark.parametrize(
    "extras",
    [["--fit-target", "1024"], ["-fitt", "1024"], ["--fit-target=1024"]],
    ids = ["spaced", "short", "equals"],
)
def test_a_user_fit_target_keeps_the_reserve(tmp_path, extras):
    backend, gguf = _cuda_backend(tmp_path)
    sidecar = tmp_path / "dspark-model-Q8_0.gguf"
    sidecar.write_bytes(b"draft")
    cmd = _placement._launch(
        backend, gguf, speculative_type = "dspark", dspark_draft_path = str(sidecar), extra_args = extras
    )["cmd"]
    margins = [
        cmd[i + 1] if tok in ("--fit-target", "-fitt") else tok.split("=", 1)[1]
        for i, tok in enumerate(cmd)
        if tok in ("--fit-target", "-fitt") or tok.startswith("--fit-target=")
    ]
    assert int(margins[-1]) >= 1024 + 8 * 1024


@pytest.mark.parametrize("draft_device, reserved", [("Vulkan1", False), ("Vulkan0", True)])
def test_a_draft_pinned_to_another_vulkan_device_is_not_reserved_on_the_target(
    tmp_path, draft_device, reserved
):
    backend, gguf = _cuda_backend(
        tmp_path, memory = ((0, 24_000, 24_000), (1, 24_000, 24_000)), vulkan = True
    )
    sidecar = tmp_path / "dspark-model-Q8_0.gguf"
    sidecar.write_bytes(b"draft")
    cmd = _placement._launch(
        backend,
        gguf,
        speculative_type = "dspark",
        dspark_draft_path = str(sidecar),
        gpu_ids = [0],
        extra_args = ["--device-draft", draft_device],
    )["cmd"]
    margin = int(_last(cmd, "--fit-target") or 1024)
    assert (margin >= 1024 + 8 * 1024) is reserved


def test_the_fit_on_startup_retry_carries_the_reserve():
    """The --fit off launch emits no margin, so the retry that turns the fitter on next to
    the still-loaded drafter has to add it. Checked at the source: the retry only runs
    behind a real startup crash."""
    src = inspect.getsource(LlamaCppBackend.load_model)
    # Every attempt of _spawn_and_wait (the retry re-enters its loop) reconciles the
    # reserve right before the child is spawned.
    spawn = src[src.index("def _spawn_and_wait(") :]
    spawn = spawn[: spawn.index("_last_spawn_cmd = list(run_cmd)")]
    assert "_sidecar_fit_reserve_eligible_mib" in spawn
    assert "_sidecar_reserve_margin_flags(" in spawn


def test_echoed_marker_without_a_compute_failure_does_not_retire(harness):
    """A grammar parse error prints the request's grammar, so a forged E-line can carry
    the marker: retirement needs the child to really fail compute."""
    backend, first, loads = harness(
        _respawn._Intent(speculative_type = "off"), "none", [_respawn.METAL_OOM]
    )
    backend._metal_compute_failure_confirmed = lambda: False
    threading.Thread(target = backend._drain_stdout, daemon = True).start()
    try:
        first.wait(timeout = 1.5)
    except subprocess.TimeoutExpired:
        pass
    assert first.poll() is None
    assert getattr(backend, "_metal_oom_strikes", None) is None


def test_retirement_keeps_a_user_owned_drafter(harness):
    """Auto plus a bare --model-draft: an Off reload would still load that file, so the
    recovery must not claim a drafter-free reload."""
    intent = _respawn._Intent(extra_args = ("--model-draft", "/models/custom-dspark.gguf"))
    backend, first, loads = harness(intent, "draft-dspark")
    backend._retire_device_lost_server(first, metal_poison = {"draft": True})
    assert first.poll() is not None
    # The drafter-free reload runs on a worker thread: give it the time it would take.
    _respawn._wait(lambda: bool(loads), timeout = 2.0)
    assert all(load.speculative_type != "off" for load in loads)


def test_a_terminal_second_oom_is_not_left_healthy(harness):
    intent = _respawn._Intent(speculative_type = "off")
    backend, first, _loads = harness(intent, "none")
    backend._retire_device_lost_server(first, metal_poison = {"draft": False})
    assert backend._respawn_if_dead() is True
    backend._retire_device_lost_server(backend._process, metal_poison = {"draft": False})
    assert backend._healthy is False
