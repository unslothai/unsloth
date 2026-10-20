# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Third-round cells for the DSpark/DFlash sidecar reserve and the Metal OOM recovery.

Every emitter now asks one question of the child as launched (``_sidecar_reserve_margin_flags``):
which drafter the final argv + env resolve to, whether the fitter runs, and which margin the user
already asked for. The cells pin the spellings and paths that used to answer it on their own.
"""

from __future__ import annotations

import inspect
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

import test_llama_cpp_placement as _placement  # noqa: E402
import test_spec_sidecar_fit_reserve_r2 as _r2  # noqa: E402
from core.inference import llama_cpp as _lc  # noqa: E402
from core.inference.llama_cpp import GgufLoadIntent, LlamaCppBackend  # noqa: E402

_GIB = 1024 * 1024 * 1024
_RESERVE_FLOOR = 1024 + 8 * 1024  # base margin + the 8 GiB sidecar


@pytest.fixture(autouse = True)
def _studio_home(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio_home"))


def _margin(cmd):
    return int(float(_r2._last(cmd, "--fit-target") or 1024))


@pytest.mark.parametrize(
    "args, env",
    [
        (["--model-draft", "/d.gguf"], {}),
        (["-md", "/d.gguf"], {}),
        (["--spec-draft-model", "/d.gguf"], {}),
        (["--model-draft=/d.gguf"], {}),
        ([], {"LLAMA_ARG_SPEC_DRAFT_MODEL": "/d.gguf"}),
    ],
    ids = ["long", "short", "spec-draft-model", "equals", "env"],
)
def test_every_draft_spelling_gets_the_reserve(args, env):
    flags = _lc._sidecar_reserve_margin_flags(["--fit", "on", *args], (), env, 1024, 8192)
    assert flags == ["--fit-target", str(1024 + 8192)]


@pytest.mark.parametrize(
    "args, env",
    [(["--fit", "off"], {}), ([], {"LLAMA_ARG_FIT": "off"}), (["-fit", "off"], {})],
    ids = ["argv", "env", "short"],
)
def test_no_reserve_when_the_fitter_is_off(args, env):
    assert _lc._sidecar_reserve_margin_flags(["-md", "/d.gguf", *args], (), env, 1024, 8192) == []


def test_no_reserve_without_a_drafter():
    assert _lc._sidecar_reserve_margin_flags(["--fit", "on"], (), {}, 1024, 8192) == []


@pytest.mark.parametrize(
    "user_args, env, want",
    [
        ((), {"LLAMA_ARG_FIT_TARGET": "12000"}, 12000 + 8192),
        (("--fit-target", "16384"), {}, 16384 + 8192),
        (("-fitt", "512/20000"), {}, 20000 + 8192),
    ],
    ids = ["env", "argv", "per-device-list"],
)
def test_the_reserve_sits_on_top_of_the_users_margin(user_args, env, want):
    flags = _lc._sidecar_reserve_margin_flags(
        ["-md", "/d.gguf", *user_args], user_args, env, 1024, 8192
    )
    assert flags == ["--fit-target", str(want)]


@pytest.mark.parametrize(
    "extras",
    [["--fit", "on"], ["--fit=on"], ["-fit", "on"]],
    ids = ["spaced", "equals", "short"],
)
def test_extras_that_turn_the_fitter_on_get_the_reserve(tmp_path, extras):
    """Studio planned a full offload (--fit off); the extras turn the fitter back on next to a
    drafter it cannot size, so the reserve has to follow (finding 1)."""
    backend, gguf = _r2._cuda_backend(tmp_path)
    backend._select_gpus = lambda *a, **k: ([0], False)
    sidecar = tmp_path / "dspark-model-Q8_0.gguf"
    sidecar.write_bytes(b"draft")
    cmd = _placement._launch(
        backend, gguf, speculative_type = "dspark", dspark_draft_path = str(sidecar), extra_args = extras
    )["cmd"]
    assert _margin(cmd) >= _RESERVE_FLOOR


def test_an_inherited_env_margin_is_kept_under_the_reserve(tmp_path, monkeypatch):
    """LLAMA_ARG_FIT_TARGET=12000 must not be lowered to Studio's own margin + reserve
    (findings 3 and 7)."""
    monkeypatch.setenv("LLAMA_ARG_FIT_TARGET", "12000")
    backend, gguf = _r2._cuda_backend(tmp_path)
    sidecar = tmp_path / "dspark-model-Q8_0.gguf"
    sidecar.write_bytes(b"draft")
    cmd = _placement._launch(
        backend, gguf, speculative_type = "dspark", dspark_draft_path = str(sidecar)
    )["cmd"]
    assert _margin(cmd) >= 12000 + 8 * 1024


def test_the_startup_retry_reads_the_child_not_a_literal_flag():
    """Findings 2 and 4: the --fit on retry must resolve the drafter through every alias and
    env spelling and keep the child env's margin. Checked at the source: the retry only runs
    behind a real startup crash."""
    src = inspect.getsource(LlamaCppBackend.load_model)
    retry = src[src.index("retrying once with --fit on so it can offload") :]
    retry = retry[: retry.index("_did_fit_retry = True")]
    assert "_sidecar_reserve_margin_flags(" in retry
    assert '"--model-draft" in _run' not in retry
    assert "fit_target_margin_in(extra_args, None)" not in retry


@pytest.mark.parametrize("draft_device, reserved", [("Vulkan0", True), ("Vulkan1", False)])
def test_a_stale_device_extra_does_not_decide_the_draft_device(tmp_path, draft_device, reserved):
    """gpu_ids=[0] owns placement and strips the stale --device Vulkan1 at emission, so the
    draft shares the target exactly when it is pinned to Vulkan0 (finding 5)."""
    backend, gguf = _r2._cuda_backend(
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
        extra_args = ["--device", "Vulkan1", "--device-draft", draft_device],
    )["cmd"]
    assert (_margin(cmd) >= _RESERVE_FLOOR) is reserved


def test_the_drafter_free_fallback_keeps_the_reserve_for_a_user_drafter(tmp_path):
    """The speculative fallback removes Studio's own block; a bare user --model-draft survives
    and still loads, so its reserve must too (finding 6)."""
    backend, gguf = _r2._cuda_backend(tmp_path)
    sidecar = tmp_path / "dspark-model-Q8_0.gguf"
    sidecar.write_bytes(b"draft")
    custom = tmp_path / "custom-dspark-Q8_0.gguf"
    custom.write_bytes(b"draft")
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
        backend.load_model(
            GgufLoadIntent(
                gguf_path = str(gguf),
                model_identifier = "test",
                n_ctx = 4096,
                speculative_type = "dspark",
                dspark_draft_path = str(sidecar),
                extra_args = ("--model-draft", str(custom)),
            )
        )
    retry = cmds[-1]
    if "draft-dspark" in retry:
        pytest.skip("the user drafter kept the speculative block; no fallback ran")
    assert str(custom) in retry
    assert _margin(retry) >= _RESERVE_FLOOR


def test_the_metal_auto_drop_recomputes_the_embedding_discount(tmp_path, monkeypatch):
    """The pageable-embedding discount was decided with the drafter engaged (excluded NextN
    blocks forfeit it); once Auto drops the drafter it must be decided again, or Auto keeps a
    smaller context than Off (finding 8)."""

    def embd(self, *a, mtp_may_engage = False, **k):
        return 0 if mtp_may_engage else 3 * _GIB

    def run(mode):
        monkeypatch.setattr(LlamaCppBackend, "_metal_demand_paged_embedding_bytes", embd)
        return _r2._metal_cmd(tmp_path, monkeypatch, mode, budget_gib = 9.1, sub = f"embd-{mode}")[0]

    auto, off = run("auto"), run("off")
    assert "--model-draft" not in auto
    assert _r2._last(auto, "-c") == _r2._last(off, "-c")


@pytest.mark.parametrize(
    "message, confirmed",
    [
        ("Compute error.", True),
        ("decode() failed: failed to process speculative batch", True),
        ("the request exceeds the available context size", False),
    ],
    ids = ["target", "draft", "unrelated"],
)
def test_the_draft_context_failure_confirms_the_poison(message, confirmed):
    backend = LlamaCppBackend()
    backend._port = 1
    resp = type(
        "Resp",
        (),
        {"status_code": 500, "json": lambda self: {"error": {"message": message}}},
    )()
    with patch.object(_lc.httpx, "post", return_value = resp):
        assert backend._metal_compute_failure_confirmed() is confirmed


@pytest.mark.parametrize("retired", [False, True])
def test_a_retired_child_is_not_reused_by_an_identical_load(tmp_path, retired):
    """After a first retirement the killed child is still recorded and _healthy is still true:
    an identical explicit load must spawn a replacement, not reuse it (finding 10)."""
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"gguf")
    backend = LlamaCppBackend()
    proc = type(
        "P",
        (),
        {"poll": lambda self: None, "kill": lambda self: None, "wait": lambda self, timeout = None: 0},
    )()
    backend._process = proc
    backend._healthy = True
    backend._model_identifier = "m"
    backend._gguf_path = str(gguf)
    if retired:
        backend._retire_device_lost_server(proc)
    intent = GgufLoadIntent(gguf_path = str(gguf), model_identifier = "m")
    assert backend.matches_load_source(intent) is (not retired)
