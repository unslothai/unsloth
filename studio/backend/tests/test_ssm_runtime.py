# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for utils.ssm_runtime: the inference-side auto-install of SSM/Mamba kernels.

Covers detection, wheel-first install, idempotency, the failure path, the inference
worker wiring, and that the training worker shares the same tables and pinned kernels.
"""

import json
import contextlib
import sys
import types
from pathlib import Path
from unittest import mock

import pytest

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from utils import ssm_runtime  # noqa: E402

_REAL_MAMBA_SSM_FLOOR_REASON = ssm_runtime._mamba_ssm_floor_reason


@pytest.fixture(autouse = True)
def _clear_offline_environment(monkeypatch, tmp_path):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising = False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising = False)
    monkeypatch.delenv("UV_OFFLINE", raising = False)
    # Per-process memos: each test is its own process here.
    monkeypatch.setattr(ssm_runtime, "_TORCH_ENV", {})
    monkeypatch.setattr(ssm_runtime, "_FAILED_INSTALLS", set())
    monkeypatch.setattr(ssm_runtime, "_MODEL_CONFIGS", {})
    monkeypatch.setattr(ssm_runtime, "_REMOTE_SOURCES", {})
    monkeypatch.setattr(ssm_runtime, "_NATIVE_MODELING_CACHE", {})
    from utils import kernel_install

    monkeypatch.setattr(
        kernel_install, "_kernel_lock_path", lambda name: str(tmp_path / f"{name}.lock")
    )
    # Every copy of the module (and any child process) uses the test's folder too.
    monkeypatch.setenv("UNSLOTH_KERNEL_LOCK_DIR", str(tmp_path / "locks"))
    # This host's GPU must not decide the optional-mamba floor; its own tests set it.
    monkeypatch.setattr(ssm_runtime, "_mamba_ssm_floor_reason", lambda: None)
    # Nor its torch build: a CUDA torch unless a test says otherwise.
    monkeypatch.setattr(
        ssm_runtime,
        "probe_torch_wheel_env",
        lambda **k: {"cuda_version": "12.8", "cuda_major": "12"},
    )


class _Result:
    def __init__(
        self,
        returncode = 0,
        stdout = "",
    ):
        self.returncode = returncode
        self.stdout = stdout


# ── detection ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "name",
    [
        "unsloth/NVIDIA-Nemotron-3-Nano-4B",
        "unsloth/Nemotron-3-Nano-30B-A3B",
        "nvidia/Nemotron-H-8B",
        "tiiuae/Falcon-H1-0.5B-Instruct",
        "ibm-granite/granite-4.0-h-micro",
        "ibm/granitemoehybrid-test",
        "state-spaces/mamba-2.8b-hf",
        "ai21labs/Jamba-v0.1",
        "Zyphra/Zamba2-7B",
        "ibm/Bamba-9B",
        "tiiuae/falcon-mamba-7b",
    ],
)
def test_ssm_models_detected(name):
    assert ssm_runtime.model_is_ssm(name) is True
    # every SSM model also needs causal-conv1d
    assert ssm_runtime.model_wants_causal_conv1d(name) is True


@pytest.mark.parametrize(
    "name",
    ["LiquidAI/LFM2-1.2B", "LiquidAI/LFM2-8B-A1B"],
)
def test_causal_conv1d_only_models(name):
    # linear-attention hybrids need causal-conv1d but not mamba-ssm
    assert ssm_runtime.model_wants_causal_conv1d(name) is True
    assert ssm_runtime.model_is_ssm(name) is False


@pytest.mark.parametrize(
    "name",
    [
        "unsloth/Llama-3.2-1B-Instruct",
        "unsloth/Qwen2.5-7B",
        "unsloth/gemma-3-4b-it",
        "",
        None,
    ],
)
def test_non_ssm_models_not_detected(name):
    assert ssm_runtime.model_is_ssm(name) is False
    assert ssm_runtime.model_wants_causal_conv1d(name) is False


# ── ssm_probe_identifier: match a real model id, never an arbitrary name ───────


def test_probe_lora_uses_base_not_adapter_name():
    # A plain-Llama LoRA whose adapter id contains an SSM substring is not SSM.
    probe = ssm_runtime.ssm_probe_identifier("user/falcon-h1-lora", "meta-llama/Llama-3-8B")
    assert probe == "meta-llama/Llama-3-8B"
    assert ssm_runtime.model_is_ssm(probe) is False


def test_probe_lora_on_ssm_base_detected():
    probe = ssm_runtime.ssm_probe_identifier("user/my-adapter", "nvidia/Nemotron-H-8B")
    assert ssm_runtime.model_is_ssm(probe) is True


def test_probe_plain_hf_id_unchanged():
    assert ssm_runtime.ssm_probe_identifier("nvidia/Nemotron-H-8B") == "nvidia/Nemotron-H-8B"


def test_probe_local_path_uses_basename(tmp_path):
    # Parent folders are arbitrary: a Llama checkpoint under a falcon-h1 dir is not SSM.
    d = tmp_path / "falcon-h1-experiment" / "llama-checkpoint"
    d.mkdir(parents = True)
    probe = ssm_runtime.ssm_probe_identifier(str(d))
    assert probe == "llama-checkpoint"
    assert ssm_runtime.model_is_ssm(probe) is False


def test_probe_local_ssm_checkpoint_basename_detected(tmp_path):
    d = tmp_path / "runs" / "nemotron-h-finetune"
    d.mkdir(parents = True)
    assert ssm_runtime.model_is_ssm(ssm_runtime.ssm_probe_identifier(str(d))) is True


# ── ensure_ssm_runtime behaviour ─────────────────────────────────────────────


def test_noop_for_non_ssm_model(monkeypatch):
    calls = []
    monkeypatch.setattr(ssm_runtime, "_install_kernel", lambda **k: calls.append(k) or True)
    ssm_runtime.ensure_ssm_runtime("unsloth/Llama-3.2-1B-Instruct", run = lambda *a, **k: _Result())
    assert calls == []  # nothing installed for a plain transformer


def test_ssm_model_installs_causal_then_mamba(monkeypatch):
    order = []

    def fake_install(*, import_name, **_):
        order.append(import_name)
        return True

    monkeypatch.setattr(ssm_runtime, "_install_kernel", fake_install)
    monkeypatch.setattr(ssm_runtime.sys, "platform", "linux")
    ssm_runtime.ensure_ssm_runtime("unsloth/NVIDIA-Nemotron-3-Nano-4B")
    assert order == ["causal_conv1d", "mamba_ssm"]


def test_causal_only_model_skips_mamba(monkeypatch):
    order = []
    monkeypatch.setattr(
        ssm_runtime,
        "_install_kernel",
        lambda *, import_name, **_: order.append(import_name) or True,
    )
    ssm_runtime.ensure_ssm_runtime("LiquidAI/LFM2-1.2B")
    assert order == ["causal_conv1d"]


def test_failure_raises_runtime_error(monkeypatch):
    # A required-family model this transformers has no native implementation of cannot load
    # without mamba-ssm, so a failed install is fatal (cryptic mid-load import otherwise).
    monkeypatch.setattr(ssm_runtime, "_install_kernel", lambda **k: False)
    monkeypatch.setattr(ssm_runtime, "native_modeling_available", lambda model_type: False)
    with pytest.raises(RuntimeError):
        ssm_runtime.ensure_ssm_runtime("unsloth/Nemotron-3-Nano-30B-A3B")


def test_causal_only_install_failure_is_not_fatal(monkeypatch):
    # LFM2 wants causal-conv1d but falls back to torch; a failed install must
    # not block the load (best-effort, mirrors training).
    monkeypatch.setattr(ssm_runtime, "_install_kernel", lambda **k: False)
    ssm_runtime.ensure_ssm_runtime("LiquidAI/LFM2-1.2B")  # no raise


def test_ssm_causal_failure_nonfatal_when_mamba_ok(monkeypatch):
    # causal-conv1d is best-effort even for a true SSM model; only mamba-ssm is fatal.
    monkeypatch.setattr(
        ssm_runtime, "_install_kernel", lambda *, import_name, **_: import_name == "mamba_ssm"
    )
    ssm_runtime.ensure_ssm_runtime("unsloth/NVIDIA-Nemotron-3-Nano-4B")  # no raise


def test_install_kernel_idempotent_when_present(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr(ssm_runtime, "_is_importable", lambda name: True)
    called = []
    monkeypatch.setattr(ssm_runtime, "url_exists", lambda u: called.append("url") or True)
    ok = ssm_runtime._install_kernel(
        import_name = "mamba_ssm",
        display_name = "mamba-ssm",
        pypi_name = "mamba-ssm",
        package_version = "2.3.1",
        release_tag = "v2.3.1",
        release_base_url = "x",
        status_cb = None,
        run = lambda *a, **k: _Result(),
    )
    assert ok is True
    assert called == []  # short-circuits before touching the network


# UV_OFFLINE too: uv refuses the index and install_wheel's pip fallback would not.
@pytest.mark.parametrize(
    "offline_variable", ["HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "UV_OFFLINE"]
)
def test_install_kernel_skips_all_install_work_offline(monkeypatch, offline_variable):
    monkeypatch.setenv(offline_variable, "on")
    monkeypatch.setattr(ssm_runtime, "_is_importable", lambda name: False)
    torch_probe = mock.Mock()
    wheel_builder = mock.Mock()
    url_probe = mock.Mock()
    wheel_install = mock.Mock()
    process_run = mock.Mock()
    monkeypatch.setattr(ssm_runtime, "probe_torch_wheel_env", torch_probe)
    monkeypatch.setattr(ssm_runtime, "direct_wheel_url", wheel_builder)
    monkeypatch.setattr(ssm_runtime, "url_exists", url_probe)
    monkeypatch.setattr(ssm_runtime, "install_wheel", wheel_install)

    installed = ssm_runtime._install_kernel(
        import_name = "causal_conv1d",
        display_name = "causal-conv1d",
        pypi_name = "causal-conv1d",
        package_version = "1.6.1",
        release_tag = "v1.6.1.post4",
        release_base_url = "https://example.invalid/releases",
        status_cb = None,
        run = process_run,
    )

    assert installed is False
    torch_probe.assert_not_called()
    wheel_builder.assert_not_called()
    url_probe.assert_not_called()
    wheel_install.assert_not_called()
    process_run.assert_not_called()


def test_install_kernel_uses_prebuilt_wheel(monkeypatch):
    # not importable before install (or once the lock is held), importable after the wheel lands
    states = iter([False, False, True])
    monkeypatch.setattr(ssm_runtime, "_is_importable", lambda name: next(states))
    monkeypatch.setattr(
        ssm_runtime,
        "probe_torch_wheel_env",
        lambda timeout = 30, include_windows = False: {"x": "y", "cuda_version": "12.8"},
    )
    seen = {}
    monkeypatch.setattr(
        ssm_runtime,
        "direct_wheel_url",
        lambda **k: seen.update(k) or "https://example/mamba_ssm-2.3.1-cp313.whl",
    )
    monkeypatch.setattr(ssm_runtime, "url_exists", lambda u: True)
    installed = {}

    def fake_install_wheel(url, **k):
        installed["url"] = url
        return [("uv", _Result(returncode = 0))]

    monkeypatch.setattr(ssm_runtime, "install_wheel", fake_install_wheel)
    ran = []
    ok = ssm_runtime._install_kernel(
        import_name = "mamba_ssm",
        display_name = "mamba-ssm",
        pypi_name = "mamba-ssm",
        package_version = "2.3.1",
        release_tag = "v2.3.1",
        release_base_url = "https://github.com/state-spaces/mamba/releases/download",
        status_cb = None,
        run = lambda *a, **k: ran.append(a) or _Result(),
    )
    assert ok is True
    assert installed["url"].endswith(".whl")
    assert seen["filename_prefix"] == "mamba_ssm"
    assert ran == []  # wheel succeeded; no PyPI source build


def test_install_kernel_heartbeats_during_prebuilt_wheel(monkeypatch):
    # Keep the parent alive during a quiet wheel install (#9398).
    import threading
    import time

    monkeypatch.setattr(ssm_runtime, "_HEARTBEAT_SECONDS", 0.05)
    monkeypatch.setattr(ssm_runtime, "_is_importable", lambda name: False)
    monkeypatch.setattr(
        ssm_runtime,
        "probe_torch_wheel_env",
        lambda timeout = 30, include_windows = False: {"cuda_version": "12.8"},
    )
    monkeypatch.setattr(
        ssm_runtime,
        "direct_wheel_url",
        lambda **k: "https://example/causal_conv1d-1.6.1.whl",
    )
    monkeypatch.setattr(ssm_runtime, "url_exists", lambda u: True)

    statuses = []
    released = threading.Event()

    def slow_install_wheel(url, **k):
        # Hold long enough for at least one heartbeat tick.
        assert released.wait(1.0)
        return [("uv", _Result(returncode = 1, stdout = "nope"))]

    monkeypatch.setattr(ssm_runtime, "install_wheel", slow_install_wheel)
    # Force the source-build fallback to no-op after the wheel attempt.
    monkeypatch.setattr(ssm_runtime.shutil, "which", lambda name: None)

    def run_fail(cmd, **k):
        return _Result(returncode = 1)

    thread = threading.Thread(
        target = lambda: ssm_runtime._install_kernel(
            import_name = "causal_conv1d",
            display_name = "causal-conv1d",
            pypi_name = "causal-conv1d",
            package_version = "1.6.1",
            release_tag = "v1.6.1.post4",
            release_base_url = "x",
            status_cb = statuses.append,
            run = run_fail,
        ),
        daemon = True,
    )
    thread.start()
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        if any("Still installing causal-conv1d (prebuilt kernel)" in s for s in statuses):
            break
        time.sleep(0.02)
    released.set()
    thread.join(timeout = 2.0)
    assert any("Still installing causal-conv1d (prebuilt kernel)" in s for s in statuses), statuses
    assert any("Installing causal-conv1d (prebuilt kernel)" in s for s in statuses)


def test_install_kernel_heartbeats_through_the_import_check(monkeypatch):
    # The first torch import can be quiet long enough to trip the inactivity
    # deadline, so keep heartbeats running through it.
    import threading
    import time

    monkeypatch.setattr(ssm_runtime, "_HEARTBEAT_SECONDS", 0.05)
    monkeypatch.setattr(
        ssm_runtime,
        "probe_torch_wheel_env",
        lambda timeout = 30, include_windows = False: {"cuda_version": "12.8"},
    )
    monkeypatch.setattr(
        ssm_runtime,
        "direct_wheel_url",
        lambda **k: "https://example/causal_conv1d-1.6.1.whl",
    )
    monkeypatch.setattr(ssm_runtime, "url_exists", lambda u: True)
    monkeypatch.setattr(
        ssm_runtime, "install_wheel", lambda url, **k: [("uv", _Result(returncode = 0))]
    )

    import_started = threading.Event()
    import_released = threading.Event()
    seen = {"n": 0}

    def slow_importable(name):
        seen["n"] += 1
        if seen["n"] <= 2:  # before install, and again once the lock is held
            return False
        import_started.set()
        assert import_released.wait(1.0)
        return True

    monkeypatch.setattr(ssm_runtime, "_is_importable", slow_importable)

    statuses = []
    thread = threading.Thread(
        target = lambda: ssm_runtime._install_kernel(
            import_name = "causal_conv1d",
            display_name = "causal-conv1d",
            pypi_name = "causal-conv1d",
            package_version = "1.6.1",
            release_tag = "v1.6.1.post4",
            release_base_url = "x",
            status_cb = statuses.append,
            run = lambda *a, **k: _Result(returncode = 1),
        ),
        daemon = True,
    )
    thread.start()
    assert import_started.wait(timeout = 2.0), "must reach the post-wheel import check"
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        if any("Still installing causal-conv1d (prebuilt kernel)" in s for s in statuses):
            break
        time.sleep(0.02)
    import_released.set()
    thread.join(timeout = 2.0)
    assert any("Still installing causal-conv1d (prebuilt kernel)" in s for s in statuses), statuses


def test_heartbeat_does_not_emit_after_the_block(monkeypatch):
    # No in-flight tick may emit after the block exits.
    import time

    monkeypatch.setattr(ssm_runtime, "_HEARTBEAT_SECONDS", 0.05)
    statuses = []
    with ssm_runtime._heartbeat(statuses.append, "still going"):
        time.sleep(0.12)
    n = len(statuses)
    time.sleep(0.15)
    assert len(statuses) == n, statuses


def test_install_kernel_falls_back_to_source(monkeypatch):
    # no wheel -> source build -> importable after install
    states = iter([False, False, True])  # before install, after the lock, after install
    monkeypatch.setattr(ssm_runtime, "_is_importable", lambda name: next(states))
    monkeypatch.setattr(
        ssm_runtime,
        "probe_torch_wheel_env",
        lambda timeout = 30, include_windows = False: {"cuda_version": "12.8"},
    )
    monkeypatch.setattr(ssm_runtime, "direct_wheel_url", lambda **k: None)
    pip_cmds = []
    ok = ssm_runtime._install_kernel(
        import_name = "causal_conv1d",
        display_name = "causal-conv1d",
        pypi_name = "causal-conv1d",
        package_version = "1.6.1",
        release_tag = "v1.6.1.post4",
        release_base_url = "x",
        status_cb = None,
        run = lambda cmd, **k: pip_cmds.append(cmd) or _Result(returncode = 0),
    )
    assert ok is True
    assert any("causal-conv1d==1.6.1" in c for c in pip_cmds[0])


# ── import-cache invalidation (so a just-installed kernel is importable) ───────


def test_is_importable_invalidates_caches(monkeypatch):
    calls = []
    monkeypatch.setattr(ssm_runtime.importlib, "invalidate_caches", lambda: calls.append(1))
    assert ssm_runtime._is_importable("sys") is True
    assert calls  # caches invalidated before attempting the import


@pytest.mark.parametrize(
    "exc",
    [
        ImportError("no module"),
        OSError("undefined symbol: cuLaunchKernel"),
        RuntimeError("CUDA error: ABI mismatch"),
    ],
)
def test_is_importable_treats_broken_kernel_as_not_importable(monkeypatch, exc):
    # ABI-incompatible kernels raise OSError/RuntimeError, not ImportError; all must read as
    # not-importable. _is_importable calls bare __import__(), so patching ssm_runtime.__import__
    # (resolved via module globals) leaves real `import` statements untouched.
    def _raise(name):
        raise exc

    monkeypatch.setattr(ssm_runtime, "__import__", _raise, raising = False)
    monkeypatch.setattr(ssm_runtime.importlib, "invalidate_caches", lambda: None)
    assert ssm_runtime._is_importable("causal_conv1d") is False


def test_causal_conv1d_attempted_on_windows(monkeypatch):
    monkeypatch.setattr(ssm_runtime.sys, "platform", "win32")
    installed = []
    monkeypatch.setattr(
        ssm_runtime,
        "_install_kernel",
        lambda *, import_name, **_: installed.append(import_name) or True,
    )
    ssm_runtime.ensure_ssm_runtime("LiquidAI/LFM2-1.2B")
    assert installed == ["causal_conv1d"]


def test_ssm_model_on_windows_installs_causal_then_mamba_with_triton(monkeypatch):
    monkeypatch.setattr(ssm_runtime.sys, "platform", "win32")
    monkeypatch.setattr(
        ssm_runtime.importlib.util,
        "find_spec",
        lambda name, *a: object() if name == "triton" else None,
    )
    installed = []
    monkeypatch.setattr(
        ssm_runtime,
        "_install_kernel",
        lambda *, import_name, **_: installed.append(import_name) or True,
    )
    ssm_runtime.ensure_ssm_runtime("unsloth/NVIDIA-Nemotron-3-Nano-4B")
    assert installed == ["causal_conv1d", "mamba_ssm"]


def test_ssm_model_on_windows_without_triton_fails_clearly(monkeypatch):
    monkeypatch.setattr(ssm_runtime.sys, "platform", "win32")
    monkeypatch.setattr(ssm_runtime.importlib.util, "find_spec", lambda name, *a: None)
    installed = []
    monkeypatch.setattr(
        ssm_runtime,
        "_install_kernel",
        lambda *, import_name, **_: installed.append(import_name) or True,
    )
    with pytest.raises(RuntimeError, match = "triton-windows"):
        ssm_runtime.ensure_ssm_runtime("unsloth/NVIDIA-Nemotron-3-Nano-4B")
    assert "mamba_ssm" not in installed


def _windows_install_kernel(
    monkeypatch,
    *,
    wheel_url,
    exists,
    outcome_rc = 0,
    importable_after = True,
):
    """Drive _install_kernel as Windows; returns (result, commands run, wheel installs)."""
    monkeypatch.setattr(ssm_runtime.sys, "platform", "win32")
    # shutil.which consults sys.platform, and a Linux host cannot answer as Windows.
    monkeypatch.setattr(ssm_runtime.shutil, "which", lambda *a, **k: None)
    # Before install, once the lock is held, after the wheel.
    states = iter([False, False, importable_after])
    monkeypatch.setattr(ssm_runtime, "_is_importable", lambda name: next(states))
    probe_kwargs = {}
    monkeypatch.setattr(
        ssm_runtime,
        "probe_torch_wheel_env",
        lambda **k: probe_kwargs.update(k) or {"platform_tag": "win_amd64", "cuda_version": "13.0"},
    )
    monkeypatch.setattr(ssm_runtime, "direct_wheel_url", lambda **k: wheel_url)
    monkeypatch.setattr(ssm_runtime, "url_exists", lambda u: exists)
    present = {"value": False}
    monkeypatch.setattr(ssm_runtime, "_installed_distribution", lambda name: present["value"])
    installs = []

    def _install_wheel(url, **k):
        installs.append(url)
        present["value"] = outcome_rc == 0
        return [("uv", _Result(returncode = outcome_rc))]

    monkeypatch.setattr(ssm_runtime, "install_wheel", _install_wheel)
    ran = []

    def _run(cmd, **k):
        ran.append(cmd)
        if "uninstall" in cmd:
            present["value"] = False
        return _Result()

    ok = ssm_runtime._install_kernel(
        import_name = "causal_conv1d",
        display_name = "causal-conv1d",
        pypi_name = "causal-conv1d",
        package_version = "1.6.1",
        release_tag = "v1.6.1.post4",
        release_base_url = "https://github.com/Dao-AILab/causal-conv1d/releases/download",
        status_cb = None,
        run = _run,
    )
    assert probe_kwargs.get("include_windows") is True
    return ok, ran, installs


def test_windows_installs_the_published_wheel(monkeypatch):
    ok, ran, installs = _windows_install_kernel(
        monkeypatch, wheel_url = "https://example/causal_conv1d-win_amd64.whl", exists = True
    )
    assert ok is True
    assert installs == ["https://example/causal_conv1d-win_amd64.whl"]
    assert ran == []


@pytest.mark.parametrize(("wheel_url", "exists"), [(None, False), ("https://x/w.whl", False)])
def test_windows_never_source_builds_without_a_wheel(monkeypatch, wheel_url, exists):
    ok, ran, installs = _windows_install_kernel(monkeypatch, wheel_url = wheel_url, exists = exists)
    assert ok is False
    assert installs == []
    assert ran == []  # no pip / uv source build


def test_windows_rejected_wheel_is_removed_not_source_built(monkeypatch):
    ok, ran, installs = _windows_install_kernel(
        monkeypatch, wheel_url = "https://x/w.whl", exists = True, importable_after = False
    )
    assert ok is False
    assert len(installs) == 1
    assert len(ran) == 1 and "uninstall" in ran[0]
    assert not any("--no-build-isolation" in cmd for cmd in ran)


def test_wheel_installed_but_not_importable_falls_back_to_source(monkeypatch):
    # top and under the lock: not importable; after wheel: still not importable (ABI mismatch)
    # -> source build; after source build: importable.
    states = iter([False, False, False, True])
    monkeypatch.setattr(ssm_runtime, "_is_importable", lambda name: next(states))
    monkeypatch.setattr(
        ssm_runtime,
        "probe_torch_wheel_env",
        lambda timeout = 30, include_windows = False: {"cuda_version": "12.8"},
    )
    monkeypatch.setattr(ssm_runtime, "direct_wheel_url", lambda **k: "https://x/w.whl")
    monkeypatch.setattr(ssm_runtime, "url_exists", lambda u: True)
    monkeypatch.setattr(
        ssm_runtime, "install_wheel", lambda url, **k: [("uv", _Result(returncode = 0))]
    )
    pip_cmds = []
    ok = ssm_runtime._install_kernel(
        import_name = "mamba_ssm",
        display_name = "mamba-ssm",
        pypi_name = "mamba-ssm",
        package_version = "2.3.1",
        release_tag = "v2.3.1",
        release_base_url = "x",
        status_cb = None,
        run = lambda cmd, **k: pip_cmds.append(cmd) or _Result(returncode = 0),
    )
    assert ok is True
    assert pip_cmds, "a non-importable wheel must fall back to a source build"


def test_hip_source_build_requires_hipcc(monkeypatch):
    # ROCm env (hip_version set) with no wheel and no hipcc must fail clearly, not build.
    monkeypatch.setattr(ssm_runtime, "_is_importable", lambda name: False)
    monkeypatch.setattr(
        ssm_runtime,
        "probe_torch_wheel_env",
        lambda timeout = 30, include_windows = False: {"hip_version": "6.2"},
    )
    monkeypatch.setattr(ssm_runtime, "direct_wheel_url", lambda **k: None)
    monkeypatch.setattr(ssm_runtime.shutil, "which", lambda name: None)  # no uv, no hipcc
    ran = []
    ok = ssm_runtime._install_kernel(
        import_name = "causal_conv1d",
        display_name = "causal-conv1d",
        pypi_name = "causal-conv1d",
        package_version = "1.6.1",
        release_tag = "v1.6.1.post4",
        release_base_url = "x",
        status_cb = None,
        run = lambda cmd, **k: ran.append(cmd) or _Result(returncode = 0),
    )
    assert ok is False
    assert ran == []  # bailed before invoking pip


def test_source_build_reinstalls_to_replace_broken_wheel(monkeypatch):
    # Reached only when not importable (possibly a broken wheel at the pinned version);
    # the source build must reinstall so it replaces it instead of no-opping.
    states = iter([False, False, True])
    monkeypatch.setattr(ssm_runtime, "_is_importable", lambda name: next(states))
    monkeypatch.setattr(
        ssm_runtime,
        "probe_torch_wheel_env",
        lambda timeout = 30, include_windows = False: {"cuda_version": "12.8"},
    )
    monkeypatch.setattr(ssm_runtime, "direct_wheel_url", lambda **k: None)
    cmds = []
    ssm_runtime._install_kernel(
        import_name = "causal_conv1d",
        display_name = "causal-conv1d",
        pypi_name = "causal-conv1d",
        package_version = "1.6.1",
        release_tag = "v1.6.1.post4",
        release_base_url = "x",
        status_cb = None,
        run = lambda cmd, **k: cmds.append(cmd) or _Result(returncode = 0),
    )
    assert "--reinstall" in cmds[0] or "--force-reinstall" in cmds[0]


def test_hip_uv_source_build_uses_no_cache(monkeypatch):
    # ROCm uv source build must skip the cache to avoid reusing stale partial HIP builds.
    states = iter([False, False, True])
    monkeypatch.setattr(ssm_runtime, "_is_importable", lambda name: next(states))
    monkeypatch.setattr(
        ssm_runtime,
        "probe_torch_wheel_env",
        lambda timeout = 30, include_windows = False: {"hip_version": "6.2"},
    )
    monkeypatch.setattr(ssm_runtime, "direct_wheel_url", lambda **k: None)
    monkeypatch.setattr(ssm_runtime.shutil, "which", lambda name: "/usr/bin/" + name)  # uv + hipcc
    monkeypatch.setattr(ssm_runtime, "_hipcc_gcc_install_dir", lambda: None)
    cmds = []
    ssm_runtime._install_kernel(
        import_name = "causal_conv1d",
        display_name = "causal-conv1d",
        pypi_name = "causal-conv1d",
        package_version = "1.6.1",
        release_tag = "v1.6.1.post4",
        release_base_url = "x",
        status_cb = None,
        run = lambda cmd, **k: cmds.append(cmd) or _Result(returncode = 0),
    )
    assert cmds[0][0] == "uv"
    assert "--no-cache" in cmds[0] and "--reinstall" in cmds[0]


# ── inference worker wiring ───────────────────────────────────────────────────


def test_inference_worker_calls_ensure_ssm_runtime():
    src = (_BACKEND / "core" / "inference" / "worker.py").read_text(encoding = "utf-8")
    assert "provision_ssm_kernels" in src
    assert "provision_ssm_kernels(" in src


def test_inference_worker_skips_ssm_on_mlx_and_checks_lora_base():
    src = (_BACKEND / "core" / "inference" / "worker.py").read_text(encoding = "utf-8")
    # MLX (Apple Silicon) must not try to build CUDA/ROCm SSM kernels.
    assert 'getattr(backend, "device", None) != "mlx"' in src
    # A LoRA load must also check its base model, not just the adapter id.
    assert "mc.base_model" in src


def test_inference_worker_resolves_remote_lora_base_pre_import():
    # A remote LoRA's base (from the Hub adapter_config.json) must be resolved before the
    # transformers import so its SSM kernels are pre-installed, not too late in _handle_load.
    src = (_BACKEND / "core" / "inference" / "worker.py").read_text(encoding = "utf-8")
    assert "_remote_lora_base" in src


def test_inference_worker_tiers_on_base_and_gates_lora_base_only():
    src = (_BACKEND / "core" / "inference" / "worker.py").read_text(encoding = "utf-8")
    # Tier activation runs on the resolved base, not the raw adapter id (remote-LoRA fix).
    assert "_activate_transformers_version(_base" in src
    # The gate only adds a genuine LoRA base, never a full fine-tune's recorded (unloaded) base.
    assert "_gate_targets" in src and "_lora_base" in src


def test_inference_worker_probes_base_for_ssm_kernels():
    # Both the pre-import path and _handle_load must derive SSM targets from a real model id
    # via ssm_probe_identifier, not the raw adapter id / local checkpoint path.
    src = (_BACKEND / "core" / "inference" / "worker.py").read_text(encoding = "utf-8")
    assert src.count("ssm_probe_identifier(") >= 2


def test_pre_import_gate_is_transformers_free():
    # The pre-import gate must not import transformers: security_load_subdirs pulls
    # model_config -> transformers, which would snapshot SSM backend availability before the
    # kernels install. With load_subdirs=() the malware + consent scans stay transformers-free.
    import sys as _sys
    from unittest.mock import patch
    import utils.security.file_security as fs
    import utils.security.consent as consent

    def _is_gated_module(name: str) -> bool:
        return (
            name == "transformers"
            or name.startswith("transformers.")
            or name == "utils.models.model_config"
        )

    # Snapshot then remove the modules so we can assert the gate does not re-import them.
    # Restore the originals afterwards (finally): popping utils.models.model_config without
    # restoring it makes a later importer get a fresh instance, so tests that patched the
    # first instance (e.g. test_vision_cache) miss and hit the real network path.
    _saved = {m: _sys.modules[m] for m in list(_sys.modules) if _is_gated_module(m)}
    for m in _saved:
        _sys.modules.pop(m, None)

    try:
        with patch.object(fs, "_fetch_security_status", return_value = None):
            fs.evaluate_file_security("nvidia/Nemotron-H-8B", load_subdirs = ())
        with patch.object(
            consent, "_load_remote_code_configs", return_value = [{"model_type": "nemotron_h"}]
        ):
            from utils.security import evaluate_remote_code_consent_for_targets
            evaluate_remote_code_consent_for_targets(
                ["nvidia/Nemotron-H-8B"], trust_remote_code = True
            )

        assert "transformers" not in _sys.modules
        assert "utils.models.model_config" not in _sys.modules
    finally:
        # Drop anything the gate imported, then rebind the original module objects so later
        # tests see the same instances they captured at import time.
        for m in [m for m in list(_sys.modules) if _is_gated_module(m) and m not in _saved]:
            _sys.modules.pop(m, None)
        _sys.modules.update(_saved)


def test_pre_import_gate_skips_subdir_computation():
    # The worker's pre-import preflight must call the gate with compute_subdirs=False so it
    # never imports model_config/transformers before the SSM kernels are installed.
    src = (_BACKEND / "core" / "inference" / "worker.py").read_text(encoding = "utf-8")
    assert "compute_subdirs = False" in src


def _call_linenos(tree, func_name, call_name):
    import ast
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            return [
                c.lineno
                for c in ast.walk(node)
                if isinstance(c, ast.Call)
                and isinstance(c.func, ast.Name)
                and c.func.id == call_name
            ]
    return []


def test_security_gates_run_before_ssm_install():
    # The SSM install is name-based and can source-build native packages, so a malware /
    # blocked-code model must be refused first -- in both the pre-import path and _handle_load.
    import ast
    tree = ast.parse((_BACKEND / "core" / "inference" / "worker.py").read_text(encoding = "utf-8"))
    for fn in ("run_inference_process", "_handle_load"):
        gates = _call_linenos(tree, fn, "_run_security_gates")
        ssm = _call_linenos(tree, fn, "_ensure_ssm_kernels")
        assert gates, f"{fn} must call _run_security_gates"
        assert ssm, f"{fn} must call _ensure_ssm_kernels"
        assert min(gates) < min(ssm), f"{fn} must gate before installing SSM kernels"


# ── training worker shares these tables and pins ──────────────────────────────


def test_constants_shared_with_training_worker():
    try:
        from core.training import worker as tw
    except Exception as exc:  # pragma: no cover - only when training deps absent
        pytest.skip(f"training worker not importable here: {exc}")
    from utils import kernel_install

    assert tw._model_is_ssm is ssm_runtime.model_is_ssm
    assert tw._model_wants_causal_conv1d is ssm_runtime.model_wants_causal_conv1d
    assert tw.CAUSAL_CONV1D is ssm_runtime.CAUSAL_CONV1D is kernel_install.CAUSAL_CONV1D
    assert tw.MAMBA_SSM is ssm_runtime.MAMBA_SSM is kernel_install.MAMBA_SSM
    assert tw._hipcc_gcc_install_dir is ssm_runtime._hipcc_gcc_install_dir

    # detection must agree with the training worker across SSM + non-SSM names
    for name in (
        "unsloth/NVIDIA-Nemotron-3-Nano-4B",
        "nvidia/Nemotron-H-8B",
        "tiiuae/Falcon-H1-0.5B",
        "ibm-granite/granite-4.0-h-micro",
        "Qwen/Qwen3-Next-80B",
        "unsloth/Qwen3.5-2B",
        "LiquidAI/LFM2-1.2B",
        "state-spaces/mamba-2.8b-hf",
        "ai21labs/Jamba-v0.1",
        "Zyphra/Zamba2-7B",
        "ibm/Bamba-9B",
        "unsloth/Llama-3.2-1B-Instruct",
        "unsloth/Qwen2.5-7B",
    ):
        assert ssm_runtime.model_wants_causal_conv1d(name) == tw._model_wants_causal_conv1d(
            name
        ), name


# ── renamed / local checkpoints resolve from the config, not the name ─────────


def _write_config(directory: Path, config: dict) -> Path:
    directory.mkdir(parents = True, exist_ok = True)
    (directory / "config.json").write_text(json.dumps(config), encoding = "utf-8")
    return directory


def test_renamed_local_checkpoint_resolves_causal_conv1d_from_its_config(tmp_path):
    """A local LFM2 checkpoint whose folder has no allowlisted substring must
    still be detected, so the availability hook is installed for it."""
    checkpoint = _write_config(
        tmp_path / "my-model",
        {"model_type": "lfm2", "architectures": ["Lfm2ForCausalLM"]},
    )
    target = str(checkpoint)

    # The name-only predicate cannot see it ...
    assert ssm_runtime.model_wants_causal_conv1d(target) is False
    # ... but the resolved predicate reads config.json off the local directory.
    assert ssm_runtime.resolved_model_wants_causal_conv1d(target, target, None) is True
    # Same for a renamed Hub id pointing at a local snapshot.
    assert (
        ssm_runtime.resolved_model_wants_causal_conv1d("acme/internal-llm-v3", target, None) is True
    )


def test_renamed_local_checkpoint_without_ssm_config_stays_off(tmp_path):
    checkpoint = _write_config(
        tmp_path / "my-model-llama",
        {"model_type": "llama", "architectures": ["LlamaForCausalLM"]},
    )
    target = str(checkpoint)
    assert ssm_runtime.resolved_model_wants_causal_conv1d(target, target, None) is False


def test_nested_text_config_resolves_causal_conv1d(tmp_path):
    checkpoint = _write_config(
        tmp_path / "my-vl-model",
        {
            "model_type": "some_vlm",
            "architectures": ["SomeVlmForConditionalGeneration"],
            "text_config": {"model_type": "lfm2", "architectures": ["Lfm2ForCausalLM"]},
        },
    )
    target = str(checkpoint)
    assert ssm_runtime.model_wants_causal_conv1d(target) is False
    assert ssm_runtime.resolved_model_wants_causal_conv1d(target, target, None) is True


# ── Qwen3.5 / 3.6 / Qwen3-Next get causal-conv1d on demand, on every platform ──


_QWEN_GDN_NAMES = [
    "Qwen/Qwen3-Next-80B-A3B",
    "unsloth/Qwen3.5-2B",
    "unsloth/Qwen3.6-27B",
    "my-org/qwen3_5-finetune",
]


@pytest.mark.parametrize("name", _QWEN_GDN_NAMES)
def test_qwen_gdn_models_install_causal_conv1d_on_demand(monkeypatch, name):
    # zoo's Triton conv has no decode update and a sequence-sized dW scratch: keep the native kernel.
    monkeypatch.delenv("UNSLOTH_DISABLE_TRITON_CAUSAL_CONV1D", raising = False)
    assert ssm_runtime.model_wants_causal_conv1d(name) is True
    assert ssm_runtime.model_is_ssm(name) is False
    installed = []
    monkeypatch.setattr(
        ssm_runtime,
        "_install_kernel",
        lambda *, import_name, **_: installed.append(import_name) or True,
    )
    ssm_runtime.ensure_ssm_runtime(name)
    assert installed == ["causal_conv1d"]


@pytest.mark.parametrize(
    "config",
    [
        {"model_type": "qwen3_next", "architectures": ["Qwen3NextForCausalLM"]},
        {
            "model_type": "qwen3_5",
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "text_config": {"model_type": "qwen3_5_text"},
            "vision_config": {"model_type": "qwen3_5_vision"},
        },
    ],
    ids = ["qwen3_next", "qwen3_5"],
)
def test_qwen_gdn_configs_resolve_to_causal_conv1d(monkeypatch, tmp_path, config):
    monkeypatch.delenv("UNSLOTH_DISABLE_TRITON_CAUSAL_CONV1D", raising = False)
    target = str(_write_config(tmp_path / "renamed", config))
    assert ssm_runtime.model_config_wants_causal_conv1d(config) is True
    assert ssm_runtime.resolved_model_wants_causal_conv1d("acme/x", target, None) is True


def _zoo_covers_gdn(
    monkeypatch,
    *,
    decode = True,
    platform = True,
):
    monkeypatch.delenv("UNSLOTH_DISABLE_TRITON_CAUSAL_CONV1D", raising = False)
    monkeypatch.setattr(ssm_runtime, "_zoo_gdn_decode_kernel_present", lambda: decode)
    monkeypatch.setattr(ssm_runtime, "_zoo_triton_platform_usable", lambda: platform)


@pytest.mark.parametrize("name", _QWEN_GDN_NAMES)
def test_qwen_gdn_models_skip_causal_conv1d_when_zoo_covers_decode(monkeypatch, name):
    _zoo_covers_gdn(monkeypatch)
    assert ssm_runtime.model_wants_causal_conv1d(name) is False
    installed = []
    monkeypatch.setattr(
        ssm_runtime,
        "_install_kernel",
        lambda *, import_name, **_: installed.append(import_name) or True,
    )
    ssm_runtime.ensure_ssm_runtime(name)
    assert installed == []


@pytest.mark.parametrize(
    "config",
    [
        {"model_type": "qwen3_next", "architectures": ["Qwen3NextForCausalLM"]},
        {
            "model_type": "qwen3_5_moe",
            "architectures": ["Qwen3_5MoeForConditionalGeneration"],
            "text_config": {"model_type": "qwen3_5_moe_text"},
            "vision_config": {"model_type": "qwen3_5_moe_vision"},
        },
    ],
    ids = ["qwen3_next", "qwen3_5_moe"],
)
def test_qwen_gdn_configs_skip_causal_conv1d_when_zoo_covers_decode(monkeypatch, tmp_path, config):
    _zoo_covers_gdn(monkeypatch)
    target = str(_write_config(tmp_path / "renamed", config))
    # The transformers modeling source names causal_conv1d; zoo's coverage is checked first.
    assert ssm_runtime.model_config_wants_causal_conv1d(config) is False
    assert ssm_runtime.resolved_model_wants_causal_conv1d("acme/x", target, None) is False


@pytest.mark.parametrize("missing", ["kill_switch", "decode_kernel", "platform"])
def test_qwen_gdn_request_returns_without_full_zoo_coverage(monkeypatch, missing):
    _zoo_covers_gdn(monkeypatch, decode = missing != "decode_kernel", platform = missing != "platform")
    if missing == "kill_switch":
        monkeypatch.setenv("UNSLOTH_DISABLE_TRITON_CAUSAL_CONV1D", "1")
    assert ssm_runtime.model_wants_causal_conv1d("unsloth/Qwen3.5-2B") is True
    assert ssm_runtime.model_config_wants_causal_conv1d({"model_type": "qwen3_5"}) is True


def test_zoo_coverage_never_drops_other_causal_conv1d_models(monkeypatch):
    _zoo_covers_gdn(monkeypatch)
    for name in ("LiquidAI/LFM2-1.2B", "nvidia/Nemotron-H-8B", "state-spaces/mamba-2.8b-hf"):
        assert ssm_runtime.model_wants_causal_conv1d(name) is True, name
    # A config mixing a zoo-covered type with another causal-conv1d user still asks.
    mixed = {"model_type": "qwen3_5", "text_config": {"model_type": "lfm2"}}
    assert ssm_runtime.model_config_wants_causal_conv1d(mixed) is True


def test_the_decode_kernel_probe_reads_zoo_source_without_importing_it(monkeypatch, tmp_path):
    monkeypatch.undo()  # drop the conftest pin: probe the real function
    probe = ssm_runtime._zoo_gdn_decode_kernel_present
    patches = tmp_path / "unsloth_zoo" / "temporary_patches"
    patches.mkdir(parents = True)
    (tmp_path / "unsloth_zoo" / "__init__.py").write_text("raise ImportError('never imported')\n")
    source = patches / "gdn_causal_conv1d.py"
    monkeypatch.syspath_prepend(str(tmp_path))
    # An import-guard stub in sys.modules does not answer for the real package.
    monkeypatch.setitem(sys.modules, "unsloth_zoo", types.ModuleType("unsloth_zoo"))
    for text, expected in (
        ("def triton_causal_conv1d(x): ...\n", False),
        ("def triton_causal_conv1d_update(x, conv_state, weight): ...\n", True),
    ):
        source.write_text(text)
        probe.cache_clear()
        assert probe() is expected
    probe.cache_clear()


@pytest.mark.parametrize(
    "name",
    ["tiiuae/Falcon-H1-0.5B", "nvidia/Nemotron-H-8B", "ibm-granite/granite-4.0-h-micro"],
)
def test_mamba2_hybrids_still_install_both_on_demand(monkeypatch, name):
    order = []
    monkeypatch.setattr(
        ssm_runtime,
        "_install_kernel",
        lambda *, import_name, **_: order.append(import_name) or True,
    )
    ssm_runtime.ensure_ssm_runtime(name)
    assert order == ["causal_conv1d", "mamba_ssm"]


# ── mamba-ssm resolved from the config, not only the name ─────────────────────


@pytest.mark.parametrize(
    "name, model_type",
    [
        ("nvidia/NVIDIA-Nemotron-Nano-9B-v2", "nemotron_h"),
        ("ibm-granite/granite-4.0-tiny-preview", "granitemoehybrid"),
        ("acme/internal-hybrid", "falcon_h1"),
        ("acme/internal-hybrid", "bamba"),
        ("acme/internal-hybrid", "jamba"),
        ("acme/internal-hybrid", "zamba"),
        ("acme/internal-hybrid", "zamba2"),
        ("acme/internal-hybrid", "mamba"),
        ("acme/internal-hybrid", "mamba2"),
        ("acme/internal-hybrid", "falcon_mamba"),
    ],
)
def test_renamed_or_unlisted_ssm_models_resolve_from_their_config(tmp_path, name, model_type):
    target = str(_write_config(tmp_path / "ckpt", {"model_type": model_type}))
    assert ssm_runtime.resolved_model_is_ssm(name, target, None) is True
    # Every mamba-ssm model also wants causal-conv1d.
    assert ssm_runtime.resolved_model_wants_causal_conv1d(name, target, None) is True


def test_a_nested_ssm_text_config_resolves_too(tmp_path):
    config = {"model_type": "some_vlm", "text_config": {"model_type": "nemotron_h"}}
    target = str(_write_config(tmp_path / "vlm", config))
    assert ssm_runtime.resolved_model_is_ssm("acme/x", target, None) is True


def test_a_known_non_ssm_config_overrides_an_ssm_looking_name(tmp_path):
    target = str(_write_config(tmp_path / "llama", {"model_type": "llama"}))
    assert ssm_runtime.resolved_model_is_ssm("user/falcon-h1-style-llama", target, None) is False


def test_no_config_falls_back_to_the_name(tmp_path):
    missing = str(tmp_path / "no-config")
    assert ssm_runtime.resolved_model_is_ssm("tiiuae/Falcon-H1-0.5B", missing, None) is True
    assert ssm_runtime.resolved_model_is_ssm("unsloth/Llama-3.2-1B", missing, None) is False


def test_resolved_answers_override_the_name(monkeypatch):
    order = []
    monkeypatch.setattr(
        ssm_runtime,
        "_install_kernel",
        lambda *, import_name, **_: order.append(import_name) or True,
    )
    ssm_runtime.ensure_ssm_runtime("acme/renamed", wants_causal_conv1d = False, is_ssm = True)
    assert order == ["causal_conv1d", "mamba_ssm"]
    order.clear()
    ssm_runtime.ensure_ssm_runtime("tiiuae/Falcon-H1-0.5B", wants_causal_conv1d = False, is_ssm = False)
    assert order == []


def test_the_mamba_ssm_failure_names_the_offline_remedy(monkeypatch):
    monkeypatch.setattr(ssm_runtime, "_install_kernel", lambda **k: False)
    monkeypatch.setattr(ssm_runtime, "native_modeling_available", lambda model_type: False)
    with pytest.raises(RuntimeError, match = "unsloth install-kernels"):
        ssm_runtime.ensure_ssm_runtime("tiiuae/Falcon-H1-0.5B")


# ── in-process installs reach an already-imported transformers ────────────────


def test_refresh_clears_the_gates_and_cached_misses(monkeypatch):
    import functools

    calls = {"n": 0}

    @functools.lru_cache
    def is_causal_conv1d_available():
        calls["n"] += 1
        return calls["n"] > 1

    import_utils = types.SimpleNamespace(is_causal_conv1d_available = is_causal_conv1d_available)
    module_type = type(sys)
    kernel_module = module_type("causal_conv1d")
    mapping = {"causal-conv1d": None, "mamba-ssm": None, "flash-attn": None, "x": kernel_module}
    hub_kernels = types.SimpleNamespace(_KERNEL_MODULE_MAPPING = mapping)
    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace())
    monkeypatch.setitem(sys.modules, "transformers.utils.import_utils", import_utils)
    monkeypatch.setitem(sys.modules, "transformers.integrations.hub_kernels", hub_kernels)

    assert is_causal_conv1d_available() is False
    ssm_runtime.refresh_transformers_kernel_gates()
    assert is_causal_conv1d_available() is True
    assert mapping == {"flash-attn": None, "x": kernel_module}


def test_refresh_is_a_no_op_without_transformers_and_never_raises(monkeypatch):
    monkeypatch.delitem(sys.modules, "transformers", raising = False)
    ssm_runtime.refresh_transformers_kernel_gates()

    class _Broken:
        def __getattr__(self, name):
            raise RuntimeError("boom")

    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace())
    monkeypatch.setitem(sys.modules, "transformers.utils.import_utils", _Broken())
    monkeypatch.setitem(sys.modules, "transformers.integrations.hub_kernels", _Broken())
    ssm_runtime.refresh_transformers_kernel_gates()


class _Queue(list):
    def put(self, item):
        self.append(item)


def test_inference_worker_resolves_kernels_from_the_base_config(monkeypatch, tmp_path):
    from core.inference import worker as iw

    target = str(_write_config(tmp_path / "renamed", {"model_type": "nemotron_h"}))
    seen = []

    def _ensure(probe, **kwargs):
        seen.append(
            (probe, kwargs["wants_causal_conv1d"], kwargs["is_ssm"], kwargs["mamba_ssm_required"])
        )
        return []

    monkeypatch.setattr(ssm_runtime, "ensure_ssm_runtime", _ensure)
    assert iw._ensure_ssm_kernels([("renamed", target)], resp_queue = _Queue(), hf_token = None)
    assert seen == [("renamed", True, True, True)]

    src = (_BACKEND / "core" / "inference" / "worker.py").read_text(encoding = "utf-8")
    assert "_ensure_ssm_kernels(ssm_targets, resp_queue, hf_token, trust_remote_code)" in src
    assert "_ensure_ssm_kernels(_ssm_targets, resp_queue, _hf_token, _trust_remote_code)" in src


# ── mamba-ssm is fatal only where the model cannot load without it ────────────


@pytest.mark.parametrize(
    "name",
    ["state-spaces/mamba-130m-hf", "ai21labs/Jamba-v0.1", "Zyphra/Zamba2-7B", "ibm/Bamba-9B"],
)
def test_a_missing_mamba_ssm_is_not_fatal_for_models_with_a_torch_path(monkeypatch, name):
    monkeypatch.setattr(ssm_runtime, "_install_kernel", lambda **k: False)
    messages = []
    assert ssm_runtime.model_requires_mamba_ssm(name) is False
    ssm_runtime.ensure_ssm_runtime(name, status_cb = messages.append)  # no raise
    assert any("torch path" in message for message in messages)


@pytest.mark.parametrize(
    "model_type, required",
    [
        ("nemotron_h", True),
        ("falcon_h1", True),
        ("granitemoehybrid", True),
        ("mamba", False),
        ("mamba2", False),
        ("falcon_mamba", False),
        ("jamba", False),
        ("zamba", False),
        ("zamba2", False),
        ("bamba", False),
    ],
)
def test_only_the_old_fatal_set_requires_mamba_ssm(tmp_path, model_type, required):
    target = str(_write_config(tmp_path / "ckpt", {"model_type": model_type}))
    kernels = ssm_runtime.resolve_ssm_kernels("acme/x", target, None)
    assert kernels.mamba_ssm is True and kernels.causal_conv1d is True
    assert kernels.mamba_ssm_required is required


def test_windows_without_triton_is_fatal_only_for_the_required_set(monkeypatch):
    monkeypatch.setattr(ssm_runtime.sys, "platform", "win32")
    monkeypatch.setattr(ssm_runtime.importlib.util, "find_spec", lambda name, *a: None)
    monkeypatch.setattr(ssm_runtime, "_install_kernel", lambda **k: True)
    ssm_runtime.ensure_ssm_runtime("ai21labs/Jamba-v0.1")  # no raise
    with pytest.raises(RuntimeError, match = "Triton"):
        ssm_runtime.ensure_ssm_runtime("tiiuae/Falcon-H1-0.5B")


# ── a local adapter's base decides the kernels ───────────────────────────────


def _local_adapter(tmp_path, base: str) -> str:
    adapter = tmp_path / "my-finetune-v3"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": base, "peft_type": "LORA"}), encoding = "utf-8"
    )
    return str(adapter)


def test_an_unrelatedly_named_local_adapter_resolves_through_its_falcon_h1_base(tmp_path):
    base = str(_write_config(tmp_path / "base", {"model_type": "falcon_h1"}))
    adapter = _local_adapter(tmp_path, base)
    kernels = ssm_runtime.resolve_ssm_kernels(adapter, adapter, None)
    assert (kernels.causal_conv1d, kernels.mamba_ssm, kernels.mamba_ssm_required) == (
        True,
        True,
        True,
    )
    assert ssm_runtime.resolved_model_is_ssm(adapter, adapter, None) is True
    assert ssm_runtime.resolved_model_wants_causal_conv1d(adapter, adapter, None) is True


def test_the_authorized_adapter_base_pin_wins(tmp_path):
    llama = str(_write_config(tmp_path / "llama", {"model_type": "llama"}))
    falcon = str(_write_config(tmp_path / "falcon", {"model_type": "falcon_h1"}))
    adapter = _local_adapter(tmp_path, llama)
    assert ssm_runtime.resolved_model_is_ssm(adapter, adapter, None) is False
    assert ssm_runtime.resolved_model_is_ssm(adapter, adapter, None, adapter_base = falcon) is True


def test_the_training_worker_passes_the_adapter_base_pin():
    src = (_BACKEND / "core" / "training" / "worker.py").read_text(encoding = "utf-8")
    assert 'adapter_base = config.get("adapter_base_pin") or None' in src


# ── the modeling-source scan works without importing transformers ─────────────


def test_the_modeling_source_scan_reads_files_without_importing_transformers(monkeypatch, tmp_path):
    package = tmp_path / "transformers"
    model_dir = package / "models" / "convnet_x"
    model_dir.mkdir(parents = True)
    (package / "__init__.py").write_text("raise RuntimeError('must not be imported')\n")
    (model_dir / "modeling_convnet_x.py").write_text(
        "from causal_conv1d import causal_conv1d_fn\n", encoding = "utf-8"
    )
    plain = package / "models" / "plain_x"
    plain.mkdir()
    (plain / "modeling_plain_x.py").write_text("import torch\n", encoding = "utf-8")
    spec = types.SimpleNamespace(
        submodule_search_locations = [str(package)], origin = str(package / "__init__.py")
    )
    real_find_spec = ssm_runtime.importlib.util.find_spec
    monkeypatch.setattr(
        ssm_runtime.importlib.util,
        "find_spec",
        lambda name, *a: spec if name == "transformers" else real_find_spec(name, *a),
    )
    monkeypatch.setattr(ssm_runtime, "_TRANSFORMERS_CAUSAL_CONV1D_MODEL_TYPE_CACHE", {})
    monkeypatch.delitem(sys.modules, "transformers", raising = False)

    assert ssm_runtime.model_config_wants_causal_conv1d({"model_type": "convnet_x"}) is True
    assert ssm_runtime.model_config_wants_causal_conv1d({"model_type": "plain_x"}) is False
    assert "transformers" not in sys.modules


# ── wheel availability, failures, cleanup, locking, backend gate, floor (#13971 audit) ──────────


def _kernel_kwargs(**overrides):
    kwargs = dict(
        import_name = "causal_conv1d",
        display_name = "causal-conv1d",
        pypi_name = "causal-conv1d",
        package_version = "1.6.1",
        release_tag = "v1.6.1.post4",
        release_base_url = "x",
        status_cb = None,
    )
    kwargs.update(overrides)
    return kwargs


class _Host:
    """A fake environment: what imports, what is installed, what each step does, per kernel."""

    def __init__(
        self,
        monkeypatch,
        *,
        env,
        wheel_url = True,
        head = True,
        wheel_ok = True,
        build_ok = False,
        platform = "linux",
    ):
        self.env = env
        self.installed: set[str] = set()
        self.imports: set[str] = set()
        self.wheel_ok = wheel_ok
        self.build_ok = build_ok
        self.wheels: list[str] = []
        self.commands: list[list[str]] = []
        self.statuses: list[str] = []
        monkeypatch.setattr(ssm_runtime.sys, "platform", platform)
        monkeypatch.setattr(ssm_runtime.shutil, "which", lambda name: None)
        monkeypatch.setattr(ssm_runtime, "probe_torch_wheel_env", lambda **k: env)
        monkeypatch.setattr(
            ssm_runtime,
            "direct_wheel_url",
            lambda **k: f"https://x/{k['filename_prefix']}.whl" if wheel_url else None,
        )
        monkeypatch.setattr(ssm_runtime, "url_exists", lambda url: head)
        monkeypatch.setattr(ssm_runtime, "install_wheel", self.install_wheel)
        monkeypatch.setattr(ssm_runtime, "_is_importable", lambda name: name in self.imports)
        monkeypatch.setattr(
            ssm_runtime,
            "_installed_distribution",
            lambda name: name.replace("-", "_") in self.installed,
        )

    def install_wheel(self, url, **kwargs):
        self.wheels.append(url)
        name = url.rsplit("/", 1)[-1].removesuffix(".whl")
        self.installed.add(name)  # a download leaves the distribution behind either way
        if self.wheel_ok:
            self.imports.add(name)
        return [("uv", _Result(returncode = 0))]

    def run(self, cmd, **kwargs):
        self.commands.append(list(cmd))
        joined = " ".join(cmd)
        name = "mamba_ssm" if "mamba-ssm" in joined else "causal_conv1d"
        if "uninstall" in cmd:
            self.installed.discard(name)
            self.imports.discard(name)
        else:  # a source build
            self.installed.add(name)
            if self.build_ok:
                self.imports.add(name)
        return _Result(returncode = 0 if self.build_ok else 1)

    def install(self, **overrides):
        return ssm_runtime._install_kernel(
            **_kernel_kwargs(status_cb = self.statuses.append, **overrides), run = self.run
        )

    @property
    def builds(self):
        return [cmd for cmd in self.commands if "uninstall" not in cmd]


_CUDA = {"cuda_version": "12.8", "cuda_major": "12", "platform_tag": "linux_x86_64"}
_WINDOWS = {"cuda_version": "13.0", "cuda_major": "13", "platform_tag": "win_amd64"}


@pytest.mark.parametrize("platform, env", [("linux", _CUDA), ("win32", _WINDOWS)])
def test_an_unanswered_head_still_downloads_the_wheel(monkeypatch, platform, env):
    host = _Host(monkeypatch, env = env, head = None, platform = platform)
    assert host.install() is True
    assert host.wheels == ["https://x/causal_conv1d.whl"]
    assert host.builds == []  # the wheel worked: no source build


def test_a_required_hybrid_loads_when_only_head_is_refused(monkeypatch):
    host = _Host(monkeypatch, env = _CUDA, head = None)
    monkeypatch.setattr(ssm_runtime.importlib.util, "find_spec", lambda *a, **k: object())
    installed = ssm_runtime.ensure_ssm_runtime(
        "nvidia/NVIDIA-Nemotron-Nano-9B-v2",
        run = host.run,
        wants_causal_conv1d = True,
        is_ssm = True,
        mamba_ssm_required = True,
    )
    assert installed == ["causal_conv1d", "mamba_ssm"]
    assert len(host.wheels) == 2


def test_a_failed_download_after_an_unanswered_head_falls_back(monkeypatch):
    host = _Host(monkeypatch, env = _CUDA, head = None, wheel_ok = False, build_ok = True)
    assert host.install() is True
    assert len(host.builds) == 1  # the real download failed: then the source build


def test_a_failed_install_is_not_retried_in_this_process(monkeypatch):
    host = _Host(monkeypatch, env = _CUDA, head = False, build_ok = False)
    assert host.install() is False
    assert host.install() is False
    assert len(host.builds) == 1


def test_a_failed_install_is_retried_once_the_toolchain_changes(monkeypatch):
    host = _Host(monkeypatch, env = _CUDA, head = False, build_ok = False)
    assert host.install() is False
    monkeypatch.setenv("CUDA_HOME", "/opt/cuda-new")
    host.install()
    assert len(host.builds) == 2


@pytest.mark.parametrize("env", [_CUDA, {"hip_version": "6.4", "platform_tag": "linux_x86_64"}])
def test_optional_mamba_ssm_is_wheel_only(monkeypatch, env):
    host = _Host(monkeypatch, env = env, head = False, build_ok = True)
    monkeypatch.setattr(
        ssm_runtime.shutil, "which", lambda name: "/usr/bin/hipcc" if name == "hipcc" else None
    )
    host_installs = ssm_runtime.ensure_ssm_runtime(
        "ai21labs/Jamba-v0.1",
        run = host.run,
        wants_causal_conv1d = False,
        is_ssm = True,
        mamba_ssm_required = False,
    )
    # causal-conv1d keeps its source fallback; mamba-ssm for an optional family does not.
    assert not any("mamba-ssm==" in " ".join(cmd) for cmd in host.builds)
    assert any("causal-conv1d==" in " ".join(cmd) for cmd in host.builds)
    assert "mamba_ssm" not in host_installs


def test_a_required_hybrid_keeps_the_mamba_ssm_source_fallback(monkeypatch):
    host = _Host(monkeypatch, env = _CUDA, head = False, build_ok = True)
    ssm_runtime.ensure_ssm_runtime(
        "tiiuae/Falcon-H1-0.5B",
        run = host.run,
        wants_causal_conv1d = True,
        is_ssm = True,
        mamba_ssm_required = True,
    )
    assert any("mamba-ssm==" in " ".join(cmd) for cmd in host.builds)


def test_a_cuda_source_build_has_its_own_looser_deadline(monkeypatch):
    seen = {}

    def run(cmd, **kwargs):
        seen.update(kwargs)
        return _Result(returncode = 1)

    host = _Host(monkeypatch, env = _CUDA, head = False)
    ssm_runtime._install_kernel(**_kernel_kwargs(), run = run)
    from utils import kernel_install

    assert seen.get("timeout") == kernel_install.SOURCE_BUILD_TIMEOUT_CUDA
    assert seen.get("timeout") > kernel_install.SOURCE_BUILD_TIMEOUT  # HIP keeps 1800


@pytest.mark.parametrize("wheel_ok, head", [(False, True), (False, False)])
def test_a_broken_install_is_removed_not_left_behind(monkeypatch, wheel_ok, head):
    host = _Host(monkeypatch, env = _CUDA, head = head, wheel_ok = wheel_ok, build_ok = False)
    assert host.install() is False
    assert host.installed == set()
    assert any("uninstall" in cmd for cmd in host.commands)


def test_the_install_runs_under_the_kernel_lock_and_rechecks(monkeypatch):
    from utils import kernel_install

    host = _Host(monkeypatch, env = _CUDA)
    held = {}

    @contextlib.contextmanager
    def _lock(name, **kwargs):
        held["name"] = name
        host.imports.add(name)  # a peer installed it while this process waited
        yield True

    monkeypatch.setattr(kernel_install, "kernel_install_lock", _lock)
    assert host.install() is True
    assert held["name"] == "causal_conv1d"
    assert host.wheels == [] and host.commands == []


def test_a_lock_held_past_the_bound_installs_nothing(monkeypatch):
    from utils import kernel_install

    host = _Host(monkeypatch, env = _CUDA)

    @contextlib.contextmanager
    def _busy(name, **kwargs):
        yield False

    monkeypatch.setattr(kernel_install, "kernel_install_lock", _busy)
    assert host.install() is False
    assert host.wheels == [] and host.commands == []
    assert any("install lock" in message for message in host.statuses)


@pytest.mark.parametrize(
    "env",
    [
        None,  # the torch probe failed
        {"cuda_version": "", "hip_version": "", "platform_tag": "linux_x86_64"},  # CPU / XPU
    ],
)
def test_no_cuda_or_rocm_torch_installs_nothing(monkeypatch, env):
    host = _Host(monkeypatch, env = env, head = True)
    assert host.install() is False
    assert host.wheels == [] and host.commands == []
    assert any("CUDA or ROCm" in message for message in host.statuses)


def test_a_required_hybrid_on_an_xpu_torch_is_skipped_not_fatal(monkeypatch):
    host = _Host(monkeypatch, env = {"cuda_version": "", "platform_tag": "linux_x86_64"})
    monkeypatch.setattr(ssm_runtime.importlib.util, "find_spec", lambda *a, **k: object())
    assert (
        ssm_runtime.ensure_ssm_runtime(
            "tiiuae/Falcon-H1-0.5B",
            run = host.run,
            wants_causal_conv1d = True,
            is_ssm = True,
            mamba_ssm_required = True,
            # Native transformers modeling: required (worth a build) but not fatal. A fatal one
            # raises instead, see test_a_fatal_mamba_model_on_a_cpu_or_xpu_torch_raises.
            mamba_ssm_fatal = False,
        )
        == []
    )


def test_optional_mamba_ssm_is_skipped_below_the_setup_floor(monkeypatch):
    host = _Host(monkeypatch, env = _CUDA)
    monkeypatch.setattr(
        ssm_runtime,
        "_mamba_ssm_floor_reason",
        lambda: "needs sm80, or sm75 with Triton 3.4+ (torch 2.8+), and Triton is 3.3",
    )
    ssm_runtime.ensure_ssm_runtime(
        "ai21labs/Jamba-v0.1",
        status_cb = host.statuses.append,
        run = host.run,
        wants_causal_conv1d = True,
        is_ssm = True,
        mamba_ssm_required = False,
    )
    assert host.wheels == ["https://x/causal_conv1d.whl"]  # causal-conv1d only
    assert any("Skipping mamba-ssm" in message for message in host.statuses)


def test_a_required_hybrid_ignores_the_floor_as_before(monkeypatch):
    host = _Host(monkeypatch, env = _CUDA)
    monkeypatch.setattr(
        ssm_runtime, "_mamba_ssm_floor_reason", lambda: pytest.fail("no floor for these")
    )
    monkeypatch.setattr(ssm_runtime.importlib.util, "find_spec", lambda *a, **k: object())
    ssm_runtime.ensure_ssm_runtime(
        "tiiuae/Falcon-H1-0.5B",
        run = host.run,
        wants_causal_conv1d = True,
        is_ssm = True,
        mamba_ssm_required = True,
    )
    assert len(host.wheels) == 2


@pytest.mark.parametrize(
    "capability, triton, skipped",
    [((7, 5), (3, 3), True), ((7, 5), (3, 4), False), ((8, 0), None, False), (None, None, True)],
)
def test_the_runtime_floor_is_setups(monkeypatch, capability, triton, skipped):
    from utils import kernel_install

    monkeypatch.delitem(sys.modules, "torch", raising = False)
    monkeypatch.delenv("UNSLOTH_MAMBA_PRE_AMPERE_FAST", raising = False)
    monkeypatch.setattr(kernel_install, "_gpu_capability", lambda run: capability)
    monkeypatch.setattr(kernel_install, "_triton_version", lambda run: triton)
    assert (_REAL_MAMBA_SSM_FLOOR_REASON() is not None) is skipped


def test_provisioning_reads_the_config_once_per_process(monkeypatch):
    reads = []
    monkeypatch.setattr(
        "utils.transformers_version._load_config_json",
        lambda target, token: reads.append(target) or {"model_type": "llama"},
    )
    monkeypatch.setattr(ssm_runtime, "ensure_ssm_runtime", lambda *a, **k: [])
    for _ in range(3):
        ssm_runtime.provision_ssm_kernels("acme/llama", "acme/llama", None)
    assert reads == ["acme/llama"]


def test_a_caller_with_the_config_skips_the_read(monkeypatch):
    monkeypatch.setattr(
        "utils.transformers_version._load_config_json",
        lambda *a, **k: pytest.fail("the caller already has the config"),
    )
    kernels = ssm_runtime.resolve_ssm_kernels(
        "acme/hybrid", "acme/hybrid", None, model_config = {"model_type": "nemotron_h"}
    )
    assert kernels.mamba_ssm_required


def test_the_preflight_and_handle_load_share_one_failed_attempt(monkeypatch):
    # The inference worker provisions before its transformers import and again in _handle_load:
    # a failed optional build must not run twice (two 30-minute HIP builds before).
    host = _Host(
        monkeypatch, env = {"hip_version": "6.4", "platform_tag": "linux_x86_64"}, head = False
    )
    monkeypatch.setattr(ssm_runtime.shutil, "which", lambda name: "/usr/bin/" + name)
    monkeypatch.setattr(ssm_runtime, "_hipcc_gcc_install_dir", lambda: None)
    monkeypatch.setattr(ssm_runtime.importlib.util, "find_spec", lambda *a, **k: object())
    for _stage in ("preflight", "_handle_load"):
        ssm_runtime.ensure_ssm_runtime(
            "ai21labs/Jamba-v0.1",
            run = host.run,
            wants_causal_conv1d = True,
            is_ssm = True,
            mamba_ssm_required = False,
        )
    assert len(host.builds) == 1  # one causal-conv1d build; mamba-ssm is wheel-only here


@pytest.mark.parametrize("torch_mm", ["2.13", "2.14"])
def test_no_pinned_source_build_on_torch_213_and_later(monkeypatch, torch_mm):
    # causal-conv1d 1.6.1 / mamba-ssm 2.3.1 build at C++17; torch 2.13+ headers need C++20, so the
    # build is a certain multi-minute failure: only the prebuilt wheel serves these.
    host = _Host(monkeypatch, env = {**_CUDA, "torch_mm": torch_mm}, wheel_url = False)
    assert host.install() is False
    assert host.builds == []
    assert any("cannot be built from source" in message for message in host.statuses)


def test_torch_212_keeps_the_source_fallback(monkeypatch):
    host = _Host(monkeypatch, env = {**_CUDA, "torch_mm": "2.12"}, head = False, build_ok = True)
    assert host.install() is True
    assert len(host.builds) == 1


# ── when a missing mamba-ssm refuses the load (AMD hardware report, PR #13971) ──────────────────

_NVIDIA_REMOTE_NEMOTRON = """
from transformers.utils.import_utils import is_mamba_2_ssm_available

if is_mamba_2_ssm_available():
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
else:
    mamba_chunk_scan_combined = None

try:
    from mamba_ssm.ops.triton.layernorm_gated import rmsnorm_fn
except ImportError:
    raise ImportError("mamba-ssm is required by the Mamba model but cannot be imported")
"""

_GATED_REMOTE = """
try:
    from mamba_ssm.ops.triton.layernorm_gated import rmsnorm_fn
except ImportError:
    rmsnorm_fn = None
if is_mamba_2_ssm_available():
    import mamba_ssm
"""


def _checkpoint(
    tmp_path,
    model_type,
    auto_map = None,
    modeling = None,
    name = "ckpt",
):
    folder = tmp_path / name
    folder.mkdir()
    config = {"model_type": model_type}
    if auto_map:
        config["auto_map"] = auto_map
    (folder / "config.json").write_text(json.dumps(config))
    for filename, text in (modeling or {}).items():
        (folder / filename).write_text(text)
    return str(folder)


def _failing_mamba(monkeypatch):
    attempts = []

    def _install(**kwargs):
        attempts.append((kwargs["import_name"], kwargs.get("allow_source_build")))
        return False

    monkeypatch.setattr(ssm_runtime, "_install_kernel", _install)
    monkeypatch.setattr(ssm_runtime.importlib.util, "find_spec", lambda *a, **k: object())
    return attempts


@pytest.mark.parametrize("model_type", ["nemotron_h", "falcon_h1", "granitemoehybrid"])
def test_a_native_required_hybrid_loads_with_a_warning(monkeypatch, tmp_path, model_type):
    # The native transformers model runs without mamba-ssm (it trains on Windows ROCm with none):
    # still worth a source build, but a failure is a warning, not a refused load.
    attempts = _failing_mamba(monkeypatch)
    monkeypatch.setattr(ssm_runtime, "native_modeling_available", lambda t: True)
    statuses = []
    kernels, installed = ssm_runtime.provision_ssm_kernels(
        "nvidia/NVIDIA-Nemotron-Nano-9B-v2",
        _checkpoint(tmp_path, model_type),
        None,
        status_cb = statuses.append,
    )
    assert kernels.mamba_ssm_required and not kernels.mamba_ssm_fatal
    assert ("mamba_ssm", True) in attempts  # the source fallback is kept for these three
    assert installed == []
    assert any("Warning: could not install mamba-ssm" in message for message in statuses)


def test_remote_code_that_imports_mamba_ssm_still_refuses(monkeypatch, tmp_path):
    _failing_mamba(monkeypatch)
    monkeypatch.setattr(ssm_runtime, "native_modeling_available", lambda t: True)
    target = _checkpoint(
        tmp_path,
        "nemotron_h",
        auto_map = {"AutoModelForCausalLM": "modeling_nemotron_h.NemotronHForCausalLM"},
        modeling = {"modeling_nemotron_h.py": _NVIDIA_REMOTE_NEMOTRON},
    )
    with pytest.raises(RuntimeError, match = "Could not install mamba-ssm"):
        ssm_runtime.provision_ssm_kernels(target, target, None, trust_remote_code = True)


def test_the_same_checkpoint_without_trust_loads_natively(monkeypatch, tmp_path):
    # auto_map is only followed under trust_remote_code (or when transformers lacks the model).
    _failing_mamba(monkeypatch)
    monkeypatch.setattr(ssm_runtime, "native_modeling_available", lambda t: True)
    target = _checkpoint(
        tmp_path,
        "nemotron_h",
        auto_map = {"AutoModelForCausalLM": "modeling_nemotron_h.NemotronHForCausalLM"},
        modeling = {"modeling_nemotron_h.py": _NVIDIA_REMOTE_NEMOTRON},
    )
    kernels, _ = ssm_runtime.provision_ssm_kernels(target, target, None, trust_remote_code = False)
    assert not kernels.mamba_ssm_fatal


def test_remote_code_with_a_guarded_import_loads(monkeypatch, tmp_path):
    _failing_mamba(monkeypatch)
    monkeypatch.setattr(ssm_runtime, "native_modeling_available", lambda t: True)
    target = _checkpoint(
        tmp_path,
        "nemotron_h",
        auto_map = {"AutoModelForCausalLM": "model.CustomForCausalLM"},
        modeling = {"model.py": _GATED_REMOTE},
    )
    kernels, _ = ssm_runtime.provision_ssm_kernels(target, target, None, trust_remote_code = True)
    assert not kernels.mamba_ssm_fatal


def test_unreadable_remote_code_is_treated_as_requiring_it(monkeypatch, tmp_path):
    _failing_mamba(monkeypatch)
    monkeypatch.setattr(ssm_runtime, "native_modeling_available", lambda t: True)
    target = _checkpoint(
        tmp_path,
        "falcon_h1",
        auto_map = {"AutoModelForCausalLM": "modeling_falcon_h1.FalconH1ForCausalLM"},
    )
    with pytest.raises(RuntimeError):
        ssm_runtime.provision_ssm_kernels(target, target, None, trust_remote_code = True)


@pytest.mark.parametrize("model_type", ["nemotron_h", "falcon_h1"])
def test_a_transformers_without_the_native_model_still_refuses(monkeypatch, tmp_path, model_type):
    _failing_mamba(monkeypatch)
    monkeypatch.setattr(ssm_runtime, "native_modeling_available", lambda t: False)
    target = _checkpoint(tmp_path, model_type)
    with pytest.raises(RuntimeError, match = "Could not install mamba-ssm"):
        ssm_runtime.provision_ssm_kernels(target, target, None)


def test_windows_without_triton_refuses_only_a_fatal_load(monkeypatch, tmp_path):
    monkeypatch.setattr(ssm_runtime.sys, "platform", "win32")
    monkeypatch.setattr(ssm_runtime, "_install_kernel", lambda **k: False)
    monkeypatch.setattr(
        ssm_runtime.importlib.util,
        "find_spec",
        lambda name, *a, **k: None if name == "triton" else object(),
    )
    statuses = []
    monkeypatch.setattr(ssm_runtime, "native_modeling_available", lambda t: True)
    ssm_runtime.provision_ssm_kernels(
        "x", _checkpoint(tmp_path, "nemotron_h", name = "a"), None, status_cb = statuses.append
    )
    assert any("triton-windows" in message for message in statuses)
    monkeypatch.setattr(ssm_runtime, "native_modeling_available", lambda t: False)
    with pytest.raises(RuntimeError, match = "triton-windows"):
        ssm_runtime.provision_ssm_kernels("x", _checkpoint(tmp_path, "nemotron_h", name = "b"), None)


def test_native_modeling_is_read_from_the_installed_transformers(monkeypatch, tmp_path):
    models = tmp_path / "transformers" / "models"
    (models / "falcon_h1").mkdir(parents = True)
    (models / "falcon_h1" / "modeling_falcon_h1.py").write_text("")
    (models / "nemotron_h").mkdir()
    (models / "nemotron_h" / "configuration_nemotron_h.py").write_text("")
    monkeypatch.setattr(ssm_runtime, "_transformers_models_dir", lambda: models)
    monkeypatch.setattr(ssm_runtime, "_NATIVE_MODELING_CACHE", {})
    assert ssm_runtime.native_modeling_available("falcon_h1") is True
    assert ssm_runtime.native_modeling_available("nemotron_h") is False  # config only
    assert ssm_runtime.native_modeling_available("granitemoehybrid") is False


@pytest.mark.parametrize(
    "source, fatal",
    [
        ("import mamba_ssm\n", True),
        ("from mamba_ssm.ops import x\n", True),
        (_NVIDIA_REMOTE_NEMOTRON, True),
        (_GATED_REMOTE, False),
        ("def f():\n    import mamba_ssm\n", False),
        ("import torch\n", False),
        ("def broken(:\n", True),
    ],
)
def test_the_unconditional_import_scan(source, fatal):
    assert ssm_runtime._imports_mamba_ssm_unconditionally(source) is fatal


# ── final-audit items (PR #13971) ────────────────────────────────────────────────────────────────


def test_a_native_granite_4_tiny_preview_loads_with_a_warning(monkeypatch, tmp_path):
    # ibm-granite/granite-4.0-tiny-preview: model_type granitemoehybrid, native in transformers.
    _failing_mamba(monkeypatch)
    monkeypatch.setattr(ssm_runtime, "native_modeling_available", lambda t: t == "granitemoehybrid")
    statuses = []
    target = _checkpoint(tmp_path, "granitemoehybrid", name = "granite-4.0-tiny-preview")
    kernels, _ = ssm_runtime.provision_ssm_kernels(
        "ibm-granite/granite-4.0-tiny-preview", target, None, status_cb = statuses.append
    )
    assert kernels.mamba_ssm_required and not kernels.mamba_ssm_fatal
    assert any("Warning: could not install mamba-ssm" in message for message in statuses)


def test_a_transient_config_miss_is_read_again(monkeypatch):
    reads = iter([None, {"model_type": "qwen3_5"}])
    calls = []

    def _load(target, token):
        calls.append(target)
        return next(reads)

    monkeypatch.setattr("utils.transformers_version._load_config_json", _load)
    assert ssm_runtime.resolve_ssm_kernels("acme/m", "acme/m", None).causal_conv1d is False
    assert ssm_runtime.resolve_ssm_kernels("acme/m", "acme/m", None).causal_conv1d is True
    assert ssm_runtime.resolve_ssm_kernels("acme/m", "acme/m", None).causal_conv1d is True
    assert calls == ["acme/m", "acme/m"]  # the successful read is the one remembered


def test_a_kernel_a_peer_installed_counts_as_new_to_this_worker(monkeypatch):
    from utils import kernel_install

    host = _Host(monkeypatch, env = _CUDA)

    @contextlib.contextmanager
    def _lock(name, **kwargs):
        host.imports.add(name)  # a peer installed it while this process waited
        yield True

    monkeypatch.setattr(kernel_install, "kernel_install_lock", _lock)
    newly = []
    assert ssm_runtime._install_kernel(**_kernel_kwargs(), run = host.run, newly_installed = newly)
    assert newly == ["causal_conv1d"]  # so inference refreshes or restarts its modeling modules


@pytest.mark.parametrize("required, budget", [(False, 120), (True, None)])
def test_optional_accelerations_wait_briefly_for_a_peer(monkeypatch, required, budget):
    from utils import kernel_install

    budget = kernel_install.KERNEL_LOCK_WAIT if budget is None else budget

    seen = {}

    @contextlib.contextmanager
    def _lock(name, **kwargs):
        seen.update(kwargs)
        yield False

    host = _Host(monkeypatch, env = _CUDA)
    monkeypatch.setattr(kernel_install, "kernel_install_lock", _lock)
    host.install(required = required)
    assert seen["wait_seconds"] == budget


def test_ensure_waits_long_only_for_the_required_families(monkeypatch):
    budgets = {}
    monkeypatch.setattr(
        ssm_runtime,
        "_install_kernel",
        lambda **k: budgets.setdefault((k["import_name"], k["required"]), True),
    )
    monkeypatch.setattr(ssm_runtime.importlib.util, "find_spec", lambda *a, **k: object())
    ssm_runtime.ensure_ssm_runtime(
        "q", wants_causal_conv1d = True, is_ssm = False, mamba_ssm_required = False
    )
    ssm_runtime.ensure_ssm_runtime(
        "f", wants_causal_conv1d = True, is_ssm = True, mamba_ssm_required = True
    )
    assert set(budgets) == {("causal_conv1d", False), ("causal_conv1d", True), ("mamba_ssm", True)}


def test_an_installer_that_cannot_start_still_cleans_up_and_is_remembered(monkeypatch):
    host = _Host(monkeypatch, env = _CUDA, head = True)
    host.installed.add("causal_conv1d")  # a broken copy left by an earlier attempt

    def _cannot_start(url, **kwargs):
        raise OSError("uv: No such file or directory")

    monkeypatch.setattr(ssm_runtime, "install_wheel", _cannot_start)
    assert host.install() is False
    assert any("uninstall" in cmd for cmd in host.commands)  # cleanup ran
    assert host.install() is False
    assert len([cmd for cmd in host.commands if "uninstall" in cmd]) == 1  # not retried


# ── A fatal mamba-ssm model on a torch with no kernel backend fails clearly ───


@pytest.mark.parametrize("env", [{"cuda_version": "", "platform_tag": "linux_x86_64"}, {}])
def test_a_fatal_mamba_model_on_a_cpu_or_xpu_torch_raises(monkeypatch, env):
    # Its remote code imports mamba_ssm, so the torch path cannot load it: refuse with the reason.
    monkeypatch.setattr(ssm_runtime, "probe_torch_wheel_env", lambda **k: dict(env))
    monkeypatch.setattr(ssm_runtime, "_install_kernel", lambda **k: False)
    monkeypatch.setattr(ssm_runtime.importlib.util, "find_spec", lambda *a, **k: object())
    messages = []
    with pytest.raises(RuntimeError) as raised:
        ssm_runtime.ensure_ssm_runtime(
            "acme/remote-hybrid",
            status_cb = messages.append,
            wants_causal_conv1d = True,
            is_ssm = True,
            mamba_ssm_required = True,
            mamba_ssm_fatal = True,
        )
    assert "requires mamba-ssm, which needs a CUDA or ROCm torch build" in str(raised.value)
    assert "offline" not in str(raised.value)
    assert not any("torch path" in message for message in messages)


def test_a_fatal_mamba_model_on_a_cuda_torch_keeps_the_offline_hint(monkeypatch):
    monkeypatch.setattr(ssm_runtime, "_install_kernel", lambda **k: False)
    monkeypatch.setattr(ssm_runtime.importlib.util, "find_spec", lambda *a, **k: object())
    with pytest.raises(RuntimeError) as raised:
        ssm_runtime.ensure_ssm_runtime(
            "acme/remote-hybrid",
            wants_causal_conv1d = True,
            is_ssm = True,
            mamba_ssm_required = True,
            mamba_ssm_fatal = True,
        )
    assert "Could not install mamba-ssm" in str(raised.value)
    assert "unsloth install-kernels" in str(raised.value)


def test_a_non_fatal_required_model_on_a_cpu_torch_still_warns_and_loads(monkeypatch):
    monkeypatch.setattr(
        ssm_runtime, "probe_torch_wheel_env", lambda **k: {"cuda_version": "", "platform_tag": "x"}
    )
    monkeypatch.setattr(ssm_runtime, "_install_kernel", lambda **k: False)
    monkeypatch.setattr(ssm_runtime.importlib.util, "find_spec", lambda *a, **k: object())
    messages = []
    ssm_runtime.ensure_ssm_runtime(
        "acme/native-hybrid",
        status_cb = messages.append,
        wants_causal_conv1d = True,
        is_ssm = True,
        mamba_ssm_required = True,
        mamba_ssm_fatal = False,
    )  # no raise
    assert any("torch path" in message for message in messages)
