# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for utils.ssm_runtime: the inference-side auto-install of SSM/Mamba kernels.

Covers detection, wheel-first install, idempotency, the failure path, the inference
worker wiring, and that the training worker shares the same tables and pinned kernels.
"""

import json
import sys
import types
from pathlib import Path
from unittest import mock

import pytest

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from utils import ssm_runtime  # noqa: E402


@pytest.fixture(autouse = True)
def _clear_offline_environment(monkeypatch):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising = False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising = False)


@pytest.fixture(autouse = True)
def _cuda_host_with_triton(monkeypatch):
    # A CUDA torch with Triton, where zoo's Triton conv runs; the platform tests below override it.
    monkeypatch.delenv("UNSLOTH_DISABLE_TRITON_CAUSAL_CONV1D", raising = False)
    monkeypatch.setattr(ssm_runtime, "_zoo_triton_platform_usable", lambda: True)


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
    # A true SSM model whose mamba-ssm cannot install is fatal (cryptic mid-load import
    # otherwise). "Nemotron-3-Nano-30B-A3B" matches the SSM substrings.
    monkeypatch.setattr(ssm_runtime, "_install_kernel", lambda **k: False)
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


@pytest.mark.parametrize("offline_variable", ["HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"])
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
    # not importable before install, importable after the wheel lands
    states = iter([False, True])
    monkeypatch.setattr(ssm_runtime, "_is_importable", lambda name: next(states))
    monkeypatch.setattr(
        ssm_runtime, "probe_torch_wheel_env", lambda timeout = 30, include_windows = False: {"x": "y"}
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
        ssm_runtime, "probe_torch_wheel_env", lambda timeout = 30, include_windows = False: {}
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
        ssm_runtime, "probe_torch_wheel_env", lambda timeout = 30, include_windows = False: {}
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
        if seen["n"] == 1:
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
    states = iter([False, True])  # before install, after install
    monkeypatch.setattr(ssm_runtime, "_is_importable", lambda name: next(states))
    monkeypatch.setattr(
        ssm_runtime, "probe_torch_wheel_env", lambda timeout = 30, include_windows = False: {}
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
    states = iter([False, importable_after])
    monkeypatch.setattr(ssm_runtime, "_is_importable", lambda name: next(states))
    probe_kwargs = {}
    monkeypatch.setattr(
        ssm_runtime,
        "probe_torch_wheel_env",
        lambda **k: probe_kwargs.update(k) or {"platform_tag": "win_amd64"},
    )
    monkeypatch.setattr(ssm_runtime, "direct_wheel_url", lambda **k: wheel_url)
    monkeypatch.setattr(ssm_runtime, "url_exists", lambda u: exists)
    installs = []
    monkeypatch.setattr(
        ssm_runtime,
        "install_wheel",
        lambda url, **k: installs.append(url) or [("uv", _Result(returncode = outcome_rc))],
    )
    ran = []
    ok = ssm_runtime._install_kernel(
        import_name = "causal_conv1d",
        display_name = "causal-conv1d",
        pypi_name = "causal-conv1d",
        package_version = "1.6.1",
        release_tag = "v1.6.1.post4",
        release_base_url = "https://github.com/Dao-AILab/causal-conv1d/releases/download",
        status_cb = None,
        run = lambda cmd, **k: ran.append(cmd) or _Result(),
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


@pytest.mark.parametrize(
    ("wheel_url", "exists"), [(None, False), ("https://x/w.whl", False), ("https://x/w.whl", None)]
)
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
    # top: not importable; after wheel: still not importable (ABI mismatch) -> source build;
    # after source build: importable.
    states = iter([False, False, True])
    monkeypatch.setattr(ssm_runtime, "_is_importable", lambda name: next(states))
    monkeypatch.setattr(
        ssm_runtime, "probe_torch_wheel_env", lambda timeout = 30, include_windows = False: {}
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
    states = iter([False, True])
    monkeypatch.setattr(ssm_runtime, "_is_importable", lambda name: next(states))
    monkeypatch.setattr(
        ssm_runtime, "probe_torch_wheel_env", lambda timeout = 30, include_windows = False: {}
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
    states = iter([False, True])
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
    assert "from utils.ssm_runtime import (\n            ensure_ssm_runtime," in src
    assert "ensure_ssm_runtime(" in src


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


# ── Qwen3.5 / 3.6 / Qwen3-Next run on unsloth-zoo's Triton conv, never causal-conv1d ──


_ZOO_GDN_NAMES = [
    "Qwen/Qwen3-Next-80B-A3B",
    "Qwen/Qwen3-Next-80B-A3B-Instruct",
    "unsloth/Qwen3.5-2B",
    "unsloth/Qwen3.5-35B-A3B-GGUF",
    "unsloth/Qwen3.6-27B",
    "my-org/qwen3_5-finetune",
]


@pytest.mark.parametrize("name", _ZOO_GDN_NAMES)
def test_zoo_gdn_models_request_no_kernels(monkeypatch, name):
    monkeypatch.delenv(ssm_runtime.ZOO_TRITON_CAUSAL_CONV1D_KILL_SWITCH, raising = False)
    assert ssm_runtime.model_wants_causal_conv1d(name) is False
    assert ssm_runtime.model_is_ssm(name) is False
    installed = []
    monkeypatch.setattr(
        ssm_runtime,
        "_install_kernel",
        lambda *, import_name, **_: installed.append(import_name) or True,
    )
    ssm_runtime.ensure_ssm_runtime(name)
    assert installed == []


@pytest.mark.parametrize("name", _ZOO_GDN_NAMES)
def test_zoo_kill_switch_brings_the_causal_conv1d_request_back(monkeypatch, name):
    monkeypatch.setenv(ssm_runtime.ZOO_TRITON_CAUSAL_CONV1D_KILL_SWITCH, "1")
    assert ssm_runtime.model_wants_causal_conv1d(name) is True
    assert ssm_runtime.model_is_ssm(name) is False


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
        {
            "model_type": "qwen3_5_moe",
            "architectures": ["Qwen3_5MoeForConditionalGeneration"],
            "text_config": {"model_type": "qwen3_5_moe_text"},
            "vision_config": {"model_type": "qwen3_5_moe_vision"},
        },
    ],
    ids = ["qwen3_next", "qwen3_5", "qwen3_5_moe"],
)
def test_zoo_gdn_configs_resolve_to_no_causal_conv1d(monkeypatch, tmp_path, config):
    target = str(_write_config(tmp_path / "renamed", config))
    monkeypatch.delenv(ssm_runtime.ZOO_TRITON_CAUSAL_CONV1D_KILL_SWITCH, raising = False)
    assert ssm_runtime.model_config_wants_causal_conv1d(config) is False
    assert ssm_runtime.resolved_model_wants_causal_conv1d("acme/x", target, None) is False
    monkeypatch.setenv(ssm_runtime.ZOO_TRITON_CAUSAL_CONV1D_KILL_SWITCH, "1")
    assert ssm_runtime.model_config_wants_causal_conv1d(config) is True


def test_a_config_mixing_a_zoo_gdn_type_with_another_hybrid_still_requests_it():
    config = {"model_type": "qwen3_5", "text_config": {"model_type": "lfm2"}}
    assert ssm_runtime.model_config_wants_causal_conv1d(config) is True


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


def test_zoo_gdn_model_types_match_unsloth_zoo():
    """Studio's list must equal the models unsloth-zoo's Triton conv patches."""
    import ast
    import importlib.util

    spec = importlib.util.find_spec("unsloth_zoo")
    if spec is None or spec.origin is None:
        pytest.skip("unsloth_zoo not installed")
    source_path = Path(spec.origin).parent / "temporary_patches" / "gdn_causal_conv1d.py"
    if not source_path.is_file():
        pytest.skip("this unsloth_zoo has no Triton causal conv")
    tree = ast.parse(source_path.read_text(encoding = "utf-8"))
    values = [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "_GDN_MODELING" for t in node.targets)
    ]
    assert len(values) == 1
    assert set(ast.literal_eval(values[0])) == ssm_runtime.ZOO_GDN_MODEL_TYPES


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
    assert (
        ssm_runtime.resolved_model_wants_causal_conv1d(
            name, target, None, scan_modeling_source = False
        )
        is True
    )


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
    with pytest.raises(RuntimeError, match = "unsloth install-kernels"):
        ssm_runtime.ensure_ssm_runtime("tiiuae/Falcon-H1-0.5B")


# ── zoo's Triton conv only counts where it can run ────────────────────────────


# The real (cached) platform check, captured before the autouse fixture replaces it.
_REAL_PLATFORM_USABLE = ssm_runtime._zoo_triton_platform_usable.__wrapped__


@pytest.mark.parametrize(
    "platform, triton, torch_version, usable",
    [
        ("linux", True, "2.13.0", True),
        ("linux", True, "2.13.0+cu130", True),
        ("win32", True, "2.13.0+cu130", True),
        ("linux", True, "2.11.0+rocm7.1", False),
        ("win32", False, "2.13.0+cu130", False),
        ("darwin", True, "2.13.0", False),
        ("linux", True, None, False),
        ("linux", True, "2.13.0+cpu", False),
        ("win32", True, "2.13.0", False),
    ],
    ids = [
        "linux-pypi-cuda",
        "linux-cu130",
        "windows-triton",
        "rocm",
        "windows-no-triton",
        "macos",
        "no-torch",
        "cpu-torch",
        "windows-pypi-cpu",
    ],
)
def test_zoo_triton_conv_platform_check(monkeypatch, platform, triton, torch_version, usable):
    monkeypatch.setattr(ssm_runtime.sys, "platform", platform)
    monkeypatch.delitem(ssm_runtime.sys.modules, "torch", raising = False)
    real_find_spec = ssm_runtime.importlib.util.find_spec
    monkeypatch.setattr(
        ssm_runtime.importlib.util,
        "find_spec",
        lambda name, *a: (
            (object() if triton else None) if name == "triton" else real_find_spec(name, *a)
        ),
    )

    def _version(name):
        if torch_version is None:
            raise ssm_runtime.importlib.metadata.PackageNotFoundError(name)
        return torch_version

    monkeypatch.setattr(ssm_runtime.importlib.metadata, "version", _version)
    assert _REAL_PLATFORM_USABLE() is usable


def test_zoo_triton_conv_platform_check_reads_an_imported_torch(monkeypatch):
    import types

    monkeypatch.setattr(ssm_runtime.sys, "platform", "linux")
    monkeypatch.setattr(ssm_runtime.importlib.util, "find_spec", lambda name, *a: object())
    fake = types.SimpleNamespace(version = types.SimpleNamespace(cuda = None, hip = "7.1"))
    monkeypatch.setitem(ssm_runtime.sys.modules, "torch", fake)
    assert _REAL_PLATFORM_USABLE() is False
    fake.version = types.SimpleNamespace(cuda = "13.0", hip = None)
    assert _REAL_PLATFORM_USABLE() is True


@pytest.mark.parametrize("name", ["unsloth/Qwen3.5-2B", "Qwen/Qwen3-Next-80B-A3B"])
def test_qwen_keeps_causal_conv1d_where_zoo_triton_conv_cannot_run(monkeypatch, tmp_path, name):
    monkeypatch.setattr(ssm_runtime, "_zoo_triton_platform_usable", lambda: False)
    assert ssm_runtime.model_wants_causal_conv1d(name) is True
    config = {"model_type": "qwen3_5", "text_config": {"model_type": "qwen3_5_text"}}
    assert ssm_runtime.model_config_wants_causal_conv1d(config) is True


def test_the_platform_check_is_not_run_for_unrelated_models(monkeypatch):
    def _boom():
        raise AssertionError("the platform check must not run for a non-Qwen model")

    monkeypatch.setattr(ssm_runtime, "_zoo_triton_platform_usable", _boom)
    assert ssm_runtime.model_wants_causal_conv1d("unsloth/Llama-3.2-1B") is False
    assert ssm_runtime.model_wants_causal_conv1d("LiquidAI/LFM2-1.2B") is True


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


def test_inference_worker_resolves_kernels_from_the_base_config(monkeypatch, tmp_path):
    from core.inference import worker as iw

    target = str(_write_config(tmp_path / "renamed", {"model_type": "nemotron_h"}))
    seen = []
    monkeypatch.setattr(
        ssm_runtime,
        "ensure_ssm_runtime",
        lambda probe, **kwargs: seen.append(
            (probe, kwargs["wants_causal_conv1d"], kwargs["is_ssm"])
        ),
    )
    refreshed = []
    monkeypatch.setattr(
        ssm_runtime, "refresh_transformers_kernel_gates", lambda: refreshed.append(1)
    )
    assert iw._ensure_ssm_kernels([("renamed", target)], resp_queue = None, hf_token = None)
    assert seen == [("renamed", True, True)]
    assert refreshed == [1]

    src = (_BACKEND / "core" / "inference" / "worker.py").read_text(encoding = "utf-8")
    assert "_ensure_ssm_kernels(ssm_targets, resp_queue, hf_token)" in src
    assert "_ensure_ssm_kernels(_ssm_targets, resp_queue, _hf_token)" in src
