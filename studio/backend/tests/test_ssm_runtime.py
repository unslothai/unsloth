# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for utils.ssm_runtime: the inference-side auto-install of SSM/Mamba kernels.

Covers detection, wheel-first install, idempotency, the failure path, the inference
worker wiring, and a drift guard so the constants/detection stay in lockstep with the
training worker (the original source of this behaviour).
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


class _Result:
    def __init__(
        self,
        returncode = 0,
        stdout = "",
    ):
        self.returncode = returncode
        self.stdout = stdout




@pytest.mark.parametrize(
    "name",
    [
        "unsloth/NVIDIA-Nemotron-3-Nano-4B",
        "unsloth/Nemotron-3-Nano-30B-A3B",
        "nvidia/Nemotron-H-8B",
        "tiiuae/Falcon-H1-0.5B-Instruct",
        "ibm-granite/granite-4.0-h-micro",
        "ibm/granitemoehybrid-test",
    ],
)
def test_ssm_models_detected(name):
    assert ssm_runtime.model_is_ssm(name) is True
    assert ssm_runtime.model_wants_causal_conv1d(name) is True


@pytest.mark.parametrize(
    "name",
    [
        "Qwen/Qwen3-Next-80B-A3B",
        "unsloth/Qwen3.5-2B",
        "LiquidAI/LFM2-1.2B",
        "state-spaces/mamba-2.8b-hf",
        "ai21labs/Jamba-v0.1",
        "Zyphra/Zamba2-7B",
        "ibm/Bamba-9B",
        "tiiuae/falcon-mamba-7b",
    ],
)
def test_causal_conv1d_only_models(name):
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




def test_probe_lora_uses_base_not_adapter_name():
    probe = ssm_runtime.ssm_probe_identifier("user/falcon-h1-lora", "meta-llama/Llama-3-8B")
    assert probe == "meta-llama/Llama-3-8B"
    assert ssm_runtime.model_is_ssm(probe) is False


def test_probe_lora_on_ssm_base_detected():
    probe = ssm_runtime.ssm_probe_identifier("user/my-adapter", "nvidia/Nemotron-H-8B")
    assert ssm_runtime.model_is_ssm(probe) is True


def test_probe_plain_hf_id_unchanged():
    assert ssm_runtime.ssm_probe_identifier("nvidia/Nemotron-H-8B") == "nvidia/Nemotron-H-8B"


def test_probe_local_path_uses_basename(tmp_path):
    d = tmp_path / "falcon-h1-experiment" / "llama-checkpoint"
    d.mkdir(parents = True)
    probe = ssm_runtime.ssm_probe_identifier(str(d))
    assert probe == "llama-checkpoint"
    assert ssm_runtime.model_is_ssm(probe) is False


def test_probe_local_ssm_checkpoint_basename_detected(tmp_path):
    d = tmp_path / "runs" / "nemotron-h-finetune"
    d.mkdir(parents = True)
    assert ssm_runtime.model_is_ssm(ssm_runtime.ssm_probe_identifier(str(d))) is True




def test_noop_for_non_ssm_model(monkeypatch):
    calls = []
    monkeypatch.setattr(ssm_runtime, "_install_kernel", lambda **k: calls.append(k) or True)
    ssm_runtime.ensure_ssm_runtime("unsloth/Llama-3.2-1B-Instruct", run = lambda *a, **k: _Result())
    assert calls == []


def test_ssm_model_installs_causal_then_mamba(monkeypatch):
    order = []

    def fake_install(*, import_name, **_):
        order.append(import_name)
        return True

    monkeypatch.setattr(ssm_runtime, "_install_kernel", fake_install)
    ssm_runtime.ensure_ssm_runtime("unsloth/NVIDIA-Nemotron-3-Nano-4B")
    expected = ["mamba_ssm"] if sys.platform == "win32" else ["causal_conv1d", "mamba_ssm"]
    assert order == expected


@pytest.mark.skipif(
    sys.platform == "win32", reason = "causal-conv1d is skipped on Windows (no prebuilt wheel)"
)
def test_causal_only_model_skips_mamba(monkeypatch):
    order = []
    monkeypatch.setattr(
        ssm_runtime,
        "_install_kernel",
        lambda *, import_name, **_: order.append(import_name) or True,
    )
    ssm_runtime.ensure_ssm_runtime("Qwen/Qwen3-Next-80B-A3B")
    assert order == ["causal_conv1d"]


def test_failure_raises_runtime_error(monkeypatch):
    # A true SSM model whose mamba-ssm cannot install is fatal, or the load dies on a cryptic mid-load import.
    monkeypatch.setattr(ssm_runtime, "_install_kernel", lambda **k: False)
    with pytest.raises(RuntimeError):
        ssm_runtime.ensure_ssm_runtime("unsloth/Nemotron-3-Nano-30B-A3B")


def test_causal_only_install_failure_is_not_fatal(monkeypatch):
    # Qwen3-Next/LFM2 want causal-conv1d but fall back to torch, so a failed install must not block the load.
    monkeypatch.setattr(ssm_runtime, "_install_kernel", lambda **k: False)
    ssm_runtime.ensure_ssm_runtime("Qwen/Qwen3-Next-80B-A3B")


def test_ssm_causal_failure_nonfatal_when_mamba_ok(monkeypatch):
    monkeypatch.setattr(
        ssm_runtime, "_install_kernel", lambda *, import_name, **_: import_name == "mamba_ssm"
    )
    ssm_runtime.ensure_ssm_runtime("unsloth/NVIDIA-Nemotron-3-Nano-4B")


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
    assert called == []


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
    states = iter([False, True])
    monkeypatch.setattr(ssm_runtime, "_is_importable", lambda name: next(states))
    monkeypatch.setattr(ssm_runtime, "probe_torch_wheel_env", lambda timeout = 30: {"x": "y"})
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
    assert ran == []


def test_install_kernel_heartbeats_during_prebuilt_wheel(monkeypatch):
    import threading
    import time

    monkeypatch.setattr(ssm_runtime, "_HEARTBEAT_SECONDS", 0.05)
    monkeypatch.setattr(ssm_runtime, "_is_importable", lambda name: False)
    monkeypatch.setattr(ssm_runtime, "probe_torch_wheel_env", lambda timeout = 30: {})
    monkeypatch.setattr(
        ssm_runtime,
        "direct_wheel_url",
        lambda **k: "https://example/causal_conv1d-1.6.1.whl",
    )
    monkeypatch.setattr(ssm_runtime, "url_exists", lambda u: True)

    statuses = []
    released = threading.Event()

    def slow_install_wheel(url, **k):
        assert released.wait(1.0)
        return [("uv", _Result(returncode = 1, stdout = "nope"))]

    monkeypatch.setattr(ssm_runtime, "install_wheel", slow_install_wheel)
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
    # The first torch import can be quiet long enough to trip the inactivity deadline: keep heartbeats going.
    import threading
    import time

    monkeypatch.setattr(ssm_runtime, "_HEARTBEAT_SECONDS", 0.05)
    monkeypatch.setattr(ssm_runtime, "probe_torch_wheel_env", lambda timeout = 30: {})
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
    import time

    monkeypatch.setattr(ssm_runtime, "_HEARTBEAT_SECONDS", 0.05)
    statuses = []
    with ssm_runtime._heartbeat(statuses.append, "still going"):
        time.sleep(0.12)
    n = len(statuses)
    time.sleep(0.15)
    assert len(statuses) == n, statuses


def test_install_kernel_falls_back_to_source(monkeypatch):
    states = iter([False, True])
    monkeypatch.setattr(ssm_runtime, "_is_importable", lambda name: next(states))
    monkeypatch.setattr(ssm_runtime, "probe_torch_wheel_env", lambda timeout = 30: {})
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




def test_is_importable_invalidates_caches(monkeypatch):
    calls = []
    monkeypatch.setattr(ssm_runtime.importlib, "invalidate_caches", lambda: calls.append(1))
    assert ssm_runtime._is_importable("sys") is True
    assert calls


@pytest.mark.parametrize(
    "exc",
    [
        ImportError("no module"),
        OSError("undefined symbol: cuLaunchKernel"),
        RuntimeError("CUDA error: ABI mismatch"),
    ],
)
def test_is_importable_treats_broken_kernel_as_not_importable(monkeypatch, exc):
    # ABI-incompatible kernels raise OSError/RuntimeError, not ImportError, and all must read as
    # not-importable. _is_importable calls bare __import__(), so patching ssm_runtime.__import__
    # leaves real `import` statements untouched.
    def _raise(name):
        raise exc

    monkeypatch.setattr(ssm_runtime, "__import__", _raise, raising = False)
    monkeypatch.setattr(ssm_runtime.importlib, "invalidate_caches", lambda: None)
    assert ssm_runtime._is_importable("causal_conv1d") is False


def test_causal_conv1d_skipped_on_windows(monkeypatch):
    # No prebuilt Windows wheel: a causal-conv1d-only model must NOT enter the source build and hang the load.
    monkeypatch.setattr(ssm_runtime.sys, "platform", "win32")
    installed = []
    monkeypatch.setattr(
        ssm_runtime,
        "_install_kernel",
        lambda *, import_name, **_: installed.append(import_name) or True,
    )
    ssm_runtime.ensure_ssm_runtime("Qwen/Qwen3-Next-80B-A3B")
    assert installed == []


def test_ssm_model_on_windows_still_installs_mamba(monkeypatch):
    # A true SSM hybrid still needs mamba-ssm on Windows; only causal-conv1d is skipped.
    monkeypatch.setattr(ssm_runtime.sys, "platform", "win32")
    installed = []
    monkeypatch.setattr(
        ssm_runtime,
        "_install_kernel",
        lambda *, import_name, **_: installed.append(import_name) or True,
    )
    ssm_runtime.ensure_ssm_runtime("unsloth/NVIDIA-Nemotron-3-Nano-4B")
    assert installed == ["mamba_ssm"]


def test_wheel_installed_but_not_importable_falls_back_to_source(monkeypatch):
    # Not importable, still not importable after the wheel (ABI mismatch), importable after the source build.
    states = iter([False, False, True])
    monkeypatch.setattr(ssm_runtime, "_is_importable", lambda name: next(states))
    monkeypatch.setattr(ssm_runtime, "probe_torch_wheel_env", lambda timeout = 30: {})
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
    monkeypatch.setattr(ssm_runtime, "_is_importable", lambda name: False)
    monkeypatch.setattr(
        ssm_runtime, "probe_torch_wheel_env", lambda timeout = 30: {"hip_version": "6.2"}
    )
    monkeypatch.setattr(ssm_runtime, "direct_wheel_url", lambda **k: None)
    monkeypatch.setattr(ssm_runtime.shutil, "which", lambda name: None)
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
    assert ran == []


def test_source_build_reinstalls_to_replace_broken_wheel(monkeypatch):
    # Reached only when not importable, perhaps a broken wheel, so the source build must reinstall, not no-op.
    states = iter([False, True])
    monkeypatch.setattr(ssm_runtime, "_is_importable", lambda name: next(states))
    monkeypatch.setattr(ssm_runtime, "probe_torch_wheel_env", lambda timeout = 30: {})
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
    states = iter([False, True])
    monkeypatch.setattr(ssm_runtime, "_is_importable", lambda name: next(states))
    monkeypatch.setattr(
        ssm_runtime, "probe_torch_wheel_env", lambda timeout = 30: {"hip_version": "6.2"}
    )
    monkeypatch.setattr(ssm_runtime, "direct_wheel_url", lambda **k: None)
    monkeypatch.setattr(ssm_runtime.shutil, "which", lambda name: "/usr/bin/" + name)
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




def test_inference_worker_calls_ensure_ssm_runtime():
    src = (_BACKEND / "core" / "inference" / "worker.py").read_text(encoding = "utf-8")
    assert "from utils.ssm_runtime import ensure_ssm_runtime" in src
    assert "ensure_ssm_runtime(" in src


def test_inference_worker_skips_ssm_on_mlx_and_checks_lora_base():
    src = (_BACKEND / "core" / "inference" / "worker.py").read_text(encoding = "utf-8")
    assert 'getattr(backend, "device", None) != "mlx"' in src
    assert "mc.base_model" in src


def test_inference_worker_resolves_remote_lora_base_pre_import():
    # A remote LoRA's base must resolve before the transformers import so its SSM kernels are pre-installed.
    src = (_BACKEND / "core" / "inference" / "worker.py").read_text(encoding = "utf-8")
    assert "_remote_lora_base" in src


def test_inference_worker_tiers_on_base_and_gates_lora_base_only():
    src = (_BACKEND / "core" / "inference" / "worker.py").read_text(encoding = "utf-8")
    assert "_activate_transformers_version(_base" in src
    assert "_gate_targets" in src and "_lora_base" in src


def test_inference_worker_probes_base_for_ssm_kernels():
    # Both paths derive SSM targets from a real model id, not the raw adapter id or a local checkpoint path.
    src = (_BACKEND / "core" / "inference" / "worker.py").read_text(encoding = "utf-8")
    assert src.count("ssm_probe_identifier(") >= 2


def test_pre_import_gate_is_transformers_free():
    # The pre-import gate must not import transformers: model_config would snapshot SSM availability too early.
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

    # Restore the originals in finally: popping utils.models.model_config without restoring it makes
    # a later importer get a fresh instance, so tests that patched the first miss the real network path.
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
        # Rebind the original module objects so later tests see the same instances they captured at import time.
        for m in [m for m in list(_sys.modules) if _is_gated_module(m) and m not in _saved]:
            _sys.modules.pop(m, None)
        _sys.modules.update(_saved)


def test_pre_import_gate_skips_subdir_computation():
    # The worker's preflight passes compute_subdirs=False so transformers is not imported before the kernels.
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
    # The SSM install is name-based and can source-build, so a blocked-code model is refused first, both paths.
    import ast
    tree = ast.parse((_BACKEND / "core" / "inference" / "worker.py").read_text(encoding = "utf-8"))
    for fn in ("run_inference_process", "_handle_load"):
        gates = _call_linenos(tree, fn, "_run_security_gates")
        ssm = _call_linenos(tree, fn, "_ensure_ssm_kernels")
        assert gates, f"{fn} must call _run_security_gates"
        assert ssm, f"{fn} must call _ensure_ssm_kernels"
        assert min(gates) < min(ssm), f"{fn} must gate before installing SSM kernels"




def test_constants_match_training_worker():
    try:
        from core.training import worker as tw
    except Exception as exc:
        pytest.skip(f"training worker not importable here: {exc}")

    assert set(ssm_runtime.SSM_MODEL_SUBSTRINGS) == set(tw._SSM_MODEL_SUBSTRINGS)
    assert set(ssm_runtime.CAUSAL_CONV1D_MODEL_SUBSTRINGS) == set(
        tw._CAUSAL_CONV1D_MODEL_SUBSTRINGS
    )
    assert ssm_runtime.MAMBA_SSM_PACKAGE_VERSION == tw._MAMBA_SSM_PACKAGE_VERSION
    assert ssm_runtime.MAMBA_SSM_RELEASE_TAG == tw._MAMBA_SSM_RELEASE_TAG
    assert ssm_runtime.CAUSAL_CONV1D_PACKAGE_VERSION == tw._CAUSAL_CONV1D_PACKAGE_VERSION
    assert ssm_runtime.CAUSAL_CONV1D_RELEASE_TAG == tw._CAUSAL_CONV1D_RELEASE_TAG

    for name in (
        "unsloth/NVIDIA-Nemotron-3-Nano-4B",
        "nvidia/Nemotron-H-8B",
        "tiiuae/Falcon-H1-0.5B",
        "ibm-granite/granite-4.0-h-micro",
        "Qwen/Qwen3-Next-80B",
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




def _write_config(directory: Path, config: dict) -> Path:
    directory.mkdir(parents = True, exist_ok = True)
    (directory / "config.json").write_text(json.dumps(config), encoding = "utf-8")
    return directory


def test_renamed_local_checkpoint_resolves_causal_conv1d_from_its_config(tmp_path):
    """A local Qwen3-Next checkpoint whose folder has no allowlisted substring must
    still be detected, so the availability hook is installed for it."""
    checkpoint = _write_config(
        tmp_path / "my-model",
        {"model_type": "qwen3_next", "architectures": ["Qwen3NextForCausalLM"]},
    )
    target = str(checkpoint)

    assert ssm_runtime.model_wants_causal_conv1d(target) is False
    assert ssm_runtime.resolved_model_wants_causal_conv1d(target, target, None) is True
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
