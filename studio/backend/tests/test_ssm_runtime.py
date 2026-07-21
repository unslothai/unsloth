# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for utils.ssm_runtime: the inference-side auto-install of SSM/Mamba kernels.

Covers detection, wheel-first install, idempotency, the failure path, the inference
worker wiring, and a drift guard so the constants/detection stay in lockstep with the
training worker (the original source of this behaviour).
"""

import sys
import types
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from utils import ssm_runtime  # noqa: E402


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
    ],
)
def test_ssm_models_detected(name):
    assert ssm_runtime.model_is_ssm(name) is True
    # every SSM model also needs causal-conv1d
    assert ssm_runtime.model_wants_causal_conv1d(name) is True


@pytest.mark.parametrize(
    "name",
    [
        "Qwen/Qwen3-Next-80B-A3B",
        "unsloth/Qwen3.5-2B",
        "LiquidAI/LFM2-1.2B",
    ],
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
    ssm_runtime.ensure_ssm_runtime("unsloth/NVIDIA-Nemotron-3-Nano-4B")
    assert order == ["causal_conv1d", "mamba_ssm"]


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
    # A true SSM model whose mamba-ssm cannot install is fatal (cryptic mid-load import
    # otherwise). "Nemotron-3-Nano-30B-A3B" matches the SSM substrings.
    monkeypatch.setattr(ssm_runtime, "_install_kernel", lambda **k: False)
    with pytest.raises(RuntimeError):
        ssm_runtime.ensure_ssm_runtime("unsloth/Nemotron-3-Nano-30B-A3B")


def test_causal_only_install_failure_is_not_fatal(monkeypatch):
    # Qwen3-Next/LFM2 want causal-conv1d but fall back to torch; a failed install must
    # not block the load (best-effort, mirrors training).
    monkeypatch.setattr(ssm_runtime, "_install_kernel", lambda **k: False)
    ssm_runtime.ensure_ssm_runtime("Qwen/Qwen3-Next-80B-A3B")  # no raise


def test_ssm_causal_failure_nonfatal_when_mamba_ok(monkeypatch):
    # causal-conv1d is best-effort even for a true SSM model; only mamba-ssm is fatal.
    monkeypatch.setattr(
        ssm_runtime, "_install_kernel", lambda *, import_name, **_: import_name == "mamba_ssm"
    )
    ssm_runtime.ensure_ssm_runtime("unsloth/NVIDIA-Nemotron-3-Nano-4B")  # no raise


def test_install_kernel_idempotent_when_present(monkeypatch):
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


def test_install_kernel_uses_prebuilt_wheel(monkeypatch):
    # not importable before install, importable after the wheel lands
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
    assert ran == []  # wheel succeeded; no PyPI source build


def test_install_kernel_falls_back_to_source(monkeypatch):
    # no wheel -> source build -> importable after install
    states = iter([False, True])  # before install, after install
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


def test_causal_conv1d_skipped_on_windows(monkeypatch):
    # No prebuilt Windows wheel: a causal-conv1d-only model must NOT enter the source build
    # (which can hang a chat load for minutes); it falls back to torch.
    monkeypatch.setattr(ssm_runtime.sys, "platform", "win32")
    installed = []
    monkeypatch.setattr(
        ssm_runtime,
        "_install_kernel",
        lambda *, import_name, **_: installed.append(import_name) or True,
    )
    ssm_runtime.ensure_ssm_runtime("Qwen/Qwen3-Next-80B-A3B")
    assert installed == []  # never attempted to build causal-conv1d


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
    assert installed == ["mamba_ssm"]  # causal-conv1d skipped, mamba-ssm still attempted


def test_wheel_installed_but_not_importable_falls_back_to_source(monkeypatch):
    # top: not importable; after wheel: still not importable (ABI mismatch) -> source build;
    # after source build: importable.
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
    # ROCm env (hip_version set) with no wheel and no hipcc must fail clearly, not build.
    monkeypatch.setattr(ssm_runtime, "_is_importable", lambda name: False)
    monkeypatch.setattr(
        ssm_runtime, "probe_torch_wheel_env", lambda timeout = 30: {"hip_version": "6.2"}
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
    # ROCm uv source build must skip the cache to avoid reusing stale partial HIP builds.
    states = iter([False, True])
    monkeypatch.setattr(ssm_runtime, "_is_importable", lambda name: next(states))
    monkeypatch.setattr(
        ssm_runtime, "probe_torch_wheel_env", lambda timeout = 30: {"hip_version": "6.2"}
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
    src = (_BACKEND / "core" / "inference" / "worker.py").read_text()
    assert "from utils.ssm_runtime import ensure_ssm_runtime" in src
    assert "ensure_ssm_runtime(" in src


def test_inference_worker_skips_ssm_on_mlx_and_checks_lora_base():
    src = (_BACKEND / "core" / "inference" / "worker.py").read_text()
    # MLX (Apple Silicon) must not try to build CUDA/ROCm SSM kernels.
    assert 'getattr(backend, "device", None) != "mlx"' in src
    # A LoRA load must also check its base model, not just the adapter id.
    assert "mc.base_model" in src


def test_inference_worker_resolves_remote_lora_base_pre_import():
    # A remote LoRA's base (from the Hub adapter_config.json) must be resolved before the
    # transformers import so its SSM kernels are pre-installed, not too late in _handle_load.
    src = (_BACKEND / "core" / "inference" / "worker.py").read_text()
    assert "_remote_lora_base" in src


def test_inference_worker_tiers_on_base_and_gates_lora_base_only():
    src = (_BACKEND / "core" / "inference" / "worker.py").read_text()
    # Tier activation runs on the resolved base, not the raw adapter id (remote-LoRA fix).
    assert "_activate_transformers_version(_base" in src
    # The gate only adds a genuine LoRA base, never a full fine-tune's recorded (unloaded) base.
    assert "_gate_targets" in src and "_lora_base" in src


def test_inference_worker_probes_base_for_ssm_kernels():
    # Both the pre-import path and _handle_load must derive SSM targets from a real model id
    # via ssm_probe_identifier, not the raw adapter id / local checkpoint path.
    src = (_BACKEND / "core" / "inference" / "worker.py").read_text()
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
        with patch.object(fs, "_fetch_security_status", return_value = (None, None)):
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
    src = (_BACKEND / "core" / "inference" / "worker.py").read_text()
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
    tree = ast.parse((_BACKEND / "core" / "inference" / "worker.py").read_text())
    for fn in ("run_inference_process", "_handle_load"):
        gates = _call_linenos(tree, fn, "_run_security_gates")
        ssm = _call_linenos(tree, fn, "_ensure_ssm_kernels")
        assert gates, f"{fn} must call _run_security_gates"
        assert ssm, f"{fn} must call _ensure_ssm_kernels"
        assert min(gates) < min(ssm), f"{fn} must gate before installing SSM kernels"


# ── drift guard vs the training worker (single source of truth) ───────────────


def test_constants_match_training_worker():
    try:
        from core.training import worker as tw
    except Exception as exc:  # pragma: no cover - only when training deps absent
        pytest.skip(f"training worker not importable here: {exc}")

    assert set(ssm_runtime.SSM_MODEL_SUBSTRINGS) == set(tw._SSM_MODEL_SUBSTRINGS)
    assert ssm_runtime.MAMBA_SSM_PACKAGE_VERSION == tw._MAMBA_SSM_PACKAGE_VERSION
    assert ssm_runtime.MAMBA_SSM_RELEASE_TAG == tw._MAMBA_SSM_RELEASE_TAG
    assert ssm_runtime.CAUSAL_CONV1D_PACKAGE_VERSION == tw._CAUSAL_CONV1D_PACKAGE_VERSION
    assert ssm_runtime.CAUSAL_CONV1D_RELEASE_TAG == tw._CAUSAL_CONV1D_RELEASE_TAG

    # detection must agree with the training worker across SSM + non-SSM names
    for name in (
        "unsloth/NVIDIA-Nemotron-3-Nano-4B",
        "nvidia/Nemotron-H-8B",
        "tiiuae/Falcon-H1-0.5B",
        "ibm-granite/granite-4.0-h-micro",
        "Qwen/Qwen3-Next-80B",
        "LiquidAI/LFM2-1.2B",
        "unsloth/Llama-3.2-1B-Instruct",
        "unsloth/Qwen2.5-7B",
    ):
        assert ssm_runtime.model_wants_causal_conv1d(name) == tw._model_wants_causal_conv1d(
            name
        ), name
