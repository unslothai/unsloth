# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for the on-demand FlashInfer install behind NVFP4 (``diffusion_nvfp4_install.py``).

Hermetic: the installer subprocess, the torch probe, the device capability and the installed
distribution set are all fakes, so nothing here touches the network or the running venv.
"""

from __future__ import annotations

import os
import pathlib
import re
import sys
import threading
import time
import types

import pytest

from core.inference import diffusion_nvfp4_dispatch as dispatch
from core.inference import diffusion_nvfp4_install as inst
from core.inference import diffusion_nvfp4_ops as ops

_INFERENCE_DIR = pathlib.Path(inst.__file__).resolve().parent

_BASE_DISTS = {
    "torch": "2.12.1+cu130",
    "triton": "3.7.1",
    "nvidia-cublas": "13.1.0.3",
    "numpy": "2.3.5",
    "packaging": "25.0",
}
_ADDED_BY_INSTALL = {
    "flashinfer-python": "0.6.6",
    "apache-tvm-ffi": "0.1.9",
    "nvidia-cutlass-dsl": "4.4.2",
}


class _Result:
    def __init__(
        self,
        returncode = 0,
        stdout = "",
    ):
        self.returncode = returncode
        self.stdout = stdout


class _FakeEnv:
    """The venv as the installer sees it: a distribution table, a torch probe, and a subprocess."""

    def __init__(
        self,
        *,
        cuda = "13.0",
        fail_install = False,
        install_moves = None,
        install_delay = 0.0,
    ):
        self.dists = dict(_BASE_DISTS)
        self.cuda = cuda
        self.torch_version = self.dists["torch"]
        self.importable = False
        self.fail_install = fail_install
        self.install_moves = install_moves or {}
        self.install_delay = install_delay
        self.commands: list[list[str]] = []
        self.constraints_seen: list[str] = []
        self.lock = threading.Lock()

    def probe(self, timeout = 60):
        return {"torch_version": self.torch_version, "cuda_version": self.cuda}

    def run(self, cmd, **kwargs):
        cmd = [str(c) for c in cmd]
        with self.lock:
            self.commands.append(cmd)
        if cmd[1:3] == ["-c", cmd[2]] and "import flashinfer" in cmd[2]:
            return _Result(
                0 if self.importable else 1, "0.6.6" if self.importable else "ImportError"
            )
        if "uninstall" in cmd:
            for name in cmd[cmd.index("uninstall") + 1 :]:
                self.dists.pop(name, None)
            self.importable = "flashinfer-python" in self.dists
            return _Result(0)
        if "install" in cmd:
            if "-c" in cmd:
                path = cmd[cmd.index("-c") + 1]
                with open(path, encoding = "utf-8") as fh:
                    self.constraints_seen.append(fh.read())
            if self.install_delay:
                time.sleep(self.install_delay)
            if self.fail_install and any(a.startswith("flashinfer-jit-cache==") for a in cmd):
                return _Result(1, "ERROR: no matching distribution")
            if any(a.startswith("flashinfer-python==") for a in cmd):
                self.dists.update(_ADDED_BY_INSTALL)
                for name, version in self.install_moves.items():
                    self.dists[name] = version
                    if name == "torch":
                        self.torch_version = version
                self.importable = not self.install_moves
            elif any(a.startswith("flashinfer-jit-cache==") for a in cmd):
                self.dists["flashinfer-jit-cache"] = "0.6.6+cu130"
            else:
                # A rollback restore: name==old --no-deps.
                for spec in cmd:
                    if "==" in spec:
                        name, version = spec.split("==", 1)
                        self.dists[name] = version
                        if name == "torch":
                            self.torch_version = version
            return _Result(0)
        return _Result(1, "unexpected command")

    def installs(self):
        return [c for c in self.commands if "install" in c and "uninstall" not in c]


def _fake_torch(*, cuda = "13.0", hip = None):
    torch = types.ModuleType("torch")
    torch.version = types.SimpleNamespace(cuda = cuda, hip = hip)
    return torch


@pytest.fixture(autouse = True)
def _clean(monkeypatch):
    monkeypatch.delenv(inst.FLASHINFER_INSTALL_ENV, raising = False)
    monkeypatch.delenv(ops.NVFP4_BACKEND_ENV, raising = False)
    for name in inst._CUSTOM_INDEX_ENVS:
        monkeypatch.delenv(name, raising = False)
    inst.reset_install_state()
    yield
    inst.reset_install_state()


@pytest.fixture
def env(monkeypatch):
    """An eligible host (Linux, CUDA 13.0 torch, sm_100) with flashinfer absent."""
    fake = _FakeEnv()
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setitem(sys.modules, "torch", _fake_torch())
    monkeypatch.setattr(ops, "_device_capability", lambda device = None: (10, 0))
    monkeypatch.setattr(
        inst,
        "_import_flashinfer",
        lambda: (True, "0.6.6") if fake.importable else (False, "ModuleNotFoundError: flashinfer"),
    )
    monkeypatch.setattr(inst, "_dist_version", lambda name: fake.dists.get(name))
    monkeypatch.setattr(inst, "installed_distributions", lambda: dict(fake.dists))
    monkeypatch.setattr(inst, "_torch_probe", fake.probe)
    monkeypatch.setattr(inst, "_reachable", lambda url: True)
    monkeypatch.setattr(inst, "_uv_executable", lambda: "uv")
    monkeypatch.setattr(inst, "_nvcc_available", lambda: False)
    import utils.utils as uu

    monkeypatch.setattr(uu, "hf_env_offline", lambda: False)
    return fake


def _ensure(fake, device = 0):
    return inst.ensure_flashinfer_for_nvfp4(device, run = fake.run)


def test_installs_pinned_flashinfer_and_matching_jit_cache_when_missing(env):
    ok, reason = _ensure(env)
    assert ok, reason
    installs = env.installs()
    assert len(installs) == 2
    main, jit = installs
    assert "flashinfer-python==0.6.6" in main and "--only-binary" in main
    assert "--no-deps" not in main  # tvm-ffi and friends are real import deps
    assert "flashinfer-jit-cache==0.6.6" in jit and "--no-deps" in jit
    assert jit[jit.index("--index-url") + 1] == "https://flashinfer.ai/whl/cu130"
    # Every distribution already present is pinned exactly, torch/triton/nvidia included.
    pins = env.constraints_seen[0].splitlines()
    for name, version in _BASE_DISTS.items():
        assert f"{name}=={version}" in pins
    assert inst.last_install_reason() is None


def test_constraints_file_is_deleted_after_the_install(env, monkeypatch):
    written = []
    real = inst._write_constraints

    def _spy(pins):
        path = real(pins)
        written.append(path)
        return path

    monkeypatch.setattr(inst, "_write_constraints", _spy)
    _ensure(env)
    assert written and not os.path.exists(written[0])


def test_already_installed_runs_nothing(env):
    env.importable = True
    ok, reason = _ensure(env)
    assert ok and "already installed" in reason
    assert env.commands == []


def test_installed_but_broken_is_not_repaired(env):
    env.dists["flashinfer-python"] = "0.5.3"
    ok, reason = _ensure(env)
    assert not ok and "does not import" in reason
    assert env.commands == []


@pytest.mark.parametrize(
    "setup, needle",
    [
        (lambda mp: mp.setattr(sys, "platform", "darwin"), "Linux wheels only"),
        (lambda mp: mp.setattr(sys, "platform", "win32"), "Linux wheels only"),
        (lambda mp: mp.setitem(sys.modules, "torch", _fake_torch(cuda = None, hip = "6.4")), "ROCm"),
        (lambda mp: mp.setitem(sys.modules, "torch", _fake_torch(cuda = None)), "CUDA build"),
        (lambda mp: mp.setattr(ops, "_device_capability", lambda device = None: (9, 0)), "sm_90"),
        (lambda mp: mp.setattr(ops, "_device_capability", lambda device = None: (8, 9)), "sm_89"),
        (lambda mp: mp.setattr(ops, "_device_capability", lambda device = None: None), "capability"),
    ],
)
def test_never_installs_on_an_ineligible_host(env, monkeypatch, setup, needle):
    setup(monkeypatch)
    ok, reason = _ensure(env)
    assert not ok and needle in reason
    assert env.commands == []


def test_torch_version_change_rolls_back_and_refuses(env):
    env.install_moves = {"torch": "2.13.0+cu130"}
    ok, reason = _ensure(env)
    assert not ok and "rolled back" in reason and "torch 2.12.1+cu130 -> 2.13.0+cu130" in reason
    # Everything the install added is gone and torch is back where it was.
    assert env.dists == _BASE_DISTS
    uninstall = [c for c in env.commands if "uninstall" in c]
    assert uninstall and set(_ADDED_BY_INSTALL) <= set(uninstall[0])
    assert any("torch==2.12.1+cu130" in c and "--no-deps" in c for c in env.installs())
    assert inst.last_install_reason() == reason


def test_torch_probe_drift_without_metadata_drift_also_rolls_back(env, monkeypatch):
    probes = iter(
        [
            {"torch_version": "2.12.1+cu130", "cuda_version": "13.0"},
            {"torch_version": "2.12.1+cu130", "cuda_version": "12.8"},
        ]
    )
    monkeypatch.setattr(inst, "_torch_probe", lambda timeout = 60: next(probes))
    ok, reason = _ensure(env)
    assert not ok and "rolled back" in reason and "torch reads differently" in reason
    assert env.dists == _BASE_DISTS


def test_failed_jit_cache_step_rolls_back_the_first_step(env):
    env.fail_install = True
    ok, reason = _ensure(env)
    assert not ok and "rolled back" in reason and "installer failed" in reason
    assert env.dists == _BASE_DISTS


def test_failed_import_after_install_rolls_back(env, monkeypatch):
    real_run = env.run

    def _run(cmd, **kwargs):
        result = real_run(cmd, **kwargs)
        if "import flashinfer" in " ".join(map(str, cmd)):
            return _Result(1, "ImportError: libcudart.so.13")
        return result

    ok, reason = inst.ensure_flashinfer_for_nvfp4(0, run = _run)
    assert not ok and "import flashinfer failed" in reason
    assert env.dists == _BASE_DISTS


def test_offline_refuses_with_a_reason_and_is_not_memoised(env, monkeypatch):
    import utils.utils as uu

    monkeypatch.setattr(uu, "hf_env_offline", lambda: True)
    ok, reason = _ensure(env)
    assert not ok and "offline" in reason
    assert env.commands == []
    assert inst.last_install_reason() == reason
    monkeypatch.setattr(uu, "hf_env_offline", lambda: False)
    assert _ensure(env)[0]


def test_uv_offline_refuses_before_any_index_probe(env, monkeypatch):
    probes: list = []
    monkeypatch.setattr(inst, "_reachable", lambda url: probes.append(url) or True)
    monkeypatch.setenv("UV_OFFLINE", "1")
    ok, reason = _ensure(env)
    assert not ok and "offline" in reason
    assert probes == [] and env.commands == []
    monkeypatch.delenv("UV_OFFLINE")
    assert _ensure(env)[0]


def test_unreachable_index_refuses_without_running_the_installer(env, monkeypatch):
    monkeypatch.setattr(inst, "_reachable", lambda url: False)
    ok, reason = _ensure(env)
    assert not ok and "is not reachable" in reason
    assert env.installs() == []
    monkeypatch.setattr(inst, "_reachable", lambda url: True)
    assert _ensure(env)[0]


@pytest.mark.parametrize("value", ["0", "off", "false", "no"])
def test_opt_out_env_refuses(env, monkeypatch, value):
    monkeypatch.setenv(inst.FLASHINFER_INSTALL_ENV, value)
    ok, reason = _ensure(env)
    assert not ok and inst.FLASHINFER_INSTALL_ENV in reason
    assert env.commands == []


def test_explicit_torchao_backend_never_installs(env, monkeypatch):
    monkeypatch.setenv(ops.NVFP4_BACKEND_ENV, "torchao")
    ok, reason = _ensure(env)
    assert not ok and "torchao" in reason
    assert env.commands == []


def test_concurrent_loads_install_once(env):
    env.install_delay = 0.3
    results = []
    barrier = threading.Barrier(4)

    def _worker():
        barrier.wait()
        results.append(_ensure(env))

    threads = [threading.Thread(target = _worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)
    assert len(results) == 4 and all(ok for ok, _ in results)
    assert sum(1 for c in env.installs() if "flashinfer-python==0.6.6" in c) == 1


def test_a_failed_install_is_attempted_once_per_process(env):
    env.fail_install = True
    assert not _ensure(env)[0]
    count = len(env.installs())
    assert not _ensure(env)[0]
    assert len(env.installs()) == count


def test_no_jit_cache_and_no_nvcc_refuses(env):
    env.cuda = "12.6"
    ok, reason = _ensure(env)
    assert not ok and "no nvcc" in reason and "12.6" in reason
    assert env.installs() == []


def test_no_jit_cache_with_nvcc_installs_flashinfer_only(env, monkeypatch):
    env.cuda = "14.0"
    monkeypatch.setattr(inst, "_nvcc_available", lambda: True)
    ok, reason = _ensure(env)
    assert ok, reason
    assert [c for c in env.installs() if any("jit-cache" in a for a in c)] == []


def test_stray_jit_cache_of_another_version_refuses(env):
    env.dists["flashinfer-jit-cache"] = "0.6.4+cu130"
    ok, reason = _ensure(env)
    assert not ok and "flashinfer-jit-cache 0.6.4+cu130" in reason
    assert env.installs() == []


def test_success_forgets_the_dispatch_availability_memo(env, monkeypatch):
    monkeypatch.setattr(dispatch, "_AVAILABLE", (False, "ModuleNotFoundError"))
    assert _ensure(env)[0]
    assert dispatch._AVAILABLE is None


@pytest.mark.parametrize(
    "cuda, tag",
    [
        ("12.8", "cu128"),
        ("12.9", "cu129"),
        ("13.0", "cu130"),
        ("13.2", "cu130"),
        ("12.6", None),
        ("14.0", None),
        (None, None),
        ("garbage", None),
    ],
)
def test_jit_cache_tag_matches_the_running_cuda(cuda, tag):
    assert inst.jit_cache_tag(cuda) == tag


def test_status_fields_carry_the_reason_only_for_torchao(env, monkeypatch):
    monkeypatch.setenv(inst.FLASHINFER_INSTALL_ENV, "0")
    _ensure(env)
    assert inst.nvfp4_backend_fields("torchao")["transformer_quant_backend_reason"]
    assert inst.nvfp4_backend_fields("flashinfer") == {
        "transformer_quant_backend": "flashinfer",
        "transformer_quant_backend_reason": None,
    }
    assert inst.nvfp4_backend_fields(None)["transformer_quant_backend_reason"] is None


def test_never_raises(env, monkeypatch):
    def _boom(device):
        raise RuntimeError("probe exploded")

    monkeypatch.setattr(inst, "_host_refusal", _boom)
    ok, reason = _ensure(env)
    assert not ok and "probe exploded" in reason


def _load_pipeline_body(path: pathlib.Path) -> str:
    source = path.read_text(encoding = "utf-8")
    start = source.index("    def load_pipeline(")
    return source[start : source.index("\n    def ", start + 10)]


def _flat(text: str) -> str:
    """Whitespace-insensitive, so a formatter re-wrapping the condition does not fail the test."""
    return re.sub(r"[\s,]+", " ", text).replace("( ", "(").replace(" )", ")")


def test_image_loader_installs_before_the_load_locks():
    body = _load_pipeline_body(_INFERENCE_DIR / "diffusion.py")
    call = body.index("ensure_flashinfer_for_nvfp4(")
    assert call < body.index("with self._lock:")
    assert _flat(
        "TQ_NVFP4 in (normalize_transformer_quant(transformer_quant), _pipeline_prequant_planned)"
    ) in _flat(body)


def test_video_loader_installs_after_teardown_and_outside_the_locks():
    body = _load_pipeline_body(_INFERENCE_DIR / "video.py")
    call = body.index("ensure_flashinfer_for_nvfp4(")
    assert body.index("self._teardown_state_locked()") < call
    # Dedented to the method body: not nested in any with-block.
    line = body[body.rindex("\n", 0, call) + 1 : call]
    assert line == " " * 12
    assert _flat(
        '"nvfp4" in (normalize_transformer_quant(transformer_quant), _video_auto_denoiser_planned)'
    ) in _flat(body)


def test_a_configured_mirror_skips_the_pypi_probe_but_not_the_jit_cache_one(env, monkeypatch):
    probed = []
    monkeypatch.setattr(inst, "_reachable", lambda url: probed.append(url) or True)
    monkeypatch.setenv("UV_INDEX_URL", "https://mirror.example/simple")
    assert _ensure(env)[0]
    assert probed == ["https://flashinfer.ai/whl/cu130/flashinfer-jit-cache/"]


def test_status_reason_falls_back_to_a_cached_preflight_failure(monkeypatch):
    monkeypatch.setitem(ops._PREFLIGHT, 0, {"ok": False, "reason": "JIT build failed"})
    fields = inst.nvfp4_backend_fields("torchao")
    assert (
        fields["transformer_quant_backend_reason"]
        == "flashinfer preflight failed: JIT build failed"
    )


class _Logger:
    def __init__(self):
        self.records = []

    def info(self, fmt, *args):
        self.records.append(("info", fmt % args))

    def warning(self, fmt, *args):
        self.records.append(("warning", fmt % args))


def test_policy_refusals_log_at_info_and_failures_at_warning(env, monkeypatch):
    logger = _Logger()
    monkeypatch.setattr(ops, "_device_capability", lambda device = None: (9, 0))
    inst.ensure_flashinfer_for_nvfp4(0, logger = logger, run = env.run)
    assert logger.records[-1][0] == "info"
    monkeypatch.setattr(ops, "_device_capability", lambda device = None: (10, 0))
    env.fail_install = True
    inst.ensure_flashinfer_for_nvfp4(0, logger = logger, run = env.run)
    assert logger.records[-1][0] == "warning" and "rolled back" in logger.records[-1][1]


def test_status_callback_sees_the_install_and_its_outcome(env):
    seen = []
    ok, _ = inst.ensure_flashinfer_for_nvfp4(0, status_cb = seen.append, run = env.run)
    assert ok
    assert seen[0].startswith("installing flashinfer-python 0.6.6") and "cu130" in seen[0]
    assert seen[-1] == "installed flashinfer 0.6.6 for NVFP4"


def test_local_only_load_never_installs(env):
    ok, reason = inst.ensure_flashinfer_for_nvfp4(0, run = env.run, local_files_only = True)
    assert not ok and "local-only" in reason
    assert env.installs() == []


def test_status_reason_is_bound_to_the_backend_that_loaded():
    # An image model left on torchao by a refused install keeps its reason after a video load succeeds.
    class _Backend:
        pass

    image, video = _Backend(), _Backend()
    inst.reset_install_state()
    inst.record_install_reason(image, False, "offline: flashinfer is not downloaded", 0)
    inst.record_install_reason(video, True, "installed", 1)
    assert inst.nvfp4_backend_fields("torchao", owner = image)[
        "transformer_quant_backend_reason"
    ] == ("offline: flashinfer is not downloaded")
    assert (
        inst.nvfp4_backend_fields("flashinfer", owner = image)["transformer_quant_backend_reason"]
        is None
    )
    inst.reset_install_state()


def test_status_preflight_reason_is_the_loaded_device_only(monkeypatch):
    # Image on GPU 0 and video on GPU 1 both fell back in preflight, for different reasons.
    class _Backend:
        pass

    image, video = _Backend(), _Backend()
    inst.reset_install_state()
    monkeypatch.setitem(ops._PREFLIGHT, 0, {"ok": False, "reason": "JIT build failed"})
    monkeypatch.setitem(ops._PREFLIGHT, 1, {"ok": False, "reason": "sm_120 unsupported"})
    inst.record_install_reason(image, True, "already installed", 0)
    inst.record_install_reason(video, True, "already installed", "cuda:1")
    assert (
        inst.nvfp4_backend_fields("torchao", owner = image)["transformer_quant_backend_reason"]
        == "flashinfer preflight failed: JIT build failed"
    )
    assert (
        inst.nvfp4_backend_fields("torchao", owner = video)["transformer_quant_backend_reason"]
        == "flashinfer preflight failed: sm_120 unsupported"
    )
    inst.reset_install_state()


def test_image_loader_binds_the_reason_only_after_the_resident_model_is_unloaded():
    # A superseded load raises at the cancel check before the swap; the resident model keeps its reason.
    import inspect

    from core.inference import diffusion

    src = inspect.getsource(diffusion.DiffusionBackend)
    ensure_at = src.index("ensure_flashinfer_for_nvfp4(")
    call = src[ensure_at : src.index(")", ensure_at)]
    assert "owner" not in call
    unload_at = src.index("self._unload_locked()", ensure_at)
    assert src.index("record_install_reason(self", ensure_at) > unload_at


def test_install_gates_skip_kinds_the_dense_quant_path_cannot_reach():
    import inspect

    from core.inference import diffusion, video

    img = inspect.getsource(diffusion.DiffusionBackend)
    gate = img[: img.index("ensure_flashinfer_for_nvfp4(")].rsplit("if ", 1)[-1]
    assert gate.startswith("dense_quant_supported_kind(kind)")
    vid = inspect.getsource(video.VideoBackend)
    gate = vid[: vid.index("ensure_flashinfer_for_nvfp4(")].rsplit("if ", 1)[-1]
    assert gate.startswith('kind == "pipeline"')
