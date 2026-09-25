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
        concurrent_add = None,
    ):
        self.dists = dict(_BASE_DISTS)
        self.cuda = cuda
        self.torch_version = self.dists["torch"]
        self.importable = False
        self.fail_install = fail_install
        self.fail_verify = False
        self.install_moves = install_moves or {}
        self.install_delay = install_delay
        # What another installer (the built-in terminal) adds to the venv while this install runs.
        self.concurrent_add = concurrent_add or {}
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
            good = self.importable and not self.fail_verify
            return _Result(0 if good else 1, "0.6.6" if good else "ImportError")
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
            self.dists.update(self.concurrent_add)
            if self.fail_install and any(a.startswith("flashinfer-jit-cache==") for a in cmd):
                return _Result(1, "ERROR: no matching distribution")
            if any(a.startswith("flashinfer-python==") for a in cmd):
                self.dists.update(_ADDED_BY_INSTALL)
                for name, version in self.install_moves.items():
                    self.dists[name] = version
                    if name == "torch":
                        self.torch_version = version
                self.importable = not self.install_moves
                # uv reports an upgrade or downgrade as `- name==old` then `+ name==new`, like an addition.
                return _Result(
                    0,
                    f"Installed {len(_ADDED_BY_INSTALL)} packages in 1.2s\n"
                    + "\n".join(
                        f" + {n}=={v}"
                        for n, v in {**_ADDED_BY_INSTALL, **self.install_moves}.items()
                    ),
                )
            elif any(a.startswith("flashinfer-jit-cache==") for a in cmd):
                spec = next(a for a in cmd if a.startswith("flashinfer-jit-cache=="))
                self.dists["flashinfer-jit-cache"] = spec.split("==", 1)[1]
                return _Result(0, f"Installed 1 package in 3.1s\n + {spec}")
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
def _clean(monkeypatch, tmp_path):
    monkeypatch.setattr(inst, "_env_lock_path", lambda: str(tmp_path / "install.lock"))
    monkeypatch.delenv(inst.FLASHINFER_INSTALL_ENV, raising = False)
    monkeypatch.delenv(ops.NVFP4_BACKEND_ENV, raising = False)
    for name in (
        *inst._CUSTOM_INDEX_ENVS,
        "UV_INDEX",
        "UV_EXTRA_INDEX_URL",
        *inst._PIP_EXTRA_INDEX_ENVS,
        "UV_OFFLINE",
        "UV_CONFIG_FILE",
        "ALL_PROXY",
        "all_proxy",
    ):
        monkeypatch.delenv(name, raising = False)
    # The host's own uv.toml and pip.conf never leak in: the fake run answers no `pip config list`.
    monkeypatch.setenv("UV_NO_CONFIG", "1")
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


def _steps(installs):
    """``(main, jit)`` by content, whichever order the transaction runs them in."""
    main = next(c for c in installs if "flashinfer-python==0.6.6" in c)
    jit = next(c for c in installs if any(a.startswith("flashinfer-jit-cache==") for a in c))
    return main, jit


def _ensure(fake, device = 0):
    return inst.ensure_flashinfer_for_nvfp4(device, run = fake.run)


def test_installs_pinned_flashinfer_and_matching_jit_cache_when_missing(env):
    ok, reason = _ensure(env)
    assert ok, reason
    installs = env.installs()
    assert len(installs) == 2
    main, jit = _steps(installs)
    assert "flashinfer-python==0.6.6" in main and "--only-binary" in main
    assert "--no-deps" not in main  # tvm-ffi and friends are real import deps
    # The full local version: `==0.6.6` would be satisfied by a cache built for another CUDA.
    assert "flashinfer-jit-cache==0.6.6+cu130" in jit and "--no-deps" in jit
    assert jit[jit.index("--index-url") + 1] == "https://flashinfer.ai/whl/cu130"
    assert jit[jit.index("--index") + 1] == "https://flashinfer.ai/whl/cu130"
    assert env.dists["flashinfer-jit-cache"] == "0.6.6+cu130"
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


def test_rollback_leaves_packages_another_installer_added_meanwhile(env):
    # The built-in terminal does not take the env lock; what it installed during this transaction must survive.
    env.fail_verify = True
    env.concurrent_add = {"requests": "2.32.3", "rich": "14.0.0"}
    ok, reason = _ensure(env)
    assert not ok and "rolled back" in reason
    for name in _ADDED_BY_INSTALL:
        assert name not in env.dists
    assert env.dists.get("requests") == "2.32.3" and env.dists.get("rich") == "14.0.0", (
        reason,
        env.commands,
    )
    uninstall = next(c for c in env.commands if "uninstall" in c)
    assert "requests" not in uninstall and "rich" not in uninstall


def test_reported_installs_reads_uv_and_pip_output():
    uv_out = "Resolved 5 packages\nInstalled 2 packages in 9ms\n + Apache_TVM_FFI==0.1.9\n + flashinfer-python==0.6.6\n ~ numpy==2.3.5"
    assert inst._reported_installs(uv_out) == {"apache-tvm-ffi", "flashinfer-python"}
    pip_out = (
        "Collecting x\nSuccessfully installed apache-tvm-ffi-0.1.9 flashinfer-jit-cache-0.6.6+cu130"
    )
    assert inst._reported_installs(pip_out) == {"apache-tvm-ffi", "flashinfer-jit-cache"}
    assert inst._reported_installs("ERROR: no matching distribution") == set()


def test_rollback_without_installer_output_follows_the_flashinfer_dependency_tree(monkeypatch):
    # A timed-out install reports nothing: fall back to flashinfer's installed requirement tree, never the whole diff.
    import importlib.metadata as md

    requires = {
        "flashinfer-python": [
            "apache-tvm-ffi>=0.1",
            "nvidia-cutlass-dsl; python_version >= '3'",
            "pytest; extra == 'test'",
        ],
        "apache-tvm-ffi": [],
        "nvidia-cutlass-dsl": [],
    }

    def _distribution(name):
        if name not in requires:
            raise md.PackageNotFoundError(name)
        return types.SimpleNamespace(requires = requires[name])

    monkeypatch.setattr(md, "distribution", _distribution)
    before = dict(_BASE_DISTS)
    after = {**before, **_ADDED_BY_INSTALL, "pytest": "8.0.0", "requests": "2.32.3"}
    monkeypatch.setattr(inst, "installed_distributions", lambda: dict(after))
    commands = []

    def run(cmd, **kwargs):
        commands.append([str(c) for c in cmd])
        return _Result(0)

    inst._rollback(run, "uv", before, None, reported = set())
    assert len(commands) == 1
    removed = set(commands[0][commands[0].index(sys.executable) + 1 :])
    assert removed == set(_ADDED_BY_INSTALL)


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


def test_another_process_holding_the_env_lock_blocks_the_install(env, monkeypatch):
    filelock = pytest.importorskip("filelock")
    monkeypatch.setattr(inst, "ENV_LOCK_TIMEOUT_S", 0.05)
    real = inst._env_install_lock
    monkeypatch.setattr(inst, "_env_install_lock", lambda timeout = 0.05: real(timeout))
    # A second FileLock on its own file handle, held from a thread, stands in for another process.
    other = filelock.FileLock(inst._env_lock_path(), thread_local = False)
    import threading

    held, release = threading.Event(), threading.Event()

    def _hold():
        with other:
            held.set()
            release.wait(5)

    t = threading.Thread(target = _hold)
    t.start()
    held.wait(5)
    try:
        ok, reason = _ensure(env)
    finally:
        release.set()
        t.join()
    assert not ok and "another process" in reason
    assert env.commands == []
    # Not memoised: once the other install is done, this process installs.
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
    teardown = body.index("self._teardown_state_locked()")
    # The MiniMax-H3 modular dispatch installs inline; a conventional load through the seed hop.
    modular = body.index("ensure_flashinfer_for_nvfp4(")
    conventional = body.index("self._install_flashinfer_for_seed(")
    assert body.count("ensure_flashinfer_for_nvfp4(") == 1
    for call, indent in ((modular, 16), (conventional, 8)):
        assert teardown < call
        # Only the modular branch's own `if` blocks above it: not nested in any with-block.
        line = body[body.rindex("\n", 0, call) + 1 : call]
        assert line == " " * indent + "_nvfp4_install_outcome = "
    assert _flat(
        '"nvfp4" in (normalize_transformer_quant(transformer_quant), _video_auto_denoiser_planned)'
    ) in _flat(body)


def test_video_loader_installs_only_for_a_seed_the_live_memory_plan_kept():
    # The prefetch settles the seed against CAPACITY; the load re-plans against live free memory and can drop it
    # (it would offload). Installing before that re-plan bought ~1.5 GB of FlashInfer for a bf16 denoiser.
    from core.inference.video import VideoBackend

    body = _load_pipeline_body(_INFERENCE_DIR / "video.py")
    conventional = body.index("self._install_flashinfer_for_seed(")
    injection = body.index("denoiser_prequant_pipe_kwargs(")
    # After every memory-plan decision that can drop the seed, before the seeded denoiser is built.
    assert body.rindex("denoiser_seed_scheme = None", 0, injection) < conventional < injection
    assert _flat("_nvfp4_install_wanted denoiser_seed_scheme device") in _flat(
        body[conventional : body.index(")", conventional)]
    )
    modular = body.index("ensure_flashinfer_for_nvfp4(")
    assert (
        body.index("if fam.modular_workflow:")
        < modular
        < body.index("self._load_h3_modular_pipeline(")
    )

    calls = []
    import core.inference.diffusion_nvfp4_install as inst_mod

    original = inst_mod.ensure_flashinfer_for_nvfp4
    inst_mod.ensure_flashinfer_for_nvfp4 = lambda device, **kw: calls.append(device) or (True, "ok")
    try:
        assert VideoBackend._install_flashinfer_for_seed(True, None, "cuda") is None
        assert VideoBackend._install_flashinfer_for_seed(True, "int8", "cuda") is None
        assert VideoBackend._install_flashinfer_for_seed(False, "nvfp4", "cuda") is None
        assert calls == []
        assert VideoBackend._install_flashinfer_for_seed(True, "nvfp4", "cuda") == (True, "ok")
        assert calls == ["cuda"]
    finally:
        inst_mod.ensure_flashinfer_for_nvfp4 = original


def test_image_loader_gates_the_install_on_the_seed_plan_it_will_rerun():
    import inspect

    from core.inference import diffusion

    body = _load_pipeline_body(_INFERENCE_DIR / "diffusion.py")
    call = body.index("ensure_flashinfer_for_nvfp4(")
    assert "self._seed_plan_stays_resident(" in body[:call]
    # The locked re-plan and the pre-install gate are one plan, so they cannot disagree on what the seed costs.
    locked = body[body.index("with self._lock:") :]
    assert "self._seeded_pipeline_plan(" in locked
    helper = inspect.getsource(diffusion.DiffusionBackend._seed_plan_stays_resident)
    assert "self._seeded_pipeline_plan(" in helper


class _FakePlan:
    def __init__(self, offload_policy):
        self.offload_policy = offload_policy


def _seed_gate(
    monkeypatch,
    *,
    free_mib,
    reserved_mib,
    need_mib,
    total_mib = 100_000,
):
    from core.inference import diffusion
    from core.inference.diffusion_memory import DeviceMemory

    seen = []
    monkeypatch.setattr(
        diffusion,
        "snapshot_device_memory",
        lambda target: DeviceMemory("cuda", "cuda", "discrete_vram", free_mib, total_mib),
    )
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(memory_reserved = lambda: reserved_mib * 1024 * 1024)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(
        diffusion,
        "estimate_dense_quant",
        lambda fam, scheme, base_repo = None, prequant_available = False: types.SimpleNamespace(
            steady_transformer_mib = need_mib, companions_mib = 0, text_encoders_mib = 0
        ),
    )

    def _plan_memory(self, target, single_file_path, base, fam, memory_mode, cpu_offload, **kw):
        seen.append(kw)
        memory = kw["device_memory_override"]
        return _FakePlan("none" if memory.free_mib >= need_mib else "model")

    monkeypatch.setattr(diffusion.DiffusionBackend, "_plan_memory", _plan_memory)
    backend = object.__new__(diffusion.DiffusionBackend)
    ok = backend._seed_plan_stays_resident(
        "nvfp4",
        types.SimpleNamespace(device = "cuda"),
        "org/base",
        object(),
        None,
        False,
        repo_id = "org/base",
        base_local_dir = None,
        fetch_base = None,
    )
    return ok, seen


def test_seed_gate_refuses_when_live_memory_would_offload_the_seed(monkeypatch):
    # Total capacity fits (the prefetch said NVFP4) but a foreign tenant leaves too little free: no install.
    ok, seen = _seed_gate(monkeypatch, free_mib = 10_000, reserved_mib = 0, need_mib = 40_000)
    assert ok is False
    assert seen and seen[0]["transformer_resident_override_mib"] == 40_000


def test_seed_gate_credits_what_the_teardown_frees(monkeypatch):
    # The resident pipeline is this process's allocation and the teardown frees it before the loader's re-plan.
    ok, _ = _seed_gate(monkeypatch, free_mib = 10_000, reserved_mib = 35_000, need_mib = 40_000)
    assert ok is True


def test_seed_gate_unanswerable_keeps_the_old_answer(monkeypatch):
    from core.inference import diffusion

    monkeypatch.setattr(
        diffusion,
        "snapshot_device_memory",
        lambda target: (_ for _ in ()).throw(RuntimeError("probe")),
    )
    backend = object.__new__(diffusion.DiffusionBackend)
    assert (
        backend._seed_plan_stays_resident(
            "nvfp4",
            types.SimpleNamespace(device = "cuda"),
            "b",
            object(),
            None,
            False,
            repo_id = "b",
            base_local_dir = None,
            fetch_base = None,
        )
        is True
    )


def test_a_configured_mirror_skips_the_pypi_probe_but_not_the_jit_cache_one(env, monkeypatch):
    probed = []
    monkeypatch.setattr(inst, "_reachable", lambda url: probed.append(url) or True)
    monkeypatch.setenv("UV_INDEX_URL", "https://mirror.example/simple")
    assert _ensure(env)[0]
    assert probed == ["https://flashinfer.ai/whl/cu130/flashinfer-jit-cache/"]


def test_a_pip_only_mirror_is_handed_to_uv(env, monkeypatch):
    monkeypatch.setattr(inst, "_reachable", lambda url: True)
    monkeypatch.setenv("PIP_INDEX_URL", "https://pip-mirror.example/simple")
    assert _ensure(env)[0]
    installs = [c for c in env.commands if c[:3] == ["uv", "pip", "install"]]
    assert installs and "https://pip-mirror.example/simple" in _steps(installs)[0]
    # uv's own setting wins and is left to uv.
    monkeypatch.setenv("UV_INDEX_URL", "https://uv-mirror.example/simple")
    assert "--index-url" not in inst._installer_prefix("uv")
    assert "--index-url" not in inst._installer_prefix(None)


def test_a_uv_only_mirror_is_handed_to_the_pip_fallback(env, monkeypatch):
    # No uv: pip does not read UV_* settings, and the preflight skipped the pypi.org probe for them.
    monkeypatch.setattr(inst, "_uv_executable", lambda: None)
    for name in ("PIP_INDEX_URL", "UV_DEFAULT_INDEX"):
        monkeypatch.delenv(name, raising = False)
    monkeypatch.setenv("UV_INDEX_URL", "https://uv-mirror.example/simple")
    main = inst._installer_prefix(None)
    assert main[main.index("--index-url") + 1] == "https://uv-mirror.example/simple"
    monkeypatch.setenv("UV_DEFAULT_INDEX", "https://uv-default.example/simple")
    main = inst._installer_prefix(None)
    assert main[main.index("--index-url") + 1] == "https://uv-default.example/simple"
    # The jit-cache step keeps its own index only.
    jit = inst._installer_prefix(None, "https://flashinfer.ai/whl/cu130")
    assert jit.count("--index-url") == 1 and jit[-1] == "https://flashinfer.ai/whl/cu130"
    # pip's own setting wins and is left to pip.
    monkeypatch.setenv("PIP_INDEX_URL", "https://pip-mirror.example/simple")
    assert "--index-url" not in inst._installer_prefix(None)


def test_a_pip_only_mirror_never_gives_uv_two_index_urls(env, monkeypatch):
    # uv rejects a repeated --index-url; the jit-cache step's own index replaces the mirror.
    monkeypatch.setenv("PIP_INDEX_URL", "https://pip-mirror.example/simple")
    assert _ensure(env)[0]
    main, jit = _steps(env.installs())
    assert main.count("--index-url") == 1
    assert main[main.index("--index-url") + 1] == "https://pip-mirror.example/simple"
    assert jit.count("--index-url") == 1
    assert jit[jit.index("--index-url") + 1] == "https://flashinfer.ai/whl/cu130"
    pip_jit = inst._installer_prefix(None, "https://flashinfer.ai/whl/cu130")
    assert pip_jit.count("--index-url") == 1


def test_status_reason_reports_a_transient_preflight_failure(monkeypatch):
    # An OOM preflight is not memoised, but the model it put on torchao still reports why.
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device = None: (10, 0))
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device = None: "stub B200")
    ops.reset_preflight_cache()

    def _oom(dev):
        raise torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 64.00 MiB")

    class _Backend:
        pass

    image = _Backend()
    inst.record_install_reason(image, True, "already installed", 0)
    monkeypatch.setattr(ops, "_preflight_probe", _oom)
    assert ops.nvfp4_preflight(0)["ok"] is False
    assert 0 not in ops._PREFLIGHT
    reason = inst.nvfp4_backend_fields("torchao", owner = image)["transformer_quant_backend_reason"]
    assert reason and "out of memory" in reason.lower()
    assert (
        "out of memory"
        in inst.nvfp4_backend_fields("torchao")["transformer_quant_backend_reason"].lower()
    )
    # A later memoised preflight on the same device supersedes it.
    monkeypatch.setattr(ops, "_preflight_probe", lambda dev: False)
    ops.nvfp4_preflight(0)
    assert (
        inst.nvfp4_backend_fields("torchao", owner = image)["transformer_quant_backend_reason"]
        == "flashinfer preflight failed: mm_fp4 produced a non-finite result"
    )
    ops.reset_preflight_cache()


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


def test_image_loader_skips_the_install_when_the_plan_declined_the_seed():
    # A declined seed loads the released denoiser, and the Hub probe alone would still buy the install.
    body = _load_pipeline_body(_INFERENCE_DIR / "diffusion.py")
    gate = body[: body.index("ensure_flashinfer_for_nvfp4(")].rsplit("if (", 1)[-1]
    declined = gate.index("_pipeline_prequant_planned != PIPELINE_SEED_DECLINED")
    assert declined < gate.index("self._nvfp4_checkpoint_will_load(")


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


def test_video_loader_binds_the_reason_only_at_the_commit():
    # A superseded video load can return from the install after a newer load committed; only the commit may bind.
    import inspect

    from core.inference import video

    src = inspect.getsource(video.VideoBackend)
    ensure_at = src.index("ensure_flashinfer_for_nvfp4(")
    call = src[ensure_at : src.index(")", ensure_at)]
    assert "owner" not in call
    commits = [m.start() for m in re.finditer(r"self\._state = _VideoLoadState\(", src)]
    binds = [m.start() for m in re.finditer(r"record_install_reason\(self", src)]
    assert len(binds) == 2
    for bind in binds:
        commit = min(c for c in commits if c > bind)
        cancel = src.rindex("Video load was cancelled or superseded.", 0, bind)
        assert src.index("self._state = _VideoLoadState(", cancel) == commit


def test_every_loader_commit_binds_a_reason_even_when_the_install_gate_skipped():
    # A skipped gate (no hosted checkpoint) still replaces the previous model's reason at the commit.
    import inspect

    from core.inference import diffusion, video
    for cls in (diffusion.DiffusionBackend, video.VideoBackend):
        src = inspect.getsource(cls)
        assert "if _nvfp4_install_outcome is not None" not in src
        binds = re.findall(
            r"record_install_reason\(self, \*\(_nvfp4_install_outcome or \(True, None\)\)", src
        )
        assert len(binds) == (1 if cls is diffusion.DiffusionBackend else 2)


def test_install_gates_skip_kinds_the_dense_quant_path_cannot_reach():
    import inspect

    from core.inference import diffusion, video

    img = inspect.getsource(diffusion.DiffusionBackend)
    gate = img[: img.index("ensure_flashinfer_for_nvfp4(")].rsplit("if ", 1)[-1]
    assert gate.lstrip("( \n").startswith("dense_quant_supported_kind(kind)")
    vid = inspect.getsource(video.VideoBackend)
    gate = vid[vid.index("_nvfp4_install_wanted = (") + len("_nvfp4_install_wanted = (") :]
    assert gate.lstrip("( \n").startswith('kind == "pipeline"')
    # Both video installs are behind that gate.
    body = _load_pipeline_body(_INFERENCE_DIR / "video.py")
    modular = body.index("ensure_flashinfer_for_nvfp4(")
    assert "if _nvfp4_install_wanted:" in body[body.index("if fam.modular_workflow:") : modular]
    hop = body[body.index("self._install_flashinfer_for_seed(") :]
    assert hop[: hop.index(")")].split("(", 1)[1].split(",")[0].strip() == "_nvfp4_install_wanted"


def test_jit_cache_step_drops_extra_index_sources_from_the_environment(env, monkeypatch):
    # An extra index outranks uv's --index-url; under first-index one listing flashinfer-jit-cache hides the pin.
    extra = {
        "UV_INDEX": "https://corp.example/simple",
        "UV_EXTRA_INDEX_URL": "https://corp.example/extra",
        "PIP_EXTRA_INDEX_URL": "https://corp.example/pip-extra",
        "UV_FIND_LINKS": "https://corp.example/links",
    }
    for name, value in extra.items():
        monkeypatch.setenv(name, value)
    seen = []
    real = env.run

    def _run(cmd, **kwargs):
        seen.append(([str(c) for c in cmd], kwargs.get("env")))
        return real(cmd, **kwargs)

    assert inst.ensure_flashinfer_for_nvfp4(0, run = _run)[0]
    envs = {
        "main" if any(c.startswith("flashinfer-python==") for c in cmd) else "jit": child
        for cmd, child in seen
        if "install" in cmd and "uninstall" not in cmd
    }
    for name, value in extra.items():
        assert envs["main"].get(name) == value  # the user's mirrors still serve the main step
        assert name not in envs["jit"]


def test_pip_jit_cache_step_reads_no_pip_conf_but_keeps_its_transport(env, monkeypatch):
    # pip searches a pip.conf extra-index-url / find-links beside --index-url, so a corporate or stale
    # flashinfer-jit-cache there could be picked over the pinned index. The main step keeps pip.conf as designed.
    monkeypatch.setattr(inst, "_uv_executable", lambda: None)
    monkeypatch.setenv("PIP_TIMEOUT", "30")
    lines = [
        "global.extra-index-url='https://corp.example/extra'",
        "install.find-links='https://corp.example/links'",
        "global.proxy='http://proxy.corp.example:3128'",
        "global.cert='/etc/ssl/corp.pem'",
        "global.trusted-host='a.corp.example\\nb.corp.example'",
        "global.timeout='120'",
        ":env:.timeout='30'",
    ]
    seen = []
    config_run = _pip_config(env, lines)

    def _run(cmd, **kwargs):
        seen.append(([str(c) for c in cmd], kwargs.get("env")))
        return config_run(cmd, **kwargs)

    ok, reason = inst.ensure_flashinfer_for_nvfp4(0, run = _run)
    assert ok, reason
    installs = {
        "main" if any(c.startswith("flashinfer-python==") for c in cmd) else "jit": (cmd, child)
        for cmd, child in seen
        if "install" in cmd and "uninstall" not in cmd and "config" not in cmd
    }
    main_cmd, main_env = installs["main"]
    jit_cmd, jit_env = installs["jit"]
    assert main_env is None or main_env.get("PIP_CONFIG_FILE") != os.devnull
    assert jit_env["PIP_CONFIG_FILE"] == os.devnull
    assert jit_cmd[jit_cmd.index("--index-url") + 1] == "https://flashinfer.ai/whl/cu130"
    assert jit_env["PIP_PROXY"] == "http://proxy.corp.example:3128"
    assert jit_env["PIP_CERT"] == "/etc/ssl/corp.pem"
    assert jit_env["PIP_TRUSTED_HOST"].split() == ["a.corp.example", "b.corp.example"]
    assert jit_env["PIP_TIMEOUT"] == "30"  # the caller's own environment still wins
    assert not any(k in jit_env for k in ("PIP_EXTRA_INDEX_URL", "PIP_FIND_LINKS"))


def test_uv_jit_cache_step_leaves_pip_config_alone(env, monkeypatch):
    seen = []
    real = env.run

    def _run(cmd, **kwargs):
        seen.append(([str(c) for c in cmd], kwargs.get("env")))
        return real(cmd, **kwargs)

    assert inst.ensure_flashinfer_for_nvfp4(0, run = _run)[0]
    jit_env = next(
        child for cmd, child in seen if any(c.startswith("flashinfer-jit-cache==") for c in cmd)
    )
    assert jit_env.get("PIP_CONFIG_FILE") == os.environ.get("PIP_CONFIG_FILE")


def _hold_env_lock(release_after):
    filelock = pytest.importorskip("filelock")
    other = filelock.FileLock(inst._env_lock_path(), thread_local = False)
    held = threading.Event()

    def _hold():
        with other:
            held.set()
            time.sleep(release_after)

    t = threading.Thread(target = _hold)
    t.start()
    held.wait(5)
    return t


def _track_imports(env, monkeypatch):
    stamps = []
    monkeypatch.setattr(
        inst,
        "_import_flashinfer",
        lambda: stamps.append(time.monotonic())
        or ((True, "0.6.6") if env.importable else (False, "ModuleNotFoundError: flashinfer")),
    )
    monkeypatch.delitem(sys.modules, "flashinfer", raising = False)
    return stamps


def test_an_in_flight_install_by_another_process_is_waited_for_before_importing(env, monkeypatch):
    # Another process has installed flashinfer-python and is still downloading the jit-cache: importing now would
    # fix flashinfer's cache directory without it for this process's whole life.
    env.dists["flashinfer-python"] = "0.6.6"
    env.importable = True
    stamps = _track_imports(env, monkeypatch)
    start = time.monotonic()
    t = _hold_env_lock(0.5)
    try:
        ok, reason = _ensure(env)
    finally:
        t.join()
    assert ok and "already installed" in reason
    assert stamps and stamps[0] - start >= 0.45
    assert env.commands == []


def test_a_complete_install_still_holding_the_lock_is_waited_for_in_case_it_rolls_back(
    env, monkeypatch
):
    # Both distributions are on disk, but the other process is still in its drift checks and import verify: it
    # rolls back here, so importing before it releases the lock would load a flashinfer that is then uninstalled.
    env.dists.update({"flashinfer-python": "0.6.6", "flashinfer-jit-cache": "0.6.6+cu130"})
    env.importable = True
    stamps = _track_imports(env, monkeypatch)
    filelock = pytest.importorskip("filelock")
    other = filelock.FileLock(inst._env_lock_path(), thread_local = False)
    held, rolled_back = threading.Event(), []

    def _hold():
        with other:
            held.set()
            time.sleep(0.5)
            env.dists.pop("flashinfer-python")
            env.dists.pop("flashinfer-jit-cache")
            env.importable = False
            # Imports seen before the rollback: ordering by count, not by clock.
            rolled_back.append(len(stamps))

    t = threading.Thread(target = _hold)
    t.start()
    held.wait(5)
    try:
        ok, reason = _ensure(env)
    finally:
        t.join()
    assert rolled_back == [0] and stamps
    assert "already installed" not in reason


def test_a_user_flashinfer_never_waits_on_the_lock(env, monkeypatch):
    env.dists["flashinfer-python"] = "0.5.3"
    env.importable = True
    stamps = _track_imports(env, monkeypatch)
    start = time.monotonic()
    t = _hold_env_lock(1.0)
    try:
        ok, _ = _ensure(env)
    finally:
        t.join()
    assert ok and stamps[0] - start < 0.5
    assert env.commands == []


def test_a_timed_out_wait_for_an_in_flight_install_does_not_import(env, monkeypatch):
    # The other process's install outlives the wait: importing now would pin flashinfer's cache directory
    # without the jit-cache, so this load reports not ready and a later one retries.
    env.dists["flashinfer-python"] = "0.6.6"
    env.importable = True
    real = inst._env_install_lock
    monkeypatch.setattr(inst, "_env_install_lock", lambda timeout = 0.05: real(timeout))
    stamps = _track_imports(env, monkeypatch)
    t = _hold_env_lock(0.5)
    try:
        ok, reason = _ensure(env)
    finally:
        t.join()
    assert not ok and "another process" in reason
    assert stamps == [] and env.commands == []
    ok, reason = _ensure(env)
    assert ok and "already installed" in reason


@pytest.mark.parametrize("uv", ["uv", None])
@pytest.mark.parametrize("name", ["UV_INDEX", "UV_EXTRA_INDEX_URL", "UV_FIND_LINKS"])
def test_a_uv_index_skips_the_pypi_probe_only_when_uv_installs(env, monkeypatch, name, uv):
    probed = []
    monkeypatch.setattr(inst, "_reachable", lambda url: probed.append(url) or True)
    monkeypatch.setattr(inst, "_uv_executable", lambda: uv)
    monkeypatch.setenv(name, "https://uv-mirror.example/simple")
    assert _ensure(env)[0]
    # uv ranks these above its default index; pip does not read them, so its fallback still needs pypi.org.
    assert (inst._PYPI_PROBE_URL in probed) == (uv is None)


@pytest.mark.parametrize(
    "cuda, cache, nvcc",
    [
        ("13.0", "0.6.6+cu128", False),  # left over from a cu128 torch
        ("13.0", "0.6.6", False),  # no local tag: the CUDA it was built for cannot be told
        ("14.0", "0.6.6+cu130", True),  # no published build fits, yet a cache is present
    ],
)
def test_a_jit_cache_built_for_another_cuda_refuses(env, monkeypatch, cuda, cache, nvcc):
    # flashinfer only checks the public version and `==0.6.6` accepts any local tag, so the installer would keep it.
    env.cuda = cuda
    env.dists["flashinfer-jit-cache"] = cache
    monkeypatch.setattr(inst, "_nvcc_available", lambda: nvcc)
    ok, reason = _ensure(env)
    assert not ok and f"flashinfer-jit-cache {cache}" in reason and "does not match CUDA" in reason
    assert env.installs() == []


def test_a_jit_cache_of_the_running_cuda_is_kept(env):
    env.dists["flashinfer-jit-cache"] = "0.6.6+cu130"
    ok, reason = _ensure(env)
    assert ok, reason


@pytest.mark.parametrize("policy", ["local_files_only", "opt_out"])
def test_a_load_that_declined_installs_does_not_wait_on_another_install(env, monkeypatch, policy):
    # Another process has flashinfer-python on disk and is still downloading the jit-cache. A local-only load, or a
    # host with installs turned off, used to sit on the lock for up to ENV_LOCK_TIMEOUT_S before its own checks ran.
    env.dists["flashinfer-python"] = "0.6.6"
    env.importable = True
    stamps = _track_imports(env, monkeypatch)
    kwargs = {}
    if policy == "local_files_only":
        kwargs["local_files_only"] = True
    else:
        monkeypatch.setenv(inst.FLASHINFER_INSTALL_ENV, "0")
    t = _hold_env_lock(2.0)
    start = time.monotonic()
    try:
        ok, reason = inst.ensure_flashinfer_for_nvfp4(0, run = env.run, **kwargs)
        elapsed = time.monotonic() - start
    finally:
        t.join()
    assert elapsed < 1.0
    assert not ok and "another process is still installing" in reason
    assert stamps == []  # never imported mid-transaction
    assert env.commands == []


def test_a_load_that_declined_installs_still_imports_a_finished_install(env, monkeypatch):
    env.dists.update({"flashinfer-python": "0.6.6", "flashinfer-jit-cache": "0.6.6+cu130"})
    env.importable = True
    ok, reason = inst.ensure_flashinfer_for_nvfp4(0, run = env.run, local_files_only = True)
    assert ok and "already installed" in reason


def _pip_config(env, lines):
    """``env.run`` that also answers ``pip config list`` with ``lines``, as pip prints them."""
    calls = []

    def _run(cmd, **kwargs):
        cmd = [str(c) for c in cmd]
        if cmd[1:] == ["-m", "pip", "config", "list"]:
            calls.append(cmd)
            return _Result(0, "".join(f"{line}\n" for line in lines))
        return env.run(cmd, **kwargs)

    _run.calls = calls
    return _run


def test_a_pip_conf_mirror_skips_the_pypi_probe_for_the_pip_fallback(env, monkeypatch):
    probed = []
    monkeypatch.setattr(inst, "_reachable", lambda url: probed.append(url) or True)
    monkeypatch.setattr(inst, "_uv_executable", lambda: None)
    run = _pip_config(env, ["global.index-url='https://corp.example/simple'"])
    ok, reason = inst.ensure_flashinfer_for_nvfp4(0, run = run)
    assert ok, reason
    assert probed == ["https://flashinfer.ai/whl/cu130/flashinfer-jit-cache/"]


@pytest.mark.parametrize(
    "lines, envvar",
    [
        ([":env:.extra-index-url='https://corp.example/simple'"], None),
        (["global.extra-index-url='https://corp.example/simple'"], None),
        (["install.find-links='https://corp.example/links'"], None),
        ([], "PIP_EXTRA_INDEX_URL"),
        ([], "PIP_FIND_LINKS"),
    ],
)
def test_a_pip_extra_index_skips_the_pypi_probe_for_the_pip_fallback(
    env, monkeypatch, lines, envvar
):
    # pip falls through an unreachable pypi.org to the extra index, so a blocked pypi.org must not refuse the install.
    probed = []
    monkeypatch.setattr(
        inst, "_reachable", lambda url: probed.append(url) or url != inst._PYPI_PROBE_URL
    )
    monkeypatch.setattr(inst, "_uv_executable", lambda: None)
    if envvar:
        monkeypatch.setenv(envvar, "https://corp.example/simple")
    ok, reason = inst.ensure_flashinfer_for_nvfp4(0, run = _pip_config(env, lines))
    assert ok, reason
    assert inst._PYPI_PROBE_URL not in probed


def test_a_pip_conf_mirror_is_not_overridden_by_a_uv_one(env, monkeypatch):
    monkeypatch.setattr(inst, "_uv_executable", lambda: None)
    monkeypatch.setenv("UV_INDEX_URL", "https://uv-mirror.example/simple")
    run = _pip_config(env, ["install.index-url='https://corp.example/simple'"])
    assert inst.ensure_flashinfer_for_nvfp4(0, run = run)[0]
    main = _steps(env.installs())[0]
    assert "--index-url" not in main  # pip reads its own pip.conf mirror


def test_pip_config_is_only_read_when_pip_installs(env):
    run = _pip_config(env, [":env:.no-index='1'"])
    assert inst.ensure_flashinfer_for_nvfp4(0, run = run)[0]
    assert (
        run.calls == []
    )  # uv ignores pip's settings, and the installed path never pays for the subprocess


@pytest.mark.parametrize(
    "line, needle",
    [
        (":env:.no-index='1'", "no-index"),
        ("global.no-index='true'", "no-index"),
        (":env:.no-deps='yes'", "no-deps"),
        ("install.target='/opt/elsewhere'", "target directory"),
        (":env:.prefix='/opt/elsewhere'", "prefix"),
        ("install.root='/chroot'", "another root"),
        (":env:.force-reinstall='1'", "force-reinstall"),
        ("install.force-reinstall='true'", "force-reinstall"),
        (":env:.ignore-installed='yes'", "ignore installed"),
        ("global.ignore-installed='true'", "ignore installed"),
    ],
)
def test_pip_settings_it_cannot_honor_refuse_before_installing(env, monkeypatch, line, needle):
    probed = []
    monkeypatch.setattr(inst, "_reachable", lambda url: probed.append(url) or True)
    monkeypatch.setattr(inst, "_uv_executable", lambda: None)
    ok, reason = inst.ensure_flashinfer_for_nvfp4(0, run = _pip_config(env, [line]))
    assert not ok and needle in reason
    assert env.installs() == [] and probed == []
    # Not memoised: once the setting is gone, the same process installs.
    assert inst.ensure_flashinfer_for_nvfp4(0, run = _pip_config(env, []))[0]


def test_pip_require_virtualenv_refuses_only_outside_a_venv(env, monkeypatch):
    monkeypatch.setattr(inst, "_uv_executable", lambda: None)
    run = _pip_config(env, [":env:.require-virtualenv='1'"])
    monkeypatch.setattr(sys, "base_prefix", sys.prefix)
    ok, reason = inst.ensure_flashinfer_for_nvfp4(0, run = run)
    assert not ok and "virtualenv" in reason and env.installs() == []
    monkeypatch.setattr(sys, "base_prefix", sys.prefix + "-base")
    assert inst.ensure_flashinfer_for_nvfp4(0, run = run)[0]


def _uv_toml(monkeypatch, tmp_path, text):
    path = tmp_path / "uv.toml"
    path.write_text(text, encoding = "utf-8")
    monkeypatch.setenv("UV_CONFIG_FILE", str(path))
    return path


@pytest.mark.parametrize(
    "text, needle",
    [
        ("offline = true\n", "offline"),
        ("no-index = true\n", "no-index"),
        ("[pip]\nno-index = true\n", "no-index"),
        ("[pip]\nno-deps = true\n", "no-deps"),
        ('[pip]\ntarget = "/opt/elsewhere"\n', "target directory"),
        ('[pip]\nprefix = "/opt/elsewhere"\n', "prefix"),
        ("reinstall = true\n", "reinstall"),
        ("[pip]\nreinstall = true\n", "reinstall"),
        ('[pip]\nreinstall-package = ["torch"]\n', "reinstall"),
    ],
)
def test_uv_config_settings_it_cannot_honor_refuse_before_installing(
    env, monkeypatch, tmp_path, text, needle
):
    probed = []
    monkeypatch.setattr(inst, "_reachable", lambda url: probed.append(url) or True)
    _uv_toml(monkeypatch, tmp_path, text)
    ok, reason = _ensure(env)
    assert not ok and needle in reason
    assert env.installs() == [] and probed == []
    monkeypatch.delenv("UV_CONFIG_FILE")
    assert _ensure(env)[0]


def test_uv_offline_env_false_overrides_a_config_offline(env, monkeypatch, tmp_path):
    _uv_toml(monkeypatch, tmp_path, "offline = true\n")
    monkeypatch.setenv("UV_OFFLINE", "0")
    assert _ensure(env)[0]


def test_pip_section_overrides_the_top_level_in_uv_config(env, monkeypatch, tmp_path):
    _uv_toml(monkeypatch, tmp_path, "no-index = true\n[pip]\nno-index = false\n")
    assert _ensure(env)[0]


@pytest.mark.parametrize(
    "text",
    [
        'index-url = "https://corp.example/simple"\n',
        '[[index]]\nurl = "https://corp.example/simple"\n',
        '[pip]\nextra-index-url = ["https://corp.example/simple"]\n',
        "this is not toml = = \n",  # unreadable here: the installer decides, not pypi.org
    ],
)
def test_a_uv_config_mirror_skips_the_pypi_probe(env, monkeypatch, tmp_path, text):
    probed = []
    monkeypatch.setattr(inst, "_reachable", lambda url: probed.append(url) or True)
    _uv_toml(monkeypatch, tmp_path, text)
    assert _ensure(env)[0]
    assert probed == ["https://flashinfer.ai/whl/cu130/flashinfer-jit-cache/"]


def test_uv_config_discovery_finds_the_project_file_and_honors_no_config(
    env, monkeypatch, tmp_path
):
    project = tmp_path / "proj" / "sub"
    project.mkdir(parents = True)
    (tmp_path / "proj" / "pyproject.toml").write_text(
        '[project]\nname = "x"\n[tool.uv]\nindex-url = "https://corp.example/simple"\n',
        encoding = "utf-8",
    )
    # A pyproject.toml without [tool.uv] does not stop the search, as in uv.
    (project / "pyproject.toml").write_text('[project]\nname = "y"\n', encoding = "utf-8")
    monkeypatch.chdir(project)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.setenv("XDG_CONFIG_DIRS", str(tmp_path / "xdg-sys"))
    monkeypatch.delenv("UV_NO_CONFIG")
    config = inst._installer_config("uv", env.run)
    assert config["mirror"] and config["own_index"]
    # uv's own default index wins over a pip-only mirror, as its env settings already do.
    monkeypatch.setenv("PIP_INDEX_URL", "https://pip-mirror.example/simple")
    assert "--index-url" not in inst._installer_prefix("uv", own_index = config["own_index"])
    monkeypatch.setenv("UV_NO_CONFIG", "1")
    assert not inst._installer_config("uv", env.run)["own_index"]
    # A user-level uv.toml is read too.
    monkeypatch.delenv("UV_NO_CONFIG")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "xdg" / "uv").mkdir(parents = True)
    (tmp_path / "xdg" / "uv" / "uv.toml").write_text("offline = true\n", encoding = "utf-8")
    assert "offline" in inst._installer_config("uv", env.run)["refusal"]


@pytest.mark.parametrize("uv", ["uv", None])
def test_a_proxy_the_probe_cannot_use_skips_the_probes(env, monkeypatch, uv):
    monkeypatch.setattr(inst, "_reachable", lambda url: False)
    monkeypatch.setattr(inst, "_uv_executable", lambda: uv)
    monkeypatch.setenv("ALL_PROXY", "socks5h://proxy.example:1080")
    assert inst.ensure_flashinfer_for_nvfp4(0, run = _pip_config(env, []))[0]


def test_pips_own_proxy_setting_skips_the_probes(env, monkeypatch):
    monkeypatch.setattr(inst, "_reachable", lambda url: False)
    monkeypatch.setattr(inst, "_uv_executable", lambda: None)
    run = _pip_config(env, ["global.proxy='http://proxy.example:3128'"])
    assert inst.ensure_flashinfer_for_nvfp4(0, run = run)[0]


def test_reachability_counts_a_tls_failure_as_a_route(monkeypatch):
    import ssl
    import urllib.error
    import urllib.request

    def _raise(exc):
        def _urlopen(*args, **kwargs):
            raise exc

        return _urlopen

    cases = [
        (urllib.error.URLError(ssl.SSLCertVerificationError(1, "self-signed certificate")), True),
        (urllib.error.URLError(OSError(101, "Network is unreachable")), False),
        (urllib.error.HTTPError("https://x", 404, "Not Found", {}, None), False),
        (TimeoutError("timed out"), False),
    ]
    for exc, expected in cases:
        monkeypatch.setattr(urllib.request, "urlopen", _raise(exc))
        assert inst._reachable("https://pypi.org/simple/flashinfer-python/") is expected


def test_an_install_killed_between_its_steps_completes_on_the_next_load(env):
    # The server dies (closed, OOM-killed) after the first installer step and before the second. The next load must
    # finish the install, not treat an importable flashinfer-python with no jit-cache as done: without the cache every
    # kernel JIT-compiles, and without nvcc the preflight fails for the life of the environment.
    real_run = env.run
    calls = {"n": 0}

    def dies_on_second_install(cmd, **kwargs):
        if "install" in cmd and "uninstall" not in cmd:
            calls["n"] += 1
            if calls["n"] == 2:
                raise SystemExit("killed mid-install")
        return real_run(cmd, **kwargs)

    with pytest.raises(SystemExit):
        inst.ensure_flashinfer_for_nvfp4(0, run = dies_on_second_install)
    # A fresh process: real flashinfer-python imports with or without its jit-cache.
    inst.reset_install_state()
    env.importable = "flashinfer-python" in env.dists
    ok, reason = inst.ensure_flashinfer_for_nvfp4(0, run = env.run)
    assert ok, reason
    assert env.dists.get("flashinfer-python") == "0.6.6"
    assert env.dists.get("flashinfer-jit-cache") == "0.6.6+cu130", reason


@pytest.mark.parametrize("model", ["DiffusionStatusResponse", "VideoStatusResponse"])
def test_the_status_reason_hides_host_paths_from_api_key_callers(model):
    # The reason quotes installer and flashinfer JIT output, which names the environment and cache directories. The
    # status routes redact host paths for API-key callers; the new field must not bypass that.
    from hub.utils.host_paths import redact_host_paths
    from models import inference as models

    secret = "/home/alice/.unsloth/studio/venv"
    reason = f"flashinfer install rolled back: the installer failed: Using Python environment at: {secret} error"
    status = getattr(models, model)(
        loaded = True, transformer_quant_backend = "torchao", transformer_quant_backend_reason = reason
    )
    seen = redact_host_paths(status, via_api_key = True)
    seen = seen if isinstance(seen, dict) else seen.model_dump()
    assert secret not in str(seen["transformer_quant_backend_reason"])
    assert "rolled back" in seen["transformer_quant_backend_reason"]
    own = redact_host_paths(status, via_api_key = False)
    own = own if isinstance(own, dict) else own.model_dump()
    assert secret in own["transformer_quant_backend_reason"]


def test_a_concurrent_upgrade_by_another_installer_is_not_reverted(env):
    # The built-in terminal does not take the env lock. An upgrade it makes during this transaction is the user's; the
    # drift check still fails this install, but its rollback must not downgrade what it did not touch.
    env.concurrent_add = {"packaging": "26.0"}
    ok, reason = _ensure(env)
    assert not ok and "rolled back" in reason
    assert env.dists["packaging"] == "26.0", reason
    assert not [c for c in env.commands if "packaging==25.0" in c]


def test_a_later_step_that_dies_unreported_is_still_rolled_back(env):
    # One step reports, the next writes its files and then times out with no summary: both must be removed, or the
    # half install imports next time and reads as done.
    real_run = env.run

    def run(cmd, **kwargs):
        result = real_run(cmd, **kwargs)
        steps = [c for c in env.installs() if any(a.startswith("flashinfer") for a in c)]
        if "install" in cmd and "uninstall" not in cmd and len(steps) == 2:
            return _Result(1, "")
        return result

    ok, reason = inst.ensure_flashinfer_for_nvfp4(0, run = run)
    assert not ok and "rolled back" in reason
    assert "flashinfer-python" not in env.dists, reason
    assert "flashinfer-jit-cache" not in env.dists, reason


def test_a_failed_install_does_not_quote_index_credentials(env):
    # The reason reaches the status route; an index URL in the installer's error must not carry its token there.
    real_run = env.run

    def run(cmd, **kwargs):
        if "install" in cmd and "uninstall" not in cmd:
            return _Result(
                1,
                "error: https://__token__:pypi-SECRET@corp.example/simple/ ?token=SECRET2 returned 401",
            )
        return real_run(cmd, **kwargs)

    ok, reason = inst.ensure_flashinfer_for_nvfp4(0, run = run)
    assert not ok and "the installer failed" in reason
    assert "SECRET" not in reason


def test_an_unreported_failure_does_not_revert_a_concurrent_upgrade(env):
    # A step that dies with no summary gives the rollback no report to filter by. The constraints still held every
    # installed package, so an upgrade made meanwhile by the built-in terminal is the user's and must stay.
    env.concurrent_add = {"packaging": "26.0"}
    real_run = env.run

    def run(cmd, **kwargs):
        result = real_run(cmd, **kwargs)
        if (
            "install" in cmd
            and "uninstall" not in cmd
            and any(a.startswith("flashinfer-python==") for a in cmd)
        ):
            return _Result(1, "")
        return result

    ok, reason = inst.ensure_flashinfer_for_nvfp4(0, run = run)
    assert not ok and "rolled back" in reason
    assert "flashinfer-python" not in env.dists, reason
    assert env.dists["packaging"] == "26.0", reason
    assert not [c for c in env.commands if "packaging==25.0" in c]
