# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""SageAttention 2 is not on PyPI; FA4's nvidia-cutlass-dsl import works only at 4.4.x and 4.5.x."""

from __future__ import annotations

import pathlib
import sys
import types

import pytest

import core.inference.diffusion_attention as att
from core.inference.diffusion_attention import apply_attention_backend

torch = pytest.importorskip("torch")
dispatch = pytest.importorskip("diffusers.models.attention_dispatch")

_HUB = getattr(dispatch.AttentionBackendName, "SAGE_HUB", None)
needs_sage_hub = pytest.mark.skipif(_HUB is None, reason = "this diffusers has no sage_hub backend")


class _Transformer:
    def __init__(self, head_dim = 128):
        self.calls: list = []
        self.config = {"attention_head_dim": head_dim}

    def set_attention_backend(self, name):
        self.calls.append(name)


class _Logger:
    def __init__(self):
        self.warnings: list = []
        self.infos: list = []

    def warning(self, msg, *args):
        self.warnings.append(msg % args if args else msg)

    def info(self, msg, *args):
        self.infos.append(msg % args if args else msg)


def _target(dtype = "bf16"):
    return types.SimpleNamespace(device = "cuda", dtype = dtype, torch_device = "cuda")


def _exact_sageattn(
    q,
    k,
    v,
    tensor_layout = "NHD",
    **_kw,
):
    return (
        torch.nn.functional.scaled_dot_product_attention(
            *(t.float().transpose(1, 2) for t in (q, k, v))
        )
        .transpose(1, 2)
        .to(q.dtype)
    )


@pytest.fixture(autouse = True)
def _isolated(monkeypatch):
    monkeypatch.setattr(att, "_SAGE_PROBE_CACHE", {})
    monkeypatch.setattr(att, "_SAGE_HUB_PROBE_CACHE", {}, raising = False)
    monkeypatch.setattr(att, "_SAGE_HUB_FAILED", [], raising = False)
    monkeypatch.setattr(att, "_INSTALL_ATTEMPTED", set())
    monkeypatch.setattr(att, "_ensure_attention_backend_installed", lambda *a, **k: None)
    monkeypatch.setattr(att, "_active_attention_backend", lambda: "native")
    monkeypatch.setattr(att, "warn_if_sdpa_math_only", lambda *a, **k: False)
    monkeypatch.setattr(att, "_indexed_cuda_device", lambda device: device)
    monkeypatch.setattr(att, "_pip_sage2_installed", lambda: False, raising = False)
    backends = dispatch._AttentionBackendRegistry._backends
    saved = dict(backends)
    hub_cfg = dispatch._HUB_KERNELS_REGISTRY.get(_HUB) if _HUB is not None else None
    saved_fn = getattr(hub_cfg, "kernel_fn", None)
    yield
    backends.clear()
    backends.update(saved)
    if hub_cfg is not None:
        hub_cfg.kernel_fn = saved_fn


def _engage(monkeypatch, probe_error = ""):
    monkeypatch.setattr(att, "_load_sage_hub_kernel", lambda: (_exact_sageattn, ""))
    monkeypatch.setattr(att, "_run_sage_probe", lambda d, dt, hd = 128, sageattn = None: probe_error)
    t, log = _Transformer(), _Logger()
    engaged = apply_attention_backend(
        types.SimpleNamespace(transformer = t), "sage", logger = log, target = _target()
    )
    return engaged, t, log


@needs_sage_hub
def test_sage_without_sageattention2_engages_the_kernels_hub_build(monkeypatch):
    engaged, t, log = _engage(monkeypatch)
    assert engaged == "sage_hub"
    assert t.calls == ["sage_hub"]
    assert getattr(t, att.ATTENTION_BACKEND_ATTR) == "sage_hub"


@needs_sage_hub
def test_sage_hub_probe_failure_keeps_the_default_and_says_why(monkeypatch):
    engaged, t, log = _engage(
        monkeypatch, probe_error = "ValueError: Unsupported CUDA architecture: sm100"
    )
    assert engaged is None and "sage_hub" not in t.calls
    assert any(
        "sm100" in w and "source build or a community wheel" in w for w in log.warnings
    ), log.warnings


@needs_sage_hub
def test_sage_hub_unloadable_keeps_the_default_and_says_why(monkeypatch):
    monkeypatch.setattr(att, "_load_sage_hub_kernel", lambda: (None, "no build for torch 2.13"))
    t, log = _Transformer(), _Logger()
    assert (
        apply_attention_backend(
            types.SimpleNamespace(transformer = t), "sage", logger = log, target = _target()
        )
        is None
    )
    assert t.calls == [] or t.calls == ["native"]
    assert any("no build for torch 2.13" in w and "PyPI" in w for w in log.warnings), log.warnings


@needs_sage_hub
def test_sage_hub_refuses_float32_and_large_head_dims(monkeypatch):
    monkeypatch.setattr(att, "_load_sage_hub_kernel", lambda: pytest.fail("must not fetch"))
    log = _Logger()
    pipe = types.SimpleNamespace(transformer = _Transformer())
    assert apply_attention_backend(pipe, "sage", logger = log, target = _target("float32")) is None
    pipe = types.SimpleNamespace(transformer = _Transformer(head_dim = 256))
    assert apply_attention_backend(pipe, "sage", logger = log, target = _target()) is None


@needs_sage_hub
def test_sage_hub_masked_call_runs_native(monkeypatch):
    engaged, _t, _log = _engage(monkeypatch)
    assert engaged == "sage_hub"
    fn = dispatch._AttentionBackendRegistry._backends[_HUB]
    q = torch.randn(1, 16, 2, 64)
    mask = torch.ones(1, 1, 16, 16, dtype = torch.bool)
    native = dispatch._AttentionBackendRegistry._backends[dispatch.AttentionBackendName.NATIVE]
    torch.testing.assert_close(fn(q, q, q, attn_mask = mask), native(q, q, q, attn_mask = mask))


@needs_sage_hub
def test_sage_hub_loader_prefers_the_version_2_builds(monkeypatch):
    # diffusers 0.40 asks for version 1, whose builds stop at torch 2.10; version 2 covers torch 2.9 to 2.12.
    asked: list = []

    def _get_kernel(
        repo,
        version = None,
        **_kw,
    ):
        asked.append((repo, version))
        if version == 2:
            raise FileNotFoundError("Cannot install kernel")
        return types.SimpleNamespace(sageattn = _exact_sageattn)

    monkeypatch.setitem(sys.modules, "kernels", types.SimpleNamespace(get_kernel = _get_kernel))
    dispatch._HUB_KERNELS_REGISTRY[_HUB].kernel_fn = None
    fn, why = att._load_sage_hub_kernel()
    assert fn is _exact_sageattn and why == ""
    assert asked == [
        ("kernels-community/sage-attention", 2),
        ("kernels-community/sage-attention", 1),
    ]
    assert dispatch._HUB_KERNELS_REGISTRY[_HUB].kernel_fn is _exact_sageattn
    asked.clear()
    assert att._load_sage_hub_kernel()[0] is _exact_sageattn and asked == []


@needs_sage_hub
def test_sage_hub_loader_reports_every_failed_version(monkeypatch):
    def _get_kernel(
        repo,
        version = None,
        **_kw,
    ):
        raise FileNotFoundError(f"no variant v{version}")

    monkeypatch.setitem(sys.modules, "kernels", types.SimpleNamespace(get_kernel = _get_kernel))
    dispatch._HUB_KERNELS_REGISTRY[_HUB].kernel_fn = None
    fn, why = att._load_sage_hub_kernel()
    assert fn is None and "no variant v2" in why and "no variant v1" in why
    monkeypatch.setitem(
        sys.modules,
        "kernels",
        types.SimpleNamespace(get_kernel = lambda *a, **k: pytest.fail("refetch")),
    )
    assert att._load_sage_hub_kernel() == (None, why)


def test_a_pip_sageattention2_keeps_the_pip_path(monkeypatch):
    monkeypatch.setattr(att, "_pip_sage2_installed", lambda: True)
    monkeypatch.setattr(att, "_sage_version_too_old", lambda: None)
    monkeypatch.setattr(att, "_install_sage_dispatch_guard", lambda: True)
    monkeypatch.setattr(att, "_run_sage_probe", lambda d, dt, hd = 128, sageattn = None: "")
    monkeypatch.setattr(
        att, "_load_sage_hub_kernel", lambda: pytest.fail("the hub build is only the fallback")
    )
    t = _Transformer()
    assert (
        apply_attention_backend(types.SimpleNamespace(transformer = t), "sage", target = _target())
        == "sage"
    )


class _Run:
    def __init__(self):
        self.calls: list = []

    def __call__(self, cmd, **kwargs):
        self.calls.append([str(c) for c in cmd])
        return types.SimpleNamespace(returncode = 0, stdout = b"", stderr = b"")


@pytest.fixture
def real_install(monkeypatch):
    """The real installer with a recorded subprocess and a stubbed distribution table."""
    import importlib
    import subprocess

    monkeypatch.undo()  # drop the autouse no-op installer
    monkeypatch.setattr(att, "_INSTALL_ATTEMPTED", set())
    monkeypatch.setattr(att, "_SAGE_HUB_FAILED", [])
    monkeypatch.setattr(att, "_load_sage_hub_kernel", lambda: (None, "stubbed"))
    monkeypatch.setenv("UNSLOTH_DIFFUSION_ATTENTION_INSTALL", "auto")
    run = _Run()
    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(importlib, "invalidate_caches", lambda: None)
    monkeypatch.setattr(att, "_kernels_hub_compatible", lambda: True)
    monkeypatch.setattr(att, "_pinned_constraints_file", lambda: None)
    dists: dict = {"einops": "0.8.2"}
    monkeypatch.setattr(att, "_dist_version", lambda name: dists.get(name))
    monkeypatch.setattr(att, "_cutlass_dsl_dependents", lambda: [])
    return run, dists


def test_sage_install_never_asks_pypi_for_sageattention(real_install, monkeypatch):
    run, _dists = real_install
    import importlib.util

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    monkeypatch.setattr(att, "_pip_sage2_installed", lambda: False)
    att._ensure_attention_backend_installed("sage")
    assert run.calls and run.calls[0][-1] == "kernels"
    assert not any("sageattention" in part for cmd in run.calls for part in cmd)


def test_sage_preinstall_fetches_the_hub_build_outside_the_lock(real_install, monkeypatch):
    run, _dists = real_install
    import importlib.util

    fetched = []
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(att, "_pip_sage2_installed", lambda: False)
    monkeypatch.setattr(att, "_load_sage_hub_kernel", lambda: fetched.append(1) or (None, "x"))
    assert att._ensure_attention_backend_installed("sage") is None
    assert fetched == [1] and run.calls == []


def test_flash4_installs_its_python_deps_with_cutlass_dsl_held_to_the_working_range(
    real_install, monkeypatch
):
    run, dists = real_install
    import importlib.util

    dists["kernels"] = "0.12.1"
    monkeypatch.setattr(
        importlib.util, "find_spec", lambda name: object() if name == "kernels" else None
    )
    att._ensure_attention_backend_installed("flash_4_hub")
    flat = [part for cmd in run.calls for part in cmd]
    assert "nvidia-cutlass-dsl>=4.4,<4.6" in flat
    assert att.FA4_TVM_FFI_SPEC in flat
    kernels_cmd = next(cmd for cmd in run.calls if "kernels==0.12.3" in cmd)
    assert "--no-deps" in kernels_cmd and "--only-binary" in kernels_cmd
    deps_cmd = next(cmd for cmd in run.calls if "nvidia-cutlass-dsl>=4.4,<4.6" in cmd)
    assert "--no-deps" not in deps_cmd  # cutlass-dsl needs its libs and cuda-python
    n = len(run.calls)
    att._ensure_attention_backend_installed("flash_4_hub")
    assert len(run.calls) == n


def test_flash4_deps_already_satisfied_install_nothing(real_install, monkeypatch):
    run, dists = real_install
    import importlib.util

    dists.update(
        {"kernels": "0.12.3", "nvidia-cutlass-dsl": "4.5.3", "apache-tvm-ffi": "0.1.14.post1"}
    )
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())
    assert att._ensure_attention_backend_installed("flash_4_hub") is None
    assert run.calls == []


def test_flash4_refuses_to_replace_an_out_of_range_cutlass_dsl(real_install, monkeypatch):
    run, dists = real_install
    import importlib.util

    dists.update(
        {"kernels": "0.16.0", "nvidia-cutlass-dsl": "4.8.0", "apache-tvm-ffi": "0.1.14.post1"}
    )
    monkeypatch.setattr(
        att, "_cutlass_dsl_dependents", lambda: ["quack-kernels 0.6.5 (nvidia-cutlass-dsl>=4.7)"]
    )
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())
    log = _Logger()
    reason = att._ensure_attention_backend_installed("flash_4_hub", log)
    assert reason and "4.8.0" in reason and "quack-kernels" in reason
    assert run.calls == []
    assert any("4.4.x or 4.5.x" in w for w in log.warnings)


def test_flash4_install_respects_the_off_switch(real_install, monkeypatch):
    run, dists = real_install
    import importlib.util

    monkeypatch.setenv("UNSLOTH_DIFFUSION_ATTENTION_INSTALL", "0")
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())
    att._ensure_attention_backend_installed("flash_4_hub")
    assert run.calls == []


@pytest.mark.parametrize(
    "version, ok",
    [
        ("4.3.5", False),
        ("4.4.0", True),
        ("4.4.2", True),
        ("4.5.3", True),
        ("4.6.0", False),
        ("4.8.0", False),
        (None, False),
    ],
)
def test_flash4_cutlass_dsl_range(version, ok):
    # Measured on a B200 (torch 2.11 / 2.12): 4.4.0 to 4.5.3 load; 4.6.3 to 4.8.0 lack cute.core.ThrMma.
    assert att.fa4_cutlass_dsl_ok(version) is ok


def test_studio_pins_a_kernels_release_diffusers_accepts_for_flash4():
    # diffusers' flash_4_hub raises below kernels 0.12.3.
    req = pathlib.Path(att.__file__).resolve().parents[2] / "requirements" / "extras-no-deps.txt"
    pins = [
        line.split("==", 1)[1].strip()
        for line in req.read_text(encoding = "utf-8").splitlines()
        if line.startswith("kernels==")
    ]
    assert pins and all(att._version_tuple(v) >= att.FA4_KERNELS_MIN for v in pins), pins


def test_a_kernels_upgrade_is_visible_to_diffusers_in_the_same_process(monkeypatch):
    from diffusers.utils import import_utils

    monkeypatch.setattr(import_utils, "_kernels_version", "0.12.1")
    monkeypatch.setattr(import_utils, "_kernels_available", True)
    monkeypatch.delitem(sys.modules, "kernels", raising = False)
    monkeypatch.setattr(att, "_dist_version", lambda name: "0.12.3" if name == "kernels" else None)
    att._refresh_diffusers_kernels_version()
    assert import_utils.is_kernels_version(">=", "0.12.3")


def test_refresh_drops_a_memoized_pre_upgrade_check(monkeypatch):
    from diffusers.utils import import_utils

    monkeypatch.setattr(import_utils, "_kernels_version", "0.12.1")
    monkeypatch.setattr(import_utils, "_kernels_available", True)
    monkeypatch.delitem(sys.modules, "kernels", raising = False)
    clear = getattr(import_utils.is_kernels_version, "cache_clear", None)
    if clear is not None:
        clear()
    assert not import_utils.is_kernels_version(">=", "0.12.3")
    monkeypatch.setattr(att, "_dist_version", lambda name: "0.12.3" if name == "kernels" else None)
    att._refresh_diffusers_kernels_version()
    assert import_utils.is_kernels_version(">=", "0.12.3")
    if clear is not None:
        clear()


def test_an_imported_old_kernels_is_not_papered_over(monkeypatch):
    from diffusers.utils import import_utils

    monkeypatch.setattr(import_utils, "_kernels_version", "0.12.1")
    monkeypatch.setitem(sys.modules, "kernels", types.SimpleNamespace())
    monkeypatch.setattr(att, "_dist_version", lambda name: "0.12.3" if name == "kernels" else None)
    log = _Logger()
    att._refresh_diffusers_kernels_version(log)
    assert import_utils._kernels_version == "0.12.1"
    assert any("after Studio restarts" in w for w in log.warnings)


def test_a_pth_installed_mid_process_is_activated(monkeypatch, tmp_path):
    import importlib.metadata as md

    pkg_dir = tmp_path / "dsl_packages"
    pkg_dir.mkdir()
    (tmp_path / "nvidia_cutlass_dsl_packages.pth").write_text(
        f"import sys; sys.path.insert(0, {str(pkg_dir)!r})\n"
    )

    class _Dist:
        files = [
            pathlib.PurePosixPath("nvidia_cutlass_dsl_packages.pth"),
            pathlib.PurePosixPath("pkg/__init__.py"),
        ]

        def locate_file(self, entry):
            return tmp_path / entry

    monkeypatch.setattr(md, "distribution", lambda name: _Dist())
    monkeypatch.setattr(sys, "path", list(sys.path))
    att._activate_installed_pth_files()
    assert str(pkg_dir) in sys.path


def test_nothing_to_install_still_activates_cutlass_pth(monkeypatch):
    calls = []
    monkeypatch.setattr(att, "_fa4_python_deps_plan", lambda: ([], None))
    monkeypatch.setattr(att, "_activate_installed_pth_files", lambda *a: calls.append(a))
    assert att._ensure_fa4_python_deps() is None
    assert calls == [()]
