# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Mamba-2 families on a zoo with Triton causal conv drop-ins take the extension-free mamba_ssm
wheel and skip causal-conv1d; everything else keeps the CUDA wheels."""

import sys
import types
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from utils import ssm_runtime, wheel_utils  # noqa: E402


def _kernels(
    model_types,
    *,
    fatal = False,
    remote = frozenset(),
    remote_used = False,
    ssm = True,
):
    return ssm_runtime.SsmKernels(
        causal_conv1d = True,
        mamba_ssm = ssm,
        mamba_ssm_required = True,
        probe = "m",
        model_types = frozenset(model_types),
        remote_code = remote,
        mamba_ssm_fatal = fatal,
        remote_code_used = remote_used,
    )


@pytest.fixture
def capable(monkeypatch, tmp_path):
    """A host where every gate passes; tests flip one at a time."""
    zoo = tmp_path / "unsloth_zoo"
    (zoo / "temporary_patches").mkdir(parents = True)
    (zoo / "temporary_patches" / "gdn_causal_conv1d.py").write_text(
        '_MAMBA2_MODELING = ("nemotron_h",)\ndef install_mamba_ssm_triton_conv():\n    return True\n'
    )
    real_find_spec = ssm_runtime.importlib.util.find_spec

    def find_spec(name, *a, **k):
        if name == "unsloth_zoo":
            return types.SimpleNamespace(submodule_search_locations = [str(zoo)])
        if name == "triton":
            return object()
        return real_find_spec(name, *a, **k)

    monkeypatch.setattr(ssm_runtime.importlib.util, "find_spec", find_spec)
    monkeypatch.setattr(
        ssm_runtime, "_torch_env", lambda: {"cuda_version": "13.0", "cuda_major": "13"}
    )
    monkeypatch.setattr(ssm_runtime, "_transformers_resolves_kernels_per_function", lambda: True)
    monkeypatch.delenv("UNSLOTH_MAMBA_CUDA_KERNELS", raising = False)
    return zoo


@pytest.mark.parametrize(
    "types_",
    [{"nemotron_h"}, {"falcon_h1"}, {"granitemoehybrid"}, {"bamba"}, {"zamba2"}, {"mamba2"}],
)
def test_mamba2_families_run_on_triton(capable, types_):
    assert ssm_runtime.mamba2_runs_on_triton(_kernels(types_))


@pytest.mark.parametrize(
    "types_", [{"mamba"}, {"falcon_mamba"}, {"jamba"}, {"zamba"}, set(), {"nemotron_h", "jamba"}]
)
def test_mamba1_families_and_unknown_types_keep_cuda_wheels(capable, types_):
    assert not ssm_runtime.mamba2_runs_on_triton(_kernels(types_))


def test_each_gate_keeps_the_cuda_wheels(capable, monkeypatch):
    k = _kernels({"nemotron_h"})
    assert ssm_runtime.mamba2_runs_on_triton(k)
    assert not ssm_runtime.mamba2_runs_on_triton(_kernels({"nemotron_h"}, fatal = True))
    assert not ssm_runtime.mamba2_runs_on_triton(
        _kernels({"nemotron_h"}, remote = frozenset({("a", "b")}), remote_used = True)
    )
    # A config that only advertises remote code (native class used) keeps the Triton route.
    assert ssm_runtime.mamba2_runs_on_triton(
        _kernels({"nemotron_h"}, remote = frozenset({("a", "b")}))
    )
    assert not ssm_runtime.mamba2_runs_on_triton(_kernels({"nemotron_h"}, ssm = False))
    monkeypatch.setenv("UNSLOTH_MAMBA_CUDA_KERNELS", "1")
    assert not ssm_runtime.mamba2_runs_on_triton(k)
    monkeypatch.delenv("UNSLOTH_MAMBA_CUDA_KERNELS")
    monkeypatch.setattr(
        ssm_runtime, "_torch_env", lambda: {"hip_version": "7.0", "cuda_version": None}
    )
    assert not ssm_runtime.mamba2_runs_on_triton(k)
    monkeypatch.setattr(
        ssm_runtime, "_torch_env", lambda: {"cuda_version": "13.0", "cuda_major": "13"}
    )
    monkeypatch.setattr(ssm_runtime, "_transformers_resolves_kernels_per_function", lambda: False)
    assert not ssm_runtime.mamba2_runs_on_triton(k)
    monkeypatch.setattr(ssm_runtime, "_transformers_resolves_kernels_per_function", lambda: True)
    (capable / "temporary_patches" / "gdn_causal_conv1d.py").write_text(
        "def patch_gdn_causal_conv1d():\n    pass\n"
    )
    assert not ssm_runtime.mamba2_runs_on_triton(k)


def test_triton_load_installs_only_the_pure_mamba_ssm_wheel(monkeypatch):
    calls = []
    monkeypatch.setattr(ssm_runtime, "_install_kernel", lambda **k: calls.append(k) or True)
    ssm_runtime.ensure_ssm_runtime(
        "nvidia/NVIDIA-Nemotron-Nano-9B-v2",
        wants_causal_conv1d = True,
        is_ssm = True,
        mamba_ssm_required = True,
        mamba_ssm_fatal = False,
        mamba2_triton = True,
    )
    assert [c["import_name"] for c in calls] == ["mamba_ssm"]
    assert calls[0]["allow_source_build"] is False
    url = calls[0]["wheel_url_builder"]({"platform_tag": "win_amd64"})
    assert url.endswith("/prebuilt-wheels-cu13/mamba_ssm-2.3.2.post1-py3-none-any.whl")
    assert url == wheel_utils.mamba_ssm_pure_wheel_url(None)


def test_cuda_load_still_installs_both_kernels(monkeypatch):
    calls = []
    monkeypatch.setattr(ssm_runtime, "_install_kernel", lambda **k: calls.append(k) or True)
    monkeypatch.setattr(ssm_runtime.sys, "platform", "linux")
    ssm_runtime.ensure_ssm_runtime(
        "nvidia/NVIDIA-Nemotron-Nano-9B-v2",
        wants_causal_conv1d = True,
        is_ssm = True,
        mamba_ssm_required = True,
        mamba_ssm_fatal = False,
    )
    assert [c["import_name"] for c in calls] == ["causal_conv1d", "mamba_ssm"]
    assert all(c.get("wheel_url_builder") is None for c in calls)


def test_install_kernel_uses_the_wheel_url_builder(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        ssm_runtime,
        "_torch_env",
        lambda: {"cuda_version": "13.0", "cuda_major": "13", "platform_tag": "linux_x86_64"},
    )
    monkeypatch.setattr(ssm_runtime, "url_exists", lambda url: seen.setdefault("url", url) and True)
    monkeypatch.setattr(
        ssm_runtime,
        "install_prebuilt",
        lambda url, **k: seen.setdefault("installed", url) and "installed",
    )
    monkeypatch.setattr(
        ssm_runtime,
        "direct_wheel_url",
        lambda **k: pytest.fail("platform wheel must not be resolved"),
    )
    assert ssm_runtime._install_kernel_locked(
        **ssm_runtime._pinned_kwargs(ssm_runtime.MAMBA_SSM),
        status_cb = None,
        run = None,
        allow_source_build = False,
        wheel_url_builder = wheel_utils.mamba_ssm_pure_wheel_url,
    )
    assert seen["installed"] == wheel_utils.mamba_ssm_pure_wheel_url()


def test_training_worker_routes_triton_loads_to_the_pure_wheel(monkeypatch):
    from unittest import mock

    from core.training import worker

    install_mock = mock.Mock(return_value = True)
    monkeypatch.setattr(worker, "_install_package_wheel_first", install_mock)
    monkeypatch.setattr(worker.sys, "platform", "linux")
    worker._ensure_mamba_ssm(
        event_queue = [],
        model_name = "nvidia/NVIDIA-Nemotron-Nano-9B-v2",
        required = True,
        essential = True,
        triton_only = True,
    )
    kwargs = install_mock.call_args.kwargs
    assert kwargs["wheel_url_builder"] is wheel_utils.mamba_ssm_pure_wheel_url
    assert kwargs["allow_source_build"] is False and kwargs["import_name"] == "mamba_ssm"
    # And the worker's load path asks the shared predicate before installing anything.
    src = (_BACKEND / "core" / "training" / "worker.py").read_text()
    assert "triton_only = mamba2_runs_on_triton(kernels)" in src


# ── Fallback: no usable extension-free wheel -> the CUDA-wheel route ──────────


def _route(monkeypatch, *, pure_ok):
    calls = []

    def install(**kwargs):
        calls.append((kwargs["import_name"], kwargs.get("wheel_url_builder")))
        if kwargs.get("wheel_url_builder") is wheel_utils.mamba_ssm_pure_wheel_url:
            return pure_ok
        return True

    monkeypatch.setattr(ssm_runtime, "_install_kernel", install)
    monkeypatch.setattr(ssm_runtime.sys, "platform", "linux")
    ssm_runtime.ensure_ssm_runtime(
        "nvidia/NVIDIA-Nemotron-Nano-9B-v2",
        wants_causal_conv1d = True,
        is_ssm = True,
        mamba_ssm_required = True,
        mamba_ssm_fatal = False,
        mamba2_triton = True,
    )
    return calls


def test_a_missing_pure_wheel_falls_back_to_the_cuda_route(monkeypatch):
    pure = wheel_utils.mamba_ssm_pure_wheel_url
    assert _route(monkeypatch, pure_ok = False) == [
        ("mamba_ssm", pure),
        ("causal_conv1d", None),
        ("mamba_ssm", None),
    ]
    assert _route(monkeypatch, pure_ok = True) == [("mamba_ssm", pure)]


@pytest.fixture
def real_install(monkeypatch, tmp_path):
    """_install_kernel for real (lock, memo, cleanup), with the network and pip faked."""
    monkeypatch.setenv("UNSLOTH_KERNEL_LOCK_DIR", str(tmp_path / "locks"))
    monkeypatch.setattr(ssm_runtime, "_FAILED_INSTALLS", set())
    monkeypatch.setattr(ssm_runtime.sys, "platform", "linux")
    monkeypatch.setattr(
        ssm_runtime,
        "_torch_env",
        lambda: {
            "cuda_version": "13.0",
            "cuda_major": "13",
            "platform_tag": "linux_x86_64",
            "torch_mm": "2.13",
        },
    )
    import utils.utils as utils_utils

    monkeypatch.setattr(utils_utils, "package_installs_offline", lambda: False)
    monkeypatch.setattr(ssm_runtime, "_remove_broken", lambda *a, **k: None)
    state = {"importable": set(), "installed": [], "probed": []}
    monkeypatch.setattr(ssm_runtime, "_is_importable", lambda name: name in state["importable"])
    cuda_url = "https://example.invalid/mamba_ssm-2.3.2.post1+cu13torch2.13-cp313-linux_x86_64.whl"
    monkeypatch.setattr(ssm_runtime, "direct_wheel_url", lambda **k: cuda_url)
    state["cuda_url"] = cuda_url
    return state


def _pure_then_cuda(
    state,
    monkeypatch,
    *,
    pure_exists,
    pure_outcome = "installed",
):
    pure_url = wheel_utils.mamba_ssm_pure_wheel_url()

    def url_exists(url):
        state["probed"].append(url)
        return pure_exists if url == pure_url else True

    def install_prebuilt(url, *, verify, **kwargs):
        state["installed"].append(url)
        if url == pure_url and pure_outcome != "installed":
            return pure_outcome
        state["importable"].add("mamba_ssm")
        if url != pure_url:
            state["importable"].add(ssm_runtime.MAMBA_SSM_EXTENSION)  # the CUDA wheel's scan
        return "installed" if verify() else "rejected"

    monkeypatch.setattr(ssm_runtime, "url_exists", url_exists)
    monkeypatch.setattr(ssm_runtime, "install_prebuilt", install_prebuilt)
    ssm_runtime.ensure_ssm_runtime(
        "nvidia/NVIDIA-Nemotron-Nano-9B-v2",
        wants_causal_conv1d = False,
        is_ssm = True,
        mamba_ssm_required = True,
        mamba_ssm_fatal = False,
        mamba2_triton = True,
    )
    return pure_url


def test_a_404_on_the_pure_wheel_installs_the_cuda_wheel(real_install, monkeypatch):
    pure_url = _pure_then_cuda(real_install, monkeypatch, pure_exists = False)
    assert real_install["probed"] == [pure_url, real_install["cuda_url"]]
    assert real_install["installed"] == [real_install["cuda_url"]]
    assert "mamba_ssm" in real_install["importable"]


@pytest.mark.parametrize("outcome", ["rejected", "failed"])
def test_a_pure_wheel_that_does_not_install_or_import_falls_back(
    real_install, monkeypatch, outcome
):
    pure_url = _pure_then_cuda(real_install, monkeypatch, pure_exists = True, pure_outcome = outcome)
    assert real_install["installed"] == [pure_url, real_install["cuda_url"]]
    assert "mamba_ssm" in real_install["importable"]


def test_the_pure_failure_is_remembered_on_its_own_key(real_install, monkeypatch):
    pure_url = _pure_then_cuda(real_install, monkeypatch, pure_exists = False)
    keys = ssm_runtime._FAILED_INSTALLS
    assert ssm_runtime._failure_key("mamba_ssm", wheel_utils.mamba_ssm_pure_wheel_url) in keys
    # The CUDA mamba-ssm installed: nothing remembered against it.
    assert ssm_runtime._failure_key("mamba_ssm") not in keys
    # A later load in this process skips the pure probe and goes straight to the CUDA wheel.
    real_install["importable"].clear()
    real_install["probed"].clear()
    real_install["installed"].clear()
    _pure_then_cuda(real_install, monkeypatch, pure_exists = False)
    assert pure_url not in real_install["probed"]
    assert real_install["installed"] == [real_install["cuda_url"]]


def test_training_worker_reports_whether_the_pure_wheel_took(monkeypatch):
    from unittest import mock

    from core.training import worker

    monkeypatch.setattr(worker.sys, "platform", "linux")
    for result in (True, False):
        monkeypatch.setattr(worker, "_install_package_wheel_first", mock.Mock(return_value = result))
        assert (
            worker._ensure_mamba_ssm(
                event_queue = [],
                model_name = "nvidia/NVIDIA-Nemotron-Nano-9B-v2",
                required = True,
                essential = True,
                triton_only = True,
            )
            is result
        )
    src = (_BACKEND / "core" / "training" / "worker.py").read_text()
    # On a False answer the run takes the CUDA route: causal-conv1d, then the CUDA mamba-ssm.
    start = src.index("triton_only = mamba2_runs_on_triton(kernels)")
    block = src[start : src.index("_ensure_flash_attn_for_long_context(", start)]
    assert "if not triton_only:" in block
    assert block.index("_ensure_causal_conv1d_fast_path(") < block.index("if not triton_only:")


def test_training_failure_memo_follows_the_wheel_source(monkeypatch):
    from core.training import worker

    seen = []
    monkeypatch.setattr(worker, "_is_importable", lambda name: False)
    monkeypatch.setattr(worker, "_model_offline_mode_enabled", lambda: False)
    monkeypatch.setattr(worker, "_package_installs_offline", lambda: False)
    pure_failed = ssm_runtime._failure_key("mamba_ssm", wheel_utils.mamba_ssm_pure_wheel_url)
    monkeypatch.setattr(ssm_runtime, "_FAILED_INSTALLS", {pure_failed})
    monkeypatch.setattr(
        worker,
        "_attempt_package_install",
        lambda **k: seen.append(k.get("wheel_url_builder")) or True,
    )
    kwargs = worker._pinned_kernel_kwargs(worker.MAMBA_SSM)
    # The remembered pure failure skips the pure attempt ...
    assert (
        worker._install_package_wheel_first(
            event_queue = [], **kwargs, wheel_url_builder = wheel_utils.mamba_ssm_pure_wheel_url
        )
        is False
    )
    # ... but not the CUDA wheel.
    assert worker._install_package_wheel_first(event_queue = [], **kwargs) is True
    assert seen == [None]


# ── The CUDA route never settles for the extension-free wheel ─────────────────


def _cuda_route(
    state,
    monkeypatch,
    *,
    cuda_exists = True,
    required = True,
    outcome = "installed",
):
    state["kwargs"] = []
    state["removed"] = []
    state["importable"].add("causal_conv1d")  # every SSM load also wants it; already present here

    def install_prebuilt(url, *, verify, **kwargs):
        state["installed"].append(url)
        state["kwargs"].append(kwargs)
        if outcome != "installed":
            return outcome
        state["importable"].update({"mamba_ssm", ssm_runtime.MAMBA_SSM_EXTENSION})
        return "installed" if verify() else "rejected"

    monkeypatch.setattr(ssm_runtime, "url_exists", lambda url: cuda_exists)
    monkeypatch.setattr(ssm_runtime, "install_prebuilt", install_prebuilt)
    monkeypatch.setattr(
        ssm_runtime,
        "_remove_broken",
        lambda pypi_name, *a, **k: state["removed"].append(pypi_name)
        or state["importable"].discard("mamba_ssm"),
    )
    return ssm_runtime.ensure_ssm_runtime(
        "tiiuae/falcon-mamba-7b",
        wants_causal_conv1d = False,
        is_ssm = True,
        mamba_ssm_required = required,
        mamba_ssm_fatal = False,
    )


def test_cuda_route_replaces_the_extension_free_wheel(real_install, monkeypatch):
    real_install["importable"].add("mamba_ssm")  # a Mamba-2 load left the extension-free wheel
    monkeypatch.setitem(sys.modules, "mamba_ssm", types.ModuleType("mamba_ssm"))
    monkeypatch.setitem(sys.modules, "mamba_ssm.ops", types.ModuleType("mamba_ssm.ops"))
    installed = _cuda_route(real_install, monkeypatch)
    assert real_install["installed"] == [real_install["cuda_url"]]
    assert real_install["kwargs"][0].get("reinstall") is True
    assert installed == ["mamba_ssm"]  # new to this worker: restart detection sees it
    # The stale extension-free modules are gone, so the next import loads the CUDA copy.
    assert "mamba_ssm" not in sys.modules and "mamba_ssm.ops" not in sys.modules


def test_cuda_route_accepts_a_mamba_ssm_with_its_scan(real_install, monkeypatch):
    real_install["importable"].update({"mamba_ssm", ssm_runtime.MAMBA_SSM_EXTENSION})
    assert _cuda_route(real_install, monkeypatch) == []
    assert real_install["installed"] == []


def test_a_failed_replacement_removes_the_extension_free_wheel(real_install, monkeypatch):
    real_install["importable"].add("mamba_ssm")
    installed = _cuda_route(real_install, monkeypatch, cuda_exists = False, required = False)
    assert installed == []
    # Left behind, a Mamba-1 load would take the fast path and fail in selective_scan_fn.
    assert real_install["removed"] == ["mamba-ssm"]
    assert "mamba_ssm" not in real_install["importable"]


def test_the_pure_route_keeps_an_installed_cuda_mamba_ssm(real_install, monkeypatch):
    real_install["importable"].update({"mamba_ssm", ssm_runtime.MAMBA_SSM_EXTENSION})
    monkeypatch.setattr(
        ssm_runtime, "install_prebuilt", lambda *a, **k: pytest.fail("no downgrade")
    )
    assert (
        ssm_runtime.ensure_ssm_runtime(
            "nvidia/NVIDIA-Nemotron-Nano-9B-v2",
            wants_causal_conv1d = True,
            is_ssm = True,
            mamba_ssm_required = True,
            mamba_ssm_fatal = False,
            mamba2_triton = True,
        )
        == []
    )


def test_training_cuda_route_replaces_the_extension_free_wheel(monkeypatch, tmp_path):
    from core.training import worker

    importable = {"mamba_ssm"}
    seen = {}
    monkeypatch.setattr(worker, "_is_importable", lambda name: name in importable)
    monkeypatch.setattr(worker, "_model_offline_mode_enabled", lambda: False)
    monkeypatch.setattr(worker, "_package_installs_offline", lambda: False)
    monkeypatch.setattr(ssm_runtime, "_FAILED_INSTALLS", set())
    monkeypatch.setenv("UNSLOTH_KERNEL_LOCK_DIR", str(tmp_path / "locks"))

    def attempt(**kwargs):
        seen.update(kwargs)
        importable.add(ssm_runtime.MAMBA_SSM_EXTENSION)
        return True

    monkeypatch.setattr(worker, "_attempt_package_install", attempt)
    monkeypatch.setitem(sys.modules, "mamba_ssm", types.ModuleType("mamba_ssm"))
    from unittest import mock

    assert worker._install_package_wheel_first(
        event_queue = mock.Mock(),
        **worker._pinned_kernel_kwargs(worker.MAMBA_SSM),
        extension = ssm_runtime.MAMBA_SSM_EXTENSION,
    )
    assert seen["replacing"] is True and seen["extension"] == ssm_runtime.MAMBA_SSM_EXTENSION
    assert "mamba_ssm" not in sys.modules
    # And the CUDA-route call asks for the extension.
    src = (_BACKEND / "core" / "training" / "worker.py").read_text()
    assert "extension = MAMBA_SSM_EXTENSION" in src


# ── Remote code counts only when the load runs it ─────────────────────────────


@pytest.mark.parametrize(
    "trust, native, used", [(False, True, False), (True, True, True), (False, False, True)]
)
def test_remote_code_used_follows_the_selected_implementation(monkeypatch, trust, native, used):
    config = {
        "model_type": "nemotron_h",
        "architectures": ["NemotronHForCausalLM"],
        "auto_map": {"AutoModelForCausalLM": "modeling_nemotron_h.NemotronHForCausalLM"},
    }
    monkeypatch.setattr(ssm_runtime, "native_modeling_available", lambda t: native)
    monkeypatch.setattr(ssm_runtime, "_remote_source_text", lambda *a, **k: "import torch\n")
    kernels = ssm_runtime.resolve_ssm_kernels(
        "nvidia/NVIDIA-Nemotron-Nano-9B-v2",
        "nvidia/NVIDIA-Nemotron-Nano-9B-v2",
        None,
        model_config = config,
        trust_remote_code = trust,
    )
    assert kernels.remote_code  # restart detection still knows the remote modules
    assert kernels.remote_code_used is used
