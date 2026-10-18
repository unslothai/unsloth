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
