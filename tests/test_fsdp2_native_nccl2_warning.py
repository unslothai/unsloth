# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Native FSDP2 warns once when the process group runs torch 2.14's nccl2 on small GPUs, and not under
torch 2.15, whose default "nccl" is nccl2 without the GPU 0 contexts (CPU only: the process group,
the torch version and the CUDA device queries are faked)."""

import os
import sys
import types

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_fsdp2_native_units import _load_native  # noqa: E402

native = _load_native()


class ProcessGroupNCCL2:
    pass


class ProcessGroupNCCL:
    pass


def _dist(
    impl,
    name = "nccl",
    world = 2,
):
    group = types.SimpleNamespace(_get_backend = lambda device: impl())
    return types.SimpleNamespace(
        distributed_c10d = types.SimpleNamespace(_get_default_group = lambda: group),
        get_backend = lambda: name,
        get_world_size = lambda: world,
    )


@pytest.fixture
def gpus(monkeypatch):
    state = {"count": 2, "total": 15 << 30}
    monkeypatch.setattr(torch.cuda, "device_count", lambda: state["count"])
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: types.SimpleNamespace(total_memory = state["total"]),
    )
    monkeypatch.setattr(native, "_NCCL2_WARNED", False)
    monkeypatch.setattr(torch, "__version__", "2.14.1+cu130")
    return state


DEV = torch.device("cuda", 0)


def test_warns_once_for_nccl2_on_a_small_gpu(gpus, capsys):
    native._warn_nccl2_memory(_dist(ProcessGroupNCCL2), DEV)
    native._warn_nccl2_memory(_dist(ProcessGroupNCCL2), DEV)
    assert capsys.readouterr().out.count("nccl2 backend") == 1


def test_env_redirect_is_detected_from_the_backend_object(gpus, capsys):
    # TORCH_DIST_USE_NCCL2=1: get_backend() still says "nccl"; the backend object is nccl2.
    native._warn_nccl2_memory(_dist(ProcessGroupNCCL2, name = "nccl"), DEV)
    assert "nccl2 backend" in capsys.readouterr().out


@pytest.mark.parametrize(
    "case",
    ["plain_nccl", "large_gpu", "one_visible_gpu", "one_rank", "cpu"],
)
def test_no_warning(gpus, capsys, case):
    impl, world, device = ProcessGroupNCCL2, 2, DEV
    if case == "plain_nccl":
        impl = ProcessGroupNCCL
    elif case == "large_gpu":
        gpus["total"] = 80 << 30
    elif case == "one_visible_gpu":
        gpus["count"] = 1
    elif case == "one_rank":
        world = 1
    elif case == "cpu":
        device = torch.device("cpu")
    native._warn_nccl2_memory(_dist(impl, world = world), device)
    assert capsys.readouterr().out == ""


def test_name_fallback_when_backend_object_is_unavailable(gpus, capsys, monkeypatch):
    def broken(device):
        raise RuntimeError("no backend for device")

    group = types.SimpleNamespace(_get_backend = broken)
    dist = _dist(ProcessGroupNCCL)
    dist.distributed_c10d = types.SimpleNamespace(_get_default_group = lambda: group)
    monkeypatch.setenv("TORCH_DIST_USE_NCCL2", "1")
    assert native._uses_nccl2(dist, DEV) is True
    dist.get_backend = lambda: "nccl2"
    monkeypatch.delenv("TORCH_DIST_USE_NCCL2")
    assert native._uses_nccl2(dist, DEV) is True
    dist.get_backend = lambda: "gloo"
    assert native._uses_nccl2(dist, DEV) is False


@pytest.mark.parametrize("version", ["2.15.0+cu132", "2.16.0.dev20261101+cu132", "3.0.0"])
def test_no_warning_where_nccl2_is_the_default(gpus, capsys, monkeypatch, version):
    # torch 2.15's nccl2 (the default "nccl") builds no context on GPU 0 from the other ranks.
    monkeypatch.setattr(torch, "__version__", version)
    native._warn_nccl2_memory(_dist(ProcessGroupNCCL2), DEV)
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize(
    "version, env, expected",
    [
        ("2.14.1", None, False),
        ("2.14.1", "1", True),
        ("2.14.1", "0", False),
        ("2.15.0", None, True),
        ("2.15.0", "1", True),
        ("2.15.0", "0", False),
    ],
)
def test_name_fallback_follows_the_torch_default(monkeypatch, version, env, expected):
    # torch 2.15 builds "nccl" as ProcessGroupNCCL2 unless TORCH_DIST_USE_NCCL2=0; 2.14 only with =1.
    def broken(device):
        raise RuntimeError("no backend for device")

    dist = _dist(ProcessGroupNCCL)
    dist.distributed_c10d = types.SimpleNamespace(
        _get_default_group = lambda: types.SimpleNamespace(_get_backend = broken)
    )
    monkeypatch.setattr(torch, "__version__", version)
    if env is None:
        monkeypatch.delenv("TORCH_DIST_USE_NCCL2", raising = False)
    else:
        monkeypatch.setenv("TORCH_DIST_USE_NCCL2", env)
    assert native._uses_nccl2(dist, DEV) is expected
