# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The auto tensor-split fallback across every host Unsloth Studio supports.

``test_pr10884_auto_tensor_split_sim.py`` varies the ratio, the card count and
the planner's answer. This file varies the HOST: the Cartesian product of
{Linux, Windows, WSL, macOS} x {NVIDIA, AMD ROCm, AMD or Intel via Vulkan,
CPU only}, because the code the PR changed sits upstream of every one of them
and reads `sys.platform`, `_is_wsl` and torch's ROCm markers on the way to a
launch.

Everything is simulated on a CPU-only Linux host, and the simulation is
deliberately at the lowest level the production code reads:

* the OS is ``sys.platform`` plus ``_is_wsl``, which is what the module itself
  branches on;
* the vendor is a fake ``torch`` in ``sys.modules`` carrying (or not carrying)
  ``version.hip``, which is exactly what ``_host_torch_is_rocm`` reads, plus
  the Vulkan backend flag;
* the cards are the ``(index, free, total)`` rows ``_get_gpu_memory`` returns.

What this can and cannot say is worth stating. It proves the placement
decision, the emitted argv and the child's device mask on each host. It does
not prove that a real ROCm or Metal driver then behaves; only hardware can say
that, and the NVIDIA half of the table is the half that has also been run on
two real T4s.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

import pytest  # noqa: E402

from test_llama_cpp_placement import _backend, _launch  # noqa: E402

import core.inference.llama_cpp as llama_cpp  # noqa: E402

GIB = 1024**3

# (index, free_mib, total_mib). Two roomy cards: enough that the planner decides
# an even share is safe and returns nothing, which is the branch the PR added.
TWO_CARDS = [(0, 24_000, 24_000), (1, 24_000, 24_000)]

OSES = {
    # label: (sys.platform, _is_wsl)
    "linux": ("linux", False),
    "windows": ("win32", False),
    "wsl": ("linux", True),
    "macos": ("darwin", False),
}

VENDORS = ("nvidia", "rocm", "vulkan", "cpu")


def _fake_torch(*, rocm: bool, device_count: int) -> types.ModuleType:
    """Enough torch for the three things the launch path asks it.

    ``_host_torch_is_rocm`` reads ``version.hip`` and ``__version__``;
    ``_effective_gpu_count`` falls back to ``cuda.device_count()`` when the
    caller pinned no ids. A real import is not wanted here: the point is to
    put the host under test, not the host running the test.
    """
    torch = types.ModuleType("torch")
    torch.__version__ = "2.11.0+rocm6.2" if rocm else "2.11.0+cu130"
    version = types.ModuleType("torch.version")
    version.hip = "6.2.0" if rocm else None
    version.cuda = None if rocm else "13.0"
    torch.version = version
    cuda = types.SimpleNamespace(
        is_available = lambda: device_count > 0,
        device_count = lambda: device_count,
    )
    torch.cuda = cuda
    return torch


@pytest.fixture
def host(monkeypatch):
    """Put the process on a named OS and vendor for the duration of one test."""

    def _apply(
        os_label: str,
        vendor: str,
        *,
        cards = TWO_CARDS,
        tmp_path = None,
    ):
        platform, is_wsl = OSES[os_label]
        monkeypatch.setattr(sys, "platform", platform)
        monkeypatch.setattr(llama_cpp, "_is_wsl", lambda: is_wsl, raising = False)
        # A real Mac is never the paravirtual case unless it is a VM; that shape
        # has its own test in the sibling file.
        monkeypatch.setattr(llama_cpp, "_metal_device_is_paravirtual", lambda: False, raising = False)
        visible = [] if vendor == "cpu" else cards
        monkeypatch.setitem(
            sys.modules,
            "torch",
            _fake_torch(rocm = vendor == "rocm", device_count = len(visible)),
        )
        backend, gguf = _backend(tmp_path, vulkan = vendor == "vulkan", memory = list(visible))
        backend._can_estimate_kv = lambda: True
        backend._estimate_kv_cache_bytes = lambda *a, **k: 0
        backend._compute_buffer_ctx_bytes = lambda *a, **k: 0
        backend._get_gguf_size_bytes = lambda _path: 1 * GIB
        backend._TENSOR_PARALLEL_BUFFER_RESERVE_MIB = 256
        return backend, gguf, visible

    return _apply


def _flag(cmd, name):
    hits = [i for i, tok in enumerate(cmd) if tok == name]
    assert len(hits) <= 1, f"{name} appears {len(hits)} times"
    return cmd[hits[0] + 1] if hits else None


def _auto_tp(backend, gguf, visible, **kwargs):
    params = dict(gpu_memory_mode = "auto", tensor_parallel = True, n_ctx = 4096)
    if visible:
        params["gpu_ids"] = [idx for idx, *_ in visible]
    params.update(kwargs)
    return _launch(backend, gguf, **params)


@pytest.mark.parametrize("os_label", sorted(OSES))
@pytest.mark.parametrize("vendor", VENDORS)
def test_the_user_ratio_survives_every_host(tmp_path, host, os_label, vendor):
    """One statement per host: the load launches, and the ratio either reaches
    llama-server or is absent for a reason this table can name."""
    backend, gguf, visible = host(os_label, vendor, tmp_path = tmp_path)
    captured = _auto_tp(backend, gguf, visible, tensor_split = [3, 1])
    cmd = captured["cmd"]

    if vendor == "cpu":
        # Nothing to split. The mode is dropped before the emit block, and the
        # new fallback must not resurrect it.
        assert _flag(cmd, "--tensor-split") is None
        return

    assert _flag(cmd, "--split-mode") == "tensor", f"{os_label}/{vendor}"
    assert _flag(cmd, "--tensor-split") == "3,1", f"{os_label}/{vendor}"


@pytest.mark.parametrize("os_label", sorted(OSES))
@pytest.mark.parametrize("vendor", VENDORS)
def test_a_failed_plan_still_launches_on_every_host(tmp_path, host, os_label, vendor):
    """The placement planner's except arm is a designed degradation: it drops
    the plan, sets --fit on and launches anyway. The fallback must not turn
    that into an exception on ANY host -- it was a KeyError on Vulkan and a
    TypeError on CUDA and ROCm before this was gated.

    Three cards, one of them below the tensor-parallel compute-buffer reserve,
    because that is what makes the arm's rebuilt `gpu_indices` WIDER than the
    `tp_gpus` the planner had filtered. Two equal cards never disagree, and a
    version of this test that used them passed against the unfixed revision.
    """
    cards = [(0, 24_000, 24_000), (1, 24_000, 24_000), (2, 200, 24_000)]
    backend, gguf, visible = host(os_label, vendor, cards = cards, tmp_path = tmp_path)

    def _boom(*args, **kwargs):
        raise RuntimeError("simulated GPU selection failure")

    backend._plan_tensor_parallel = _boom
    cmd = _auto_tp(backend, gguf, visible, tensor_split = [2, 1, 1])["cmd"]
    assert cmd, f"{os_label}/{vendor} did not launch"


@pytest.mark.parametrize("os_label", sorted(OSES))
@pytest.mark.parametrize("vendor", ("nvidia", "rocm", "vulkan"))
def test_the_ratio_and_the_device_pin_agree_on_every_host(tmp_path, host, os_label, vendor):
    """A split is positional over the devices the child can SEE, so the ratio
    is meaningless unless it has one weight per pinned device. Each vendor
    pins through a different channel, and the PR does not touch any of them
    (the diff contains no mask line), so what is asserted here is the
    AGREEMENT rather than a particular spelling:

    * NVIDIA masks with ``CUDA_VISIBLE_DEVICES``;
    * ROCm on Linux and WSL masks at the ROCr layer, re-indexing the CUDA mask
      to the surviving ordinals; on Windows there is no ROCr, so it keeps the
      HIP mask instead;
    * Vulkan does not mask at all, it pins ``--device VulkanN`` on the argv.
    """
    backend, gguf, visible = host(os_label, vendor, tmp_path = tmp_path)
    captured = _auto_tp(backend, gguf, visible, tensor_split = [3, 1])
    env, cmd = captured["env"], captured["cmd"]

    weights = _flag(cmd, "--tensor-split").split(",")

    if vendor == "vulkan":
        pinned = _flag(cmd, "--device").split(",")
        assert pinned == ["Vulkan0", "Vulkan1"], f"{os_label}/{vendor}"
    elif vendor == "rocm" and sys.platform == "win32":
        pinned = env["HIP_VISIBLE_DEVICES"].split(",")
        assert "ROCR_VISIBLE_DEVICES" not in env, "ROCr does not exist on Windows"
    elif vendor == "rocm":
        pinned = env["ROCR_VISIBLE_DEVICES"].split(",")
        # With ROCr masking, HIP honours the CUDA mask, which must carry the
        # POST-mask ordinals or a non-zero pick points out of range.
        assert env["CUDA_VISIBLE_DEVICES"].split(",") == [
            str(i) for i in range(len(pinned))
        ], f"{os_label}/{vendor}"
    else:
        pinned = env["CUDA_VISIBLE_DEVICES"].split(",")

    assert (
        len(weights) == len(pinned) == 2
    ), f"{os_label}/{vendor}: {len(weights)} weights for {len(pinned)} pinned devices"


@pytest.mark.parametrize("os_label", sorted(OSES))
@pytest.mark.parametrize("vendor", ("nvidia", "rocm", "vulkan"))
def test_an_identical_request_reuses_the_server_on_every_host(
    tmp_path, host, os_label, vendor, monkeypatch
):
    """The reload-dedupe half, per host. A fix that forwards the ratio and then
    reloads on every identical request has traded one bug for a worse one, and
    the comparison runs on every platform."""
    from core.inference.llama_cpp import GgufLoadIntent, LlamaCppBackend

    backend, gguf, visible = host(os_label, vendor, tmp_path = tmp_path)
    # The matcher asks the binary about MTP through a CLASSMETHOD, so the
    # instance stub `_backend` installs does not cover it, and under a
    # simulated `sys.platform = "win32"` the stdlib's own `shutil.which` then
    # reaches into `_winapi`, which does not exist on this Linux host. That is
    # this simulation's ceiling, not the product's: the real Windows answer is
    # the windows-latest runner. Stub the probe so the axis under test is the
    # only one being measured.
    monkeypatch.setattr(
        LlamaCppBackend, "probe_server_capabilities", staticmethod(lambda *a, **k: {})
    )
    _auto_tp(backend, gguf, visible, tensor_split = [3, 1])

    def _intent(split):
        return GgufLoadIntent(
            gguf_path = str(gguf),
            model_identifier = "test",
            gpu_memory_mode = "auto",
            tensor_parallel = True,
            tensor_split = split,
            gpu_ids = [idx for idx, *_ in visible],
            n_ctx = 4096,
        )

    assert backend.adopt_load_intent_if_matched(_intent([3, 1])) is True
    assert backend.adopt_load_intent_if_matched(_intent([1, 3])) is False
