# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The child's GPU ORDER, which is separate from the set the picker chooses.

Two ways a user can ask for a specific order, and neither used to survive: a
reordered CUDA_VISIBLE_DEVICES was re-emitted ascending, and a pass-through
--device was stripped by a picker selection that narrowed nothing.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_llama_cpp_placement import _backend, _launch  # noqa: E402

from core.inference.llama_cpp import LlamaCppBackend  # noqa: E402

# One roomier card and one smaller one, so a weighted split is the interesting case.
_TWO_GPUS = [(0, 15_000, 16_000), (1, 11_000, 12_000)]


def _run(
    monkeypatch,
    tmp_path,
    *,
    mask,
    gpu_ids = None,
    extra_args = None,
):
    if mask is None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising = False)
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", mask)
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    backend, gguf = _backend(tmp_path, vulkan = False, memory = _TWO_GPUS)
    backend._select_gpus = lambda *args, **kwargs: ([0, 1], False)
    kwargs = {} if extra_args is None else {"extra_args": extra_args}
    result = _launch(backend, gguf, n_ctx = 4096, gpu_ids = gpu_ids, **kwargs)
    return backend, result


def _device_arg(cmd):
    return cmd[cmd.index("--device") + 1] if "--device" in cmd else None


def test_reordered_parent_mask_reaches_the_child(monkeypatch, tmp_path):
    """A numeric mask carries ORDER, not just membership, so it must survive."""
    _, result = _run(monkeypatch, tmp_path, mask = "1,0")
    assert result["env"]["CUDA_VISIBLE_DEVICES"] == "1,0"


def test_reordered_parent_mask_survives_a_gpu_pick(monkeypatch, tmp_path):
    """The picker owns the SET; the mask still owns the order within it."""
    backend, result = _run(monkeypatch, tmp_path, mask = "1,0", gpu_ids = [0, 1])
    assert result["env"]["CUDA_VISIBLE_DEVICES"] == "1,0"
    # The ordinal -> physical map the buffer parser reads has to match what we emitted.
    assert backend._child_gpu_physical_ids == (1, 0)


def test_ascending_parent_mask_is_left_alone(monkeypatch, tmp_path):
    _, result = _run(monkeypatch, tmp_path, mask = "0,1")
    assert result["env"]["CUDA_VISIBLE_DEVICES"] == "0,1"


def test_partial_mask_cannot_order_the_rest(monkeypatch, tmp_path):
    """A mask covering only some pinned ids says nothing about where the others go."""
    _, result = _run(monkeypatch, tmp_path, mask = "1")
    assert result["env"]["CUDA_VISIBLE_DEVICES"] == "0,1"


def test_tensor_split_shares_follow_the_reorder(monkeypatch, tmp_path):
    """--tensor-split is positional, so the shares move with the devices."""
    cmd = ["llama-server", "--tensor-split", "60,40", "--top-k", "5"]
    assert LlamaCppBackend._repoint_emitted_tensor_split(cmd, [0, 1], [1, 0]) is True
    assert cmd[cmd.index("--tensor-split") + 1] == "40,60"
    assert cmd[cmd.index("--top-k") + 1] == "5"


def test_tensor_split_repoint_declines_a_mismatched_width(monkeypatch, tmp_path):
    """Refusing vetoes the reorder; emitting the mask alone would be the same bug."""
    cmd = ["llama-server", "--tensor-split", "60"]
    assert LlamaCppBackend._repoint_emitted_tensor_split(cmd, [0, 1], [1, 0]) is False
    assert cmd[cmd.index("--tensor-split") + 1] == "60"


def test_user_tensor_split_keeps_ascending_order(monkeypatch, tmp_path):
    """Their shares are positional over the order they expected. Decline, don't rewrite."""
    _, result = _run(monkeypatch, tmp_path, mask = "1,0", extra_args = ["--tensor-split", "60,40"])
    assert result["env"]["CUDA_VISIBLE_DEVICES"] == "0,1"
    assert result["cmd"][result["cmd"].index("--tensor-split") + 1] == "60,40"


def test_pick_that_keeps_every_gpu_leaves_device_flags_alone(monkeypatch, tmp_path):
    """Selecting everything narrows nothing, so there is no conflict to resolve."""
    _, result = _run(
        monkeypatch,
        tmp_path,
        mask = "0,1",
        gpu_ids = [0, 1],
        extra_args = ["--device", "CUDA1,CUDA0"],
    )
    assert _device_arg(result["cmd"]) == "CUDA1,CUDA0"


def test_narrowing_pick_still_owns_device_flags(monkeypatch, tmp_path):
    """A --device naming a deselected card must not outlive the picker."""
    backend, gguf = _backend(tmp_path, vulkan = False, memory = _TWO_GPUS)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    backend._select_gpus = lambda *args, **kwargs: ([1], False)
    result = _launch(
        backend,
        gguf,
        n_ctx = 4096,
        gpu_ids = [1],
        extra_args = ["--device", "CUDA0", "--top-k", "5"],
    )
    assert _device_arg(result["cmd"]) is None
    assert result["cmd"][result["cmd"].index("--top-k") + 1] == "5"


def test_vulkan_pick_always_owns_device_flags(monkeypatch, tmp_path):
    """Vulkan ordinals are not the CUDA ids the visible set is counted in."""
    backend, _ = _backend(tmp_path, vulkan = True, memory = _TWO_GPUS)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    assert backend._gpu_ids_own_placement([0, 1], is_vulkan = True) is True
    assert backend._gpu_ids_own_placement([0, 1], is_vulkan = False) is False
    assert backend._gpu_ids_own_placement(None, is_vulkan = False) is False
