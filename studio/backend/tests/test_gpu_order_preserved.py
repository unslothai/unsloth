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


def test_an_explicit_pick_outranks_the_inherited_mask(monkeypatch, tmp_path):
    """Both name an order. The picker is the more explicit and more recent one."""
    backend, result = _run(monkeypatch, tmp_path, mask = "1,0", gpu_ids = [0, 1])
    assert result["env"]["CUDA_VISIBLE_DEVICES"] == "0,1"
    # The ordinal -> physical map the buffer parser reads has to match what we emitted.
    assert backend._child_gpu_physical_ids == (0, 1)


def test_the_picked_order_is_the_child_order(monkeypatch, tmp_path):
    backend, result = _run(monkeypatch, tmp_path, mask = None, gpu_ids = [1, 0])
    assert result["env"]["CUDA_VISIBLE_DEVICES"] == "1,0"
    assert backend._child_gpu_physical_ids == (1, 0)
    assert backend.requested_gpu_ids == [1, 0]


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


def test_an_unparseable_user_split_still_vetoes_the_reorder(monkeypatch, tmp_path):
    """The veto is presence, not a successful parse: the repointer does no numeric
    validation and takes the LAST --tensor-split, which is the user's."""
    _, result = _run(monkeypatch, tmp_path, mask = "1,0", extra_args = ["--tensor-split", "3x,1"])
    cmd = result["cmd"]
    assert result["env"]["CUDA_VISIBLE_DEVICES"] == "0,1"
    assert cmd[len(cmd) - 1 - cmd[::-1].index("--tensor-split") + 1] == "3x,1"


def test_a_split_scrubbed_from_the_child_does_not_veto(monkeypatch, tmp_path):
    """A tensor-parallel launch clears LLAMA_ARG_TENSOR_SPLIT from the child, so
    reading os.environ let a value the child never receives suppress the reorder.

    Off this path the child DOES inherit it, and the veto is right to fire; that is
    the control below.
    """
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,0")
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    monkeypatch.setenv("LLAMA_ARG_TENSOR_SPLIT", "60,40")
    backend, gguf = _backend(tmp_path, vulkan = False, memory = _TWO_GPUS)
    backend._get_gguf_size_bytes = lambda _path: 14 * 1024**3
    backend._select_gpus = lambda *args, **kwargs: ([0, 1], False)
    # No explicit pick: here the picker outranks the inherited mask, so only an
    # unpicked load reaches the mask reorder this cell is about.
    result = _launch(backend, gguf, n_ctx = 4096, tensor_parallel = True)
    assert result["env"].get("LLAMA_ARG_TENSOR_SPLIT") is None
    assert result["env"]["CUDA_VISIBLE_DEVICES"] == "1,0"


def test_a_split_the_child_does_inherit_still_vetoes(monkeypatch, tmp_path):
    """The control: the child receives it here, so it is positional over the order
    the user expected and the reorder must decline."""
    monkeypatch.setenv("LLAMA_ARG_TENSOR_SPLIT", "60,40")
    _, result = _run(monkeypatch, tmp_path, mask = "1,0")
    assert result["env"].get("LLAMA_ARG_TENSOR_SPLIT") == "60,40"
    assert result["env"]["CUDA_VISIBLE_DEVICES"] == "0,1"


def test_an_explicit_pick_always_owns_device_flags(monkeypatch, tmp_path):
    """Withdrawn: a pick covering the whole visible set used to relinquish device
    flags so a pass-through --device could order the cards. The mask reorder above
    already does that, and the pass-through cost a recomputation every consumer of
    the strip had to agree about, so an explicit pick owns placement again.
    """
    backend, _ = _backend(tmp_path, vulkan = False, memory = _TWO_GPUS)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    own = backend._gpu_ids_own_placement
    assert own([0, 1]) is True
    assert own([1]) is True
    assert own(None) is False


def test_the_authoritative_effective_pin_keeps_the_picked_order(monkeypatch, tmp_path):
    """/status serves _gpu_ids, and two blocks assign it: the later one wins.

    Sorting in either put the order back, so a client round-tripping the effective
    value matched the stored pin and skipped a reload the child needed.
    """
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    backend, gguf = _backend(tmp_path, vulkan = False, memory = _TWO_GPUS)
    backend._select_gpus = lambda *args, **kwargs: ([1, 0], False)
    _launch(backend, gguf, n_ctx = 4096, gpu_ids = [1, 0])
    assert backend._gpu_ids == [1, 0], f"the effective pin was re-sorted: {backend._gpu_ids}"


def test_the_reported_split_follows_the_reorder(monkeypatch, tmp_path):
    """/status serves the recorded emitted split, which was captured before the
    reorder rewrote it, so each share was paired with the wrong visible device."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,0")
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    backend, gguf = _backend(tmp_path, vulkan = False, memory = _TWO_GPUS)
    backend._get_gguf_size_bytes = lambda _path: 14 * 1024**3
    backend._select_gpus = lambda *args, **kwargs: ([0, 1], False)
    # No explicit pick: in this branch an explicit pick outranks the inherited
    # mask, so only an unpicked load reaches the mask reorder at all.
    result = _launch(backend, gguf, n_ctx = 4096, tensor_parallel = True)
    cmd = result["cmd"]
    assert result["env"]["CUDA_VISIBLE_DEVICES"] == "1,0"
    in_argv = [float(x) for x in cmd[cmd.index("--tensor-split") + 1].split(",")]
    reported = backend.tensor_split
    assert reported is not None, "no split reported at all; this would be vacuous"
    assert [round(float(x), 3) for x in reported] == [
        round(v / sum(in_argv), 3) for v in in_argv
    ] or list(
        reported
    ) == in_argv, f"reported {reported} does not match the argv the child ran: {in_argv}"


def test_a_full_set_pick_strips_a_device_flag(monkeypatch, tmp_path):
    """Both spellings, including the env twin, which needs no user argv at all."""
    monkeypatch.setenv("LLAMA_ARG_DEVICE", "CUDA0")
    _, result = _run(
        monkeypatch,
        tmp_path,
        mask = "0,1",
        gpu_ids = [0, 1],
        extra_args = ["--device", "CUDA1,CUDA0"],
    )
    assert _device_arg(result["cmd"]) is None
    assert result["env"].get("LLAMA_ARG_DEVICE") is None


def test_the_picker_is_how_a_full_pick_orders_its_cards(monkeypatch, tmp_path):
    """The ordering route the withdrawn pass-through was meant to preserve.

    In this branch the picker carries the order and outranks the inherited mask,
    so the pick is what the child enumerates by.
    """
    backend, result = _run(monkeypatch, tmp_path, mask = "0,1", gpu_ids = [1, 0])
    assert result["env"]["CUDA_VISIBLE_DEVICES"] == "1,0"
    assert backend._child_gpu_physical_ids == (1, 0)
