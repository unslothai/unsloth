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
    perm = ["--device", "CUDA1,CUDA0"]
    assert backend._gpu_ids_own_placement([0, 1], is_vulkan = True, extra_args = perm) is True
    assert backend._gpu_ids_own_placement([0, 1], is_vulkan = False, extra_args = perm) is False
    assert backend._gpu_ids_own_placement(None, is_vulkan = False) is False
    # Nothing to pass through is not a reason to relinquish placement.
    assert backend._gpu_ids_own_placement([0, 1], is_vulkan = False) is True


def test_a_narrowing_device_flag_is_not_a_reorder(monkeypatch, tmp_path):
    """--device CUDA0 of two contradicts a plan budgeted across both cards.

    The pick covers the visible set either way, so the predicate has to read the
    device VALUE: a permutation is safe to keep, a narrowing value is not.
    """
    _, narrowed = _run(
        monkeypatch,
        tmp_path,
        mask = "0,1",
        gpu_ids = [0, 1],
        extra_args = ["--device", "CUDA0"],
    )
    assert _device_arg(narrowed["cmd"]) is None


def test_an_inherited_device_env_is_scrubbed_unless_it_reorders(monkeypatch, tmp_path):
    """The same guard gates the env scrub, and llama.cpp reads the env first, so a
    narrowing LLAMA_ARG_DEVICE reaches the child with no user argv at all."""
    monkeypatch.setenv("LLAMA_ARG_DEVICE", "CUDA0")
    _, result = _run(monkeypatch, tmp_path, mask = "0,1", gpu_ids = [0, 1])
    assert result["env"].get("LLAMA_ARG_DEVICE") is None


def test_a_stale_pick_naming_an_absent_card_still_owns_placement(monkeypatch, tmp_path):
    """A pick is a superset of the visible set only when it is out of date."""
    backend, _ = _backend(tmp_path, vulkan = False, memory = _TWO_GPUS)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    assert (
        backend._gpu_ids_own_placement(
            [0, 1, 2], is_vulkan = False, extra_args = ["--device", "CUDA1,CUDA0"]
        )
        is True
    )


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
    result = _launch(backend, gguf, n_ctx = 4096, gpu_ids = [0, 1], tensor_parallel = True)
    assert result["env"].get("LLAMA_ARG_TENSOR_SPLIT") is None
    assert result["env"]["CUDA_VISIBLE_DEVICES"] == "1,0"


def test_a_split_the_child_does_inherit_still_vetoes(monkeypatch, tmp_path):
    """The control: the child receives it here, so it is positional over the order
    the user expected and the reorder must decline."""
    monkeypatch.setenv("LLAMA_ARG_TENSOR_SPLIT", "60,40")
    _, result = _run(monkeypatch, tmp_path, mask = "1,0")
    assert result["env"].get("LLAMA_ARG_TENSOR_SPLIT") == "60,40"
    assert result["env"]["CUDA_VISIBLE_DEVICES"] == "0,1"


def test_a_reorder_is_read_in_the_childs_own_ordinals(monkeypatch, tmp_path):
    """A non-zero mask is where physical ids and llama.cpp device names diverge.

    Under CUDA_VISIBLE_DEVICES=2,3 the child has CUDA0 and CUDA1. Comparing the
    device value against the physical ids stripped the real reorder and kept a
    pair naming devices the child does not have.
    """
    backend, _ = _backend(tmp_path, vulkan = False, memory = _TWO_GPUS)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
    own = backend._gpu_ids_own_placement
    # The real reorder survives...
    assert own([2, 3], is_vulkan = False, extra_args = ["--device", "CUDA1,CUDA0"]) is False
    # ...and a value naming devices the child will not have does not.
    assert own([2, 3], is_vulkan = False, extra_args = ["--device", "CUDA2,CUDA3"]) is True
    # A narrowing value is still stripped.
    assert own([2, 3], is_vulkan = False, extra_args = ["--device", "CUDA0"]) is True


def test_the_reload_comparator_asks_what_the_launch_asked(monkeypatch, tmp_path):
    """Re-sending an unchanged intent must not reload.

    The launch keeps a permuting --device, so a comparator that strips it compares
    a stripped list against the preserved one and rejects the live server on every
    unchanged Apply.
    """
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    backend, gguf = _backend(tmp_path, vulkan = False, memory = _TWO_GPUS)
    backend._select_gpus = lambda *args, **kwargs: ([0, 1], False)
    extras = ["--device", "CUDA1,CUDA0"]
    _launch(backend, gguf, n_ctx = 4096, gpu_ids = [0, 1], extra_args = extras)
    assert backend._gpu_ids_own_placement([0, 1], is_vulkan = False, extra_args = extras) is False
    # What the comparator would compare, with the flag preserved on both sides.
    assert backend._strip_device_extra_args(extras) != list(extras)
    assert backend._requested_extra_args == list(
        extras
    ), f"the launch stored a stripped list: {backend._requested_extra_args}"


def test_a_duplicate_device_list_is_not_a_permutation(monkeypatch, tmp_path):
    """llama.cpp keeps duplicate entries, so CUDA0,CUDA0,CUDA1 is a three-device
    list a two-entry split would be spread across. Arity and uniqueness, not just
    membership."""
    backend, _ = _backend(tmp_path, vulkan = False, memory = _TWO_GPUS)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    own = backend._gpu_ids_own_placement
    assert own([0, 1], is_vulkan = False, extra_args = ["--device", "CUDA0,CUDA0,CUDA1"]) is True
    assert own([0, 1], is_vulkan = False, extra_args = ["--device", "CUDA1,CUDA0"]) is False


def test_a_preserved_device_reorder_moves_the_planned_split(monkeypatch, tmp_path):
    """llama.cpp applies --tensor-split positionally over the SELECTED device list,
    so preserving a reorder without moving the shares gives the roomier card's share
    to the smaller one -- the same defect the inherited-mask path repoints for."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    backend, gguf = _backend(tmp_path, vulkan = False, memory = _TWO_GPUS)
    backend._get_gguf_size_bytes = lambda _path: 14 * 1024**3
    backend._select_gpus = lambda *args, **kwargs: ([0, 1], False)
    result = _launch(
        backend,
        gguf,
        n_ctx = 4096,
        gpu_ids = [0, 1],
        tensor_parallel = True,
        extra_args = ["--device", "CUDA1,CUDA0"],
    )
    cmd = result["cmd"]
    assert _device_arg(cmd) == "CUDA1,CUDA0"
    shares = cmd[cmd.index("--tensor-split") + 1].split(",")
    # Planned ascending as the bigger card first; the reorder puts it second.
    assert int(shares[0]) < int(
        shares[1]
    ), f"the planned shares did not follow the device reorder: {shares}"
