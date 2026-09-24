# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""_widen_pin_ids_for_companion_devices (issue #11810).

A single-GPU pin sets CUDA_VISIBLE_DEVICES to the main model's cards, so user extra
args like --mmproj-device CUDA1 name a GPU the child cannot see and llama.cpp rejects
the flag at argument parsing as "invalid device" -- then Studio's fit/retry chain
burns four attempts on the same masked environment blaming fit and the drafter.
The pin is a MAIN-model placement constraint only, so the mask grows to cover the
companion GPUs, and a --device remap keeps the main model on its original cards.
"""

import core.inference.llama_cpp as llama_cpp


def _widen(pin_ids, extra_args, cmd=None):
    cmd = list(cmd or [])
    widened, note = llama_cpp._widen_pin_ids_for_companion_devices(cmd, pin_ids, extra_args)
    return cmd, widened, note


def test_a_hidden_companion_device_widens_the_mask():
    cmd, pin, note = _widen(
        [0],
        ["--mmproj-offload", "--mmproj-device", "CUDA1", "--spec-draft-device", "CUDA1"],
    )
    assert pin == [0, 1]
    assert "CUDA0" in cmd, "the main model keeps GPU 0 after the widening"
    assert "--device" in cmd
    assert note


def test_the_remap_matches_the_visible_order(pin_ids=None):
    """With an inherited non-ascending main order, CUDA<n> means n in the WIDENED set."""
    cmd, pin, note = _widen([1, 0], ["--mmproj-device", "CUDA2"])
    # main cards [1, 0] sit at indices 0 and 1 of [1, 0, 2].
    assert pin == [1, 0, 2]
    assert cmd[cmd.index("--device") + 1] == "CUDA0,CUDA1"
    assert "2" in note


def test_no_device_flag_is_emitted_when_the_user_named_one():
    cmd, pin, note = _widen([0], ["--mmproj-device", "CUDA1", "--device", "CUDA0"])
    assert pin == [0, 1]
    assert "--device" not in cmd, "the user's own --device survives untouched"
    assert note


def test_a_flag_already_inside_the_pin_changes_nothing():
    cmd, pin, note = _widen([0, 1], ["--mmproj-device", "CUDA1"])
    assert pin == [0, 1]
    assert cmd == []
    assert note == ""


def test_non_cuda_device_tokens_are_ignored():
    cmd, pin, note = _widen([0], ["--mmproj-device", "CPU", "--spec-draft-device", "none"])
    assert pin == [0]
    assert cmd == []
    assert note == ""


def test_value_forms_the_parser_reads():
    """--flag=value inline spelling and comma lists both reach the helper."""
    cmd, pin, _ = _widen([0], ["--mmproj-device=CUDA1"])
    assert pin == [0, 1]

    cmd, pin, _ = _widen([0], ["--spec-draft-device", "CUDA0,CUDA1"])
    assert pin == [0, 1]


def test_layered_flags_use_the_last_value():
    cmd, pin, _ = _widen([0], ["--mmproj-device", "CUDA7", "--mmproj-device", "CUDA2"])
    assert pin == [0, 2]
    assert "--device" in cmd and "--device" not in cmd[cmd.index("--device") + 1 :]
