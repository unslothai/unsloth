# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""_widen_pin_ids_for_companion_devices (issue #11810).

A single-GPU pin sets CUDA_VISIBLE_DEVICES to the main model's cards, so user extra
args like --mmproj-device CUDA1 name a GPU the child cannot see and llama.cpp rejects
the flag at argument parsing as "invalid device" -- then Studio's fit/retry chain
burns four attempts on the same masked environment blaming fit and the drafter.
The pin is a MAIN-model placement constraint only, so the mask grows to cover the
companion GPUs, the companion flags are renumbered to the child's positions, and a
--device keeps the main model on its original cards.
"""

import pytest

import core.inference.llama_cpp as llama_cpp


def _widen(
    pin_ids,
    cmd,
    inherited = None,
    may_widen = True,
):
    cmd = list(cmd)
    widened, note = llama_cpp._widen_pin_ids_for_companion_devices(
        cmd, pin_ids, inherited, may_widen = may_widen
    )
    return cmd, widened, note


def _value(cmd, flag):
    return cmd[len(cmd) - 1 - cmd[::-1].index(flag) + 1]


def test_a_hidden_companion_device_widens_the_mask():
    cmd, pin, note = _widen(
        [0],
        ["--mmproj-offload", "--mmproj-device", "CUDA1", "--spec-draft-device", "CUDA1"],
    )
    assert pin == [0, 1]
    assert _value(cmd, "--device") == "CUDA0", "the main model keeps GPU 0 after the widening"
    assert _value(cmd, "--mmproj-device") == "CUDA1"
    assert note


def test_a_companion_on_a_lower_card_is_renumbered_not_left_on_the_main_card():
    # Main pinned to GPU 1, projector on GPU 0: the child mask is "1,0", where CUDA0 is
    # the MAIN card, so leaving the user's token alone would stack both on GPU 1.
    cmd, pin, _ = _widen([1], ["--mmproj-device", "CUDA0"])
    assert pin == [1, 0]
    assert _value(cmd, "--mmproj-device") == "CUDA1"
    assert _value(cmd, "--device") == "CUDA0"


def test_a_companion_on_the_pinned_card_is_renumbered_to_the_childs_first_device():
    # Pinned to GPU 1 alone, the child sees one device, CUDA0; the user's CUDA1 is invalid.
    cmd, pin, note = _widen([1], ["--mmproj-device", "CUDA1"])
    assert pin == [1]
    assert _value(cmd, "--mmproj-device") == "CUDA0"
    assert "--device" not in cmd, "nothing was added to the mask"
    assert note


def test_the_remap_matches_the_visible_order():
    """With an inherited non-ascending main order, CUDA<n> means n in the WIDENED set."""
    cmd, pin, note = _widen([1, 0], ["--mmproj-device", "CUDA2"])
    assert pin == [1, 0, 2]
    assert _value(cmd, "--device") == "CUDA0,CUDA1"
    assert _value(cmd, "--mmproj-device") == "CUDA2"
    assert "2" in note


def test_tokens_are_read_through_the_inherited_mask():
    # A scheduler exposed physical 2 and 3; the user's CUDA0 is physical 2.
    cmd, pin, _ = _widen([3], ["--mmproj-device", "CUDA0"], inherited = [2, 3])
    assert pin == [3, 2]
    assert _value(cmd, "--mmproj-device") == "CUDA1"


def test_a_token_past_the_inherited_mask_never_widens_it():
    cmd, pin, note = _widen([3], ["--mmproj-device", "CUDA5"], inherited = [2, 3])
    assert pin == [3]
    assert cmd == ["--mmproj-device", "CUDA5"]
    assert note == ""


def test_no_device_flag_is_emitted_when_the_user_named_one():
    cmd, pin, note = _widen([0], ["--mmproj-device", "CUDA1", "--device", "CUDA0"])
    assert pin == [0, 1]
    assert cmd.count("--device") == 1, "the user's own --device survives untouched"
    assert note


def test_a_flag_already_on_the_childs_numbering_changes_nothing():
    cmd, pin, note = _widen([0, 1], ["--mmproj-device", "CUDA1"])
    assert pin == [0, 1]
    assert cmd == ["--mmproj-device", "CUDA1"]
    assert note == ""


def test_non_gpu_device_tokens_are_ignored():
    cmd, pin, note = _widen([0], ["--mmproj-device", "CPU", "--spec-draft-device", "none"])
    assert pin == [0]
    assert cmd == ["--mmproj-device", "CPU", "--spec-draft-device", "none"]
    assert note == ""


def test_a_stripped_flag_is_not_acted_on():
    # Explicit gpu_ids own placement and strip the user's device flags from the argv.
    cmd, pin, note = _widen([0], ["--ctx-size", "4096"])
    assert pin == [0]
    assert cmd == ["--ctx-size", "4096"]
    assert note == ""


@pytest.mark.parametrize("prefix", ["CUDA", "ROCm"])
def test_value_forms_the_parser_reads(prefix):
    """--flag=value inline spelling and comma lists both reach the helper."""
    cmd, pin, _ = _widen([0], [f"--mmproj-device={prefix}1"])
    assert pin == [0, 1]
    assert f"--mmproj-device={prefix}1" in cmd
    assert _value(cmd, "--device") == f"{prefix}0"

    cmd, pin, _ = _widen([1], ["--spec-draft-device", f"{prefix}0,{prefix}1"])
    assert pin == [1, 0]
    assert _value(cmd, "--spec-draft-device") == f"{prefix}1,{prefix}0"


def test_only_the_flag_that_wins_is_acted_on():
    cmd, pin, _ = _widen([0], ["--mmproj-device", "CUDA7", "-mmdev", "CUDA2"])
    assert pin == [0, 2], "an overridden flag must not expose another card"
    assert cmd[:2] == ["--mmproj-device", "CUDA7"]
    assert _value(cmd, "-mmdev") == "CUDA1"
    assert cmd.count("--device") == 1


def test_load_model_uses_the_argv_and_skips_an_unmappable_mask():
    import inspect

    src = inspect.getsource(llama_cpp.LlamaCppBackend.load_model)
    at = src.index("_widen_pin_ids_for_companion_devices(")
    window = src[at - 400 : at + 200]
    assert "_visibility_mask_is_unmappable()" in window
    assert 'env.get("LLAMA_ARG_DEVICE", "")' in window
    assert "may_widen = not gpu_ids" in src[at : at + 300]


def test_an_explicit_gpu_ids_pin_is_never_widened_onto_another_card():
    # gpu_ids is the pool the training guard budgeted: a companion outside it stays refused.
    cmd, pin, note = _widen([0], ["--mmproj-device", "CUDA1"], may_widen = False)
    assert pin == [0]
    assert cmd == ["--mmproj-device", "CUDA1"]
    assert note == ""


def test_an_explicit_pin_still_renumbers_a_companion_on_one_of_its_cards():
    cmd, pin, _ = _widen([2, 3], ["--mmproj-device", "CUDA3"], may_widen = False)
    assert pin == [2, 3]
    assert _value(cmd, "--mmproj-device") == "CUDA1"
    assert "--device" not in cmd


def test_an_arch_crash_retry_refits_the_companion_to_the_new_mask():
    # [0] widened to [0, 1] for the projector on CUDA1; GPU 0 then crashes and the
    # respawn is masked to physical 2. Read through the crashed launch's mask, the
    # projector still names physical 1, and the main model moves to physical 2.
    cmd, first, _ = _widen([0], ["--mmproj-device", "CUDA1"])
    assert first == [0, 1] and cmd[-2:] == ["--device", "CUDA0"]
    del cmd[-2:]  # the generated main --device, as the retry drops it
    retry, note = llama_cpp._widen_pin_ids_for_companion_devices(cmd, [2], first)
    assert retry == [2, 1]
    assert _value(cmd, "--mmproj-device") == "CUDA1"
    assert _value(cmd, "--device") == "CUDA0"
    assert note


def test_the_arch_crash_retry_refits_companions_in_source():
    import inspect

    src = inspect.getsource(llama_cpp.LlamaCppBackend.load_model)
    retry = src.index("_arch_crash_retry_gpu_ids(")
    window = src[retry : retry + 12000]
    assert "_companion_fit_mask" in window
    assert "list(_companion_fit_mask)" in window
    assert '",".join(str(i) for i in _retry_mask)' in window
