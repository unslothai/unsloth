# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""_widen_pin_ids_for_companion_devices (#11810): --mmproj-device / --spec-draft-device
naming a card the pin hides was "invalid device"; the mask now widens to cover it."""

import pytest

import core.inference.llama_cpp as llama_cpp


def _widen(
    pin_ids,
    cmd,
    inherited = None,
    allowed_ids = None,
):
    cmd = list(cmd)
    widened, note = llama_cpp._widen_pin_ids_for_companion_devices(
        cmd, pin_ids, inherited, allowed_ids = allowed_ids
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
    cmd, pin, _ = _widen([1], ["--mmproj-device", "CUDA0"])
    assert pin == [1, 0]
    assert _value(cmd, "--mmproj-device") == "CUDA1"
    assert _value(cmd, "--device") == "CUDA0"


def test_a_companion_on_the_pinned_card_is_renumbered_to_the_childs_first_device():
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


def test_a_main_device_past_the_pin_is_not_moved_onto_a_companion_card():
    cmd, pin, note = _widen([0], ["--device", "CUDA1", "--mmproj-device", "CUDA2"])
    assert pin == [0]
    assert cmd == ["--device", "CUDA1", "--mmproj-device", "CUDA2"]
    assert note == ""


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
    assert "allowed_ids = gpu_ids or None" in src[at : at + 300]


def test_a_widened_explicit_pin_is_what_status_and_dedupe_see():
    import inspect

    src = inspect.getsource(llama_cpp.LlamaCppBackend.load_model)
    at = src.index("_widen_pin_ids_for_companion_devices(")
    window = src[at : at + 900]
    assert "if gpu_ids:" in window
    assert "self._adopt_widened_pin(_pin_ids)" in window

    backend = llama_cpp.LlamaCppBackend.__new__(llama_cpp.LlamaCppBackend)
    backend._is_diffusion = False
    backend._requested_gpu_ids, backend._gpu_ids = [0, 1], [0]
    backend._adopt_widened_pin([0, 1])
    assert backend._gpu_ids == [0, 1]
    assert backend.matches_gpu_ids([0, 1]) is True
    assert backend.matches_gpu_ids([0]) is False

    backend._gpu_ids = None
    backend._adopt_widened_pin([0, 1])
    assert backend._gpu_ids is None


def test_explicit_gpu_ids_never_widen_onto_an_unselected_card():
    cmd, pin, note = _widen([0], ["--mmproj-device", "CUDA1"], allowed_ids = [0])
    assert pin == [0]
    assert cmd == ["--mmproj-device", "CUDA1"]
    assert note == ""


def test_explicit_gpu_ids_widen_onto_a_selected_card_the_fit_left_out():
    cmd, pin, note = _widen([0], ["--mmproj-device", "CUDA1"], allowed_ids = [0, 1])
    assert pin == [0, 1]
    assert _value(cmd, "--mmproj-device") == "CUDA1"
    assert cmd[-2:] == ["--device", "CUDA0"]
    assert note


def test_an_explicit_pin_still_renumbers_a_companion_on_one_of_its_cards():
    cmd, pin, _ = _widen([2, 3], ["--mmproj-device", "CUDA3"], allowed_ids = [2, 3])
    assert pin == [2, 3]
    assert _value(cmd, "--mmproj-device") == "CUDA1"
    assert "--device" not in cmd


def test_an_arch_crash_retry_refits_the_companion_to_the_new_mask():
    cmd, first, _ = _widen([0], ["--mmproj-device", "CUDA1"])
    assert first == [0, 1] and cmd[-2:] == ["--device", "CUDA0"]
    del cmd[-2:]
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
