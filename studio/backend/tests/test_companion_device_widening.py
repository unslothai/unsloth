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
    # child mask "1,0": an untouched CUDA0 would stack the projector on the main card
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


def test_every_parsed_device_flag_remains_resolvable():
    # arg.cpp resolves each device value as it reads it (parse_device_list), so an
    # overridden one must still name a card the child can see.
    cmd, pin, _ = _widen([0], ["--mmproj-device", "CUDA7", "-mmdev", "CUDA2"])
    assert pin == [0, 7, 2]
    assert cmd[:2] == ["--mmproj-device", "CUDA1"]
    assert _value(cmd, "-mmdev") == "CUDA2"
    assert cmd.count("--device") == 1


@pytest.mark.parametrize(
    "flags",
    [
        ["--mmproj-device", "CUDA1", "--mmproj-device", "none"],
        ["-mmdev=CUDA1", "--mmproj-device", "none"],
        ["--spec-draft-device", "CUDA1", "-devd", "none"],
    ],
)
def test_a_device_overridden_by_none_still_widens_the_mask(flags):
    # Pinned to GPU 0 because the final none keeps the companion off the cards, but
    # the earlier CUDA1 is still parsed: a mask of 0 fails it with "invalid device".
    cmd, pin, note = _widen([0], flags, inherited = [0, 1])
    assert pin == [0, 1]
    assert note
    assert _value(cmd, "--device") == "CUDA0"
    assert cmd[-4:-2] == flags[-2:], "the winning none is left alone"


def test_an_overridden_device_the_parent_cannot_see_is_left_alone():
    cmd, pin, note = _widen([0], ["--mmproj-device", "CUDA5", "--mmproj-device", "none"], [0, 1])
    assert (pin, note) == ([0], "")
    assert "--device" not in cmd


def test_load_model_uses_the_argv_and_skips_an_unmappable_mask():
    import inspect

    src = inspect.getsource(llama_cpp.LlamaCppBackend.load_model)
    at = src.index("_widen_pin_ids_for_companion_devices(")
    window = src[at - 400 : at + 200]
    assert "_visibility_mask_is_unmappable()" in window
    # Every device name the child parses, env twins included: no LLAMA_ARG_DEVICE skip.
    assert 'env.get("LLAMA_ARG_DEVICE", "")' not in window
    assert "allowed_ids = gpu_ids or None" in src[at : at + 300]
    assert "env = env" in src[at : at + 300]
    retry = src.index("_widen_pin_ids_for_companion_devices(", at + 1)
    assert "env = env" in src[retry : retry + 300]


def test_a_widened_explicit_pin_is_what_status_and_dedupe_see():
    # #12467: /status echoed the pre-widen pin [0], so a replayed load lost GPU 1.
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
    # #12467: gpu_ids [0, 1] saved, the fit chose [0], the projector names GPU 1
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
    # [0] widened to [0, 1]; GPU 0 crashes, respawn on 2: projector stays on physical 1
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


# --- Every device name the child parses (r11): env twins and overridden occurrences ---

_DEVICE_OPTION_NAMES = {
    # common/arg.cpp: every option resolved through parse_device_list, with its env twin
    "--device": "LLAMA_ARG_DEVICE",
    "-dev": "LLAMA_ARG_DEVICE",
    "--mmproj-device": "MTMD_BACKEND_DEVICE",
    "-mmdev": "MTMD_BACKEND_DEVICE",
    "--spec-draft-device": None,
    "-devd": None,
    "--device-draft": None,
}


def _unresolvable(cmd, env, mask_len):
    """Device names llama.cpp would reject in a child seeing ``mask_len`` cards.

    Written independently of the helper under test: env twins first, then every argv
    occurrence; a value of exactly "none" is skipped (the -mmdev handler and
    parse_device_list both special-case it)."""
    values = [env[name] for name in ("LLAMA_ARG_DEVICE", "MTMD_BACKEND_DEVICE") if name in env]
    for i, raw in enumerate(cmd):
        head, eq, inline = raw.partition("=")
        if head in _DEVICE_OPTION_NAMES:
            values.append(inline if eq else cmd[i + 1])
    bad = []
    for value in values:
        if value == "none":
            continue
        for name in value.split(","):
            match = llama_cpp._GPU_DEVICE_TOKEN_RE.match(name.strip())
            if not match or int(match.group(2)) >= mask_len:
                bad.append(name)
    return bad


def _widen_env(
    pin_ids,
    cmd,
    env,
    inherited = None,
    allowed_ids = None,
):
    cmd, env = list(cmd), dict(env)
    widened, note = llama_cpp._widen_pin_ids_for_companion_devices(
        cmd, pin_ids, inherited, allowed_ids = allowed_ids, env = env
    )
    return cmd, env, widened, note


# (argv, env): a GPU 1 name parsed BEFORE the value that places, with the pin on GPU 0.
_OVERRIDDEN_HIDDEN_CARD = [
    pytest.param(
        ["--mmproj-device", "none"], {"MTMD_BACKEND_DEVICE": "CUDA1"}, id = "env_mtmd-mmproj_none"
    ),
    pytest.param(["-mmdev", "none"], {"MTMD_BACKEND_DEVICE": "CUDA1"}, id = "env_mtmd-mmdev_none"),
    pytest.param(
        ["-mmdev=none"], {"MTMD_BACKEND_DEVICE": "ROCm1"}, id = "env_mtmd_rocm-mmdev_inline_none"
    ),
    pytest.param(
        ["--mmproj-device", "CUDA0"], {"MTMD_BACKEND_DEVICE": "CUDA1"}, id = "env_mtmd-mmproj_gpu"
    ),
    pytest.param(["--device", "CUDA0"], {"LLAMA_ARG_DEVICE": "CUDA1"}, id = "env_device-device"),
    pytest.param(["-dev", "CUDA0"], {"LLAMA_ARG_DEVICE": "CUDA1"}, id = "env_device-dev"),
    pytest.param(["--device", "CUDA1", "--device", "CUDA0"], {}, id = "device-device"),
    pytest.param(["-dev", "CUDA1", "--device=CUDA0"], {}, id = "dev-device_inline"),
    pytest.param(["--device=CUDA0,CUDA1", "-dev", "CUDA0"], {}, id = "device_list-dev"),
    pytest.param(["--device", "CUDA1", "--device", "none"], {}, id = "device-device_none"),
    # Fixed in r10 (controls for the same matrix): argv companions.
    pytest.param(["--mmproj-device", "CUDA1", "-mmdev", "none"], {}, id = "mmproj-mmdev_none"),
    pytest.param(["--spec-draft-device", "CUDA1", "-devd", "none"], {}, id = "draft-devd_none"),
    pytest.param(
        ["--device-draft", "CUDA1", "--spec-draft-device", "none"], {}, id = "device_draft-draft_none"
    ),
]


@pytest.mark.parametrize(("argv", "env"), _OVERRIDDEN_HIDDEN_CARD)
def test_every_parsed_device_name_stays_resolvable_under_the_pin(argv, env):
    cmd, child_env, pin, note = _widen_env([0], argv, env, inherited = [0, 1])
    assert pin == [0, 1], "GPU 1 is parsed, so the child must see it"
    assert note
    assert _unresolvable(cmd, child_env, len(pin)) == []
    # The value that places the main model is untouched or, absent, added on the pinned card.
    assert llama_cpp._extra_args_main_device(cmd) in (None, "CUDA0", "none", "ROCm0") or (
        child_env.get("LLAMA_ARG_DEVICE") is not None
    )
    final_main = llama_cpp._extra_args_main_device(cmd)
    if final_main is None:
        final_main = child_env.get("LLAMA_ARG_DEVICE")
    assert final_main in ("CUDA0", "ROCm0", "none")


@pytest.mark.parametrize(("argv", "env"), _OVERRIDDEN_HIDDEN_CARD)
def test_the_arch_crash_retry_refits_every_parsed_device_name(argv, env):
    # Widened [0] -> [0, 1]; GPU 0 crashes and the respawn runs on GPU 2. Each name is now
    # in the crashed launch's numbering, env twins included, and must resolve again.
    cmd, child_env, first, _ = _widen_env([0], argv, env, inherited = [0, 1, 2])
    assert first == [0, 1]
    if cmd[-2:] == ["--device", "CUDA0"] and "--device" not in argv[-2:]:
        del cmd[-2:]
    retry, _ = llama_cpp._widen_pin_ids_for_companion_devices(cmd, [2], first, env = child_env)
    assert retry[0] == 2 and 1 in retry, "the respawn card leads; GPU 1 is still parsed"
    assert _unresolvable(cmd, child_env, len(retry)) == []


def test_an_inherited_final_main_device_keeps_control_and_the_companion_card_widens():
    # LLAMA_ARG_DEVICE is the final main value here: the projector card still widens the
    # mask, and no --device is appended over the env twin.
    cmd, env, pin, note = _widen_env(
        [0], ["--mmproj-device", "CUDA1"], {"LLAMA_ARG_DEVICE": "CUDA0"}
    )
    assert pin == [0, 1] and note
    assert "--device" not in cmd
    assert env["LLAMA_ARG_DEVICE"] == "CUDA0"
    assert _unresolvable(cmd, env, len(pin)) == []


def test_an_inherited_main_device_past_the_pin_is_not_moved():
    cmd, env, pin, note = _widen_env(
        [0], ["--mmproj-device", "CUDA2"], {"LLAMA_ARG_DEVICE": "CUDA1"}
    )
    assert (pin, note) == ([0], "")
    assert cmd == ["--mmproj-device", "CUDA2"] and env == {"LLAMA_ARG_DEVICE": "CUDA1"}


def test_an_inherited_projector_device_is_renumbered_in_the_env():
    cmd, env, pin, _ = _widen_env(
        [1], ["--mmproj-device", "none"], {"MTMD_BACKEND_DEVICE": "CUDA1"}
    )
    assert pin == [1]
    assert env["MTMD_BACKEND_DEVICE"] == "CUDA0", "physical 1 is the child's CUDA0"
    assert "--device" not in cmd


@pytest.mark.parametrize(
    ("argv", "env"),
    [
        pytest.param(
            ["--mmproj-device", "none"], {"MTMD_BACKEND_DEVICE": "CUDA1"}, id = "env_mtmd-mmproj_none"
        ),
        pytest.param(
            ["--mmproj-device", "CUDA0"], {"MTMD_BACKEND_DEVICE": "CUDA1"}, id = "env_mtmd-mmproj_gpu"
        ),
        pytest.param(["--mmproj-device", "CUDA1", "-mmdev", "none"], {}, id = "mmproj-mmdev_none"),
        pytest.param(["-mmdev=CUDA1", "-mmdev", "CUDA0"], {}, id = "mmdev_inline-mmdev_gpu"),
        pytest.param(
            ["--spec-draft-device", "CUDA0,CUDA1", "-devd", "none"], {}, id = "draft_list-devd_none"
        ),
    ],
)
def test_explicit_gpu_ids_clear_an_overridden_name_they_exclude(argv, env):
    # gpu_ids [0]: the excluded GPU 1 is never added, and the overridden value that named
    # it places nothing, so it is cleared rather than left to fail the launch.
    cmd, child_env, pin, note = _widen_env([0], argv, env, inherited = [0, 1], allowed_ids = [0])
    assert pin == [0]
    assert note
    assert _unresolvable(cmd, child_env, 1) == []
    assert cmd[-2:] == argv[-2:] or cmd[-1] == argv[-1], "the value that places is untouched"


def test_explicit_gpu_ids_leave_a_final_excluded_name_alone():
    cmd, env, pin, note = _widen_env([0], ["-mmdev", "none"], {}, allowed_ids = [0])
    assert (pin, note, cmd) == ([0], "", ["-mmdev", "none"])
    cmd, env, pin, note = _widen_env([0], [], {"MTMD_BACKEND_DEVICE": "CUDA1"}, [0, 1], [0])
    assert (pin, note, env) == ([0], "", {"MTMD_BACKEND_DEVICE": "CUDA1"})


@pytest.mark.parametrize(
    ("argv", "env", "mmproj", "main_or_draft"),
    [
        ([], {"MTMD_BACKEND_DEVICE": "CUDA0"}, True, False),
        (["--mmproj-device", "none"], {"MTMD_BACKEND_DEVICE": "CUDA0"}, True, False),
        (["-mmdev", "CUDA0", "--mmproj-device", "none"], {}, True, False),
        ([], {"LLAMA_ARG_DEVICE": "CUDA0"}, False, True),
        (["--device", "none"], {"LLAMA_ARG_DEVICE": "CUDA0"}, False, True),
        (["-dev", "CUDA0", "--device", "none"], {}, False, True),
        (["--device-draft", "CUDA0", "-devd", "none"], {}, False, True),
        (["--device", "none", "-mmdev", "none", "-devd", "none"], {}, False, False),
        ([], {"MTMD_BACKEND_DEVICE": "none", "LLAMA_ARG_DEVICE": "none"}, False, False),
    ],
)
def test_the_zero_offload_mask_reads_every_parsed_device_name(argv, env, mmproj, main_or_draft):
    cmd = ["llama-server", "-m", "model.gguf", "--gpu-layers", "0", *argv]
    backend = llama_cpp.LlamaCppBackend
    assert backend._cmd_names_mmproj_device(cmd, env) is mmproj
    assert backend._cmd_has_gpu_device_pin(cmd, env) is main_or_draft
    assert backend._zero_offload_keeps_gpu_visible(cmd, env) is (mmproj or main_or_draft)


def test_device_list_sites_follow_the_parsers_order():
    sites = llama_cpp._device_list_sites(
        ["-mmdev", "CUDA1", "--device=CUDA0", "-devd", "none", "--mmproj-device", "none", "-dev"],
        {"MTMD_BACKEND_DEVICE": "CUDA2", "LLAMA_ARG_DEVICE": "CUDA3"},
    )
    assert [(s.group, s.env_name, s.value, s.final) for s in sites] == [
        ("main", "LLAMA_ARG_DEVICE", "CUDA3", False),
        ("mmproj", "MTMD_BACKEND_DEVICE", "CUDA2", False),
        ("mmproj", None, "CUDA1", False),
        ("main", None, "CUDA0", False),
        ("draft", None, "none", True),
        ("mmproj", None, "none", True),
        ("main", None, "", True),
    ]
    # Every spelling arg.cpp declares for the three device-list options.
    for flag, twin in _DEVICE_OPTION_NAMES.items():
        (site,) = llama_cpp._device_list_sites([flag, "CUDA0"])
        assert site.value == "CUDA0"
        owner = next(o for o in llama_cpp._DEVICE_LIST_OPTIONS if flag in o[1])
        assert owner[2] == twin
