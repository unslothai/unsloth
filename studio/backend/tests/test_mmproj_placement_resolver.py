# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Every projector-placement consumer reads one ordered resolver.

llama.cpp (common/arg.cpp) applies the env twins first, then argv left to right, last
wins: ``--[no-]mmproj-offload`` sets ``mmproj_use_gpu``; ``--mmproj-device none`` /
``-mmdev none`` clears it; any other device name SETS it and pins that device.
``MTMD_BACKEND_DEVICE`` is the device option's env twin. Studio's own CPU pin
(``--no-mmproj-offload``) is appended after the pass-through extras, so it is the
final token. The budget, the automatic pin gate, the GPU-residency reader, the
CPU-fallback eligibility and the companion detector each used to read a different
subset of these.

Launch fixture (``test_mmproj_placement_policy._backend``): 6 GiB model, 1 GiB
projector, Auto context over a 262144 native length.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import core.inference.llama_cpp as llama_cpp_module  # noqa: E402
from core.inference.llama_cpp import (  # noqa: E402
    GgufLoadIntent,
    LlamaCppBackend,
    _AUTO_OFFLOAD_CTX,
)
from test_mmproj_placement_policy import _backend, _launch  # noqa: E402

_PLACEMENT_ENV = (
    "LLAMA_ARG_MMPROJ_OFFLOAD",
    "LLAMA_ARG_NO_MMPROJ_OFFLOAD",
    "MTMD_BACKEND_DEVICE",
    "LLAMA_ARG_MMPROJ",
    "LLAMA_ARG_MMPROJ_URL",
)


def _set_env(monkeypatch, env):
    for name in _PLACEMENT_ENV:
        monkeypatch.delenv(name, raising = False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)


# --- the initial VRAM budget ----------------------------------------------------------


@pytest.mark.parametrize(
    "extra_args, env",
    [
        (["--no-mmproj-offload", "--mmproj-device", "CUDA0"], {}),
        (["--no-mmproj-offload", "-mmdev=CUDA0"], {}),
        (["--mmproj-device", "CUDA0"], {"LLAMA_ARG_MMPROJ_OFFLOAD": "0"}),
        ([], {"LLAMA_ARG_MMPROJ_OFFLOAD": "0", "MTMD_BACKEND_DEVICE": "CUDA0"}),
    ],
    ids = ["argv_named_after_off", "short_named_after_off", "argv_named_over_env_off", "env_pair"],
)
def test_a_named_device_after_an_offload_off_is_charged_to_the_card(
    tmp_path, monkeypatch, extra_args, env
):
    """A named device turns the offload back on, so the projector is on a GPU and the
    budget pays for it like --mmproj-offload does (context shrinks to the offload
    fallback). Reading the earlier offload-off zeroed it and pinned a context the card
    cannot hold the projector beside."""
    _set_env(monkeypatch, env)
    backend, gguf = _backend(tmp_path, memory = [(0, 7_600, 8_192)])

    cmd = _launch(backend, gguf, extra_args = extra_args)["cmd"]

    assert "--mmproj" in cmd
    assert cmd[cmd.index("-c") + 1] == str(_AUTO_OFFLOAD_CTX), cmd


@pytest.mark.parametrize(
    "extra_args, env",
    [
        (["--mmproj-offload", "--mmproj-device", "none"], {}),
        (["--mmproj-device", "none"], {"LLAMA_ARG_MMPROJ_OFFLOAD": "1"}),
        (["--no-mmproj-offload"], {}),
    ],
    ids = ["argv_none_after_on", "argv_none_over_env_on", "no_offload"],
)
def test_a_projector_resolved_to_host_ram_is_not_charged(tmp_path, monkeypatch, extra_args, env):
    # Controls: device none clears the offload whatever came before it.
    _set_env(monkeypatch, env)
    backend, gguf = _backend(tmp_path, memory = [(0, 7_600, 8_192)])

    cmd = _launch(backend, gguf, extra_args = extra_args)["cmd"]

    assert cmd[cmd.index("-c") + 1] == "9984", cmd


# --- the automatic CPU pin --------------------------------------------------------------


@pytest.mark.parametrize(
    "extra_args, env",
    [
        (["--mmproj-device", "CUDA0"], {}),
        (["-mmdev=CUDA0"], {}),
        (["--mmproj_device", "CUDA0"], {}),
        ([], {"MTMD_BACKEND_DEVICE": "CUDA0"}),
    ],
    ids = ["argv_device", "short_inline", "underscore", "env_device"],
)
def test_an_explicit_projector_device_is_not_overridden_by_the_automatic_pin(
    tmp_path, monkeypatch, extra_args, env
):
    """A named device is the user owning the placement, as an offload spelling is: the
    trailing --no-mmproj-offload would beat it (last wins) and move the projector the
    user put on CUDA0 to the CPU."""
    _set_env(monkeypatch, env)
    backend, gguf = _backend(tmp_path, memory = [(0, 8_692, 16_384)])

    cmd = _launch(backend, gguf, extra_args = extra_args)["cmd"]

    assert "--mmproj" in cmd
    assert "--no-mmproj-offload" not in cmd, cmd


def test_the_automatic_pin_still_fires_when_nobody_placed_the_projector(tmp_path, monkeypatch):
    # Control: the same card with no placement option pins the projector.
    _set_env(monkeypatch, {})
    backend, gguf = _backend(tmp_path, memory = [(0, 8_692, 16_384)])

    cmd = _launch(backend, gguf)["cmd"]

    assert "--no-mmproj-offload" in cmd, cmd


# --- the ownership detectors ------------------------------------------------------------


@pytest.mark.parametrize(
    "extra_args, expected",
    [
        (["--mmproj-device", "CUDA0"], True),
        (["-mmdev", "none"], True),
        (["--mmproj-device=CUDA1"], True),
        (["--mmproj_device", "CUDA1"], True),
        (["--mmproj-offload"], True),
        (["--no-mmproj-offload"], True),
        (["--mmproj", "x.gguf"], False),
        ([], False),
    ],
)
def test_the_cli_detector_sees_the_device_option(extra_args, expected):
    assert llama_cpp_module._extra_args_set_mmproj_offload(extra_args) is expected


@pytest.mark.parametrize(
    "env, expected",
    [
        ({"MTMD_BACKEND_DEVICE": "CUDA1"}, True),
        ({"MTMD_BACKEND_DEVICE": "none"}, True),
        ({"LLAMA_ARG_MMPROJ_OFFLOAD": "1"}, True),
        ({"LLAMA_ARG_NO_MMPROJ_OFFLOAD": ""}, True),
        ({"LLAMA_ARG_MMPROJ": "x.gguf"}, False),
        ({}, False),
    ],
)
def test_the_env_detector_sees_the_device_twin(env, expected):
    assert llama_cpp_module._env_sets_mmproj_offload(env) is expected


# --- the boolean view and the GPU-residency reader --------------------------------------


@pytest.mark.parametrize(
    "extra_args, env, expected",
    [
        (["--no-mmproj-offload", "--mmproj-device", "CUDA0"], {}, True),
        (["--mmproj-device", "CUDA0"], {"LLAMA_ARG_MMPROJ_OFFLOAD": "0"}, True),
        (["--mmproj-offload", "--mmproj-device", "none"], {}, False),
        ([], {"LLAMA_ARG_MMPROJ_OFFLOAD": "1", "MTMD_BACKEND_DEVICE": "none"}, False),
        # arg.cpp throws on an unreadable env value before argv is read.
        (["--no-mmproj-offload"], {"LLAMA_ARG_MMPROJ_OFFLOAD": "maybe"}, None),
        ([], {}, None),
    ],
)
def test_the_resolved_offload_is_the_ordered_placement(extra_args, env, expected):
    assert llama_cpp_module._resolved_mmproj_offload(extra_args, env) is expected


@pytest.mark.parametrize(
    "extra_args, env, expected",
    [
        (["--no-mmproj-offload", "--mmproj-device", "CUDA0"], {}, True),
        (["--mmproj-device", "CUDA0"], {"LLAMA_ARG_MMPROJ_OFFLOAD": "0"}, True),
        (["--mmproj-offload", "--mmproj-device", "none"], {}, False),
        (["-mmdev=CUDA0", "-mmdev=none"], {}, False),
        ([], {"MTMD_BACKEND_DEVICE": "none"}, False),
        # Controls the legacy reader already answered.
        (["--no-mmproj-offload"], {}, False),
        ([], {}, True),
    ],
)
def test_the_gpu_residency_reader_replays_the_last_placement_token(extra_args, env, expected):
    cmd = ["/fake/llama-server", "-m", "m.gguf", "--mmproj", "p.gguf", *extra_args]
    assert llama_cpp_module._argv_keeps_projector_on_gpu(cmd, env) is expected


@pytest.mark.parametrize(
    "extra_args, env, expected",
    [
        (["--mmproj-device", "CUDA0"], {}, "host"),
        (["--mmproj-offload"], {"MTMD_BACKEND_DEVICE": "CUDA1"}, "host"),
        ([], {}, "host"),
        # Unreadable still throws first: the child never places anything.
        ([], {"LLAMA_ARG_MMPROJ_OFFLOAD": "maybe"}, None),
    ],
)
def test_studios_own_pin_is_the_final_placement_token(extra_args, env, expected):
    """The residency and load-mode guards price the projector Studio's appended
    --no-mmproj-offload moved to host RAM."""
    assert llama_cpp_module._mmproj_placement(extra_args, env, studio_cpu_pin = True) == expected


# --- the companion detector and CPU-fallback eligibility --------------------------------


@pytest.mark.parametrize(
    "extra_args, env, expected",
    [
        (["--no-mmproj-offload", "--mmproj-device", "CUDA0"], {}, True),
        (["--mmproj-offload", "--mmproj-device", "none"], {}, False),
        ([], {"LLAMA_ARG_MMPROJ_OFFLOAD": "0"}, False),
        # Controls.
        (["--no-mmproj-offload"], {}, False),
        ([], {}, True),
    ],
)
def test_the_gpu_companion_detector_reads_the_ordered_placement(extra_args, env, expected):
    cmd = ["/fake/llama-server", "-m", "m.gguf", "--mmproj", "p.gguf", *extra_args]
    assert LlamaCppBackend._cmd_has_gpu_companion(cmd, env) is expected


@pytest.mark.parametrize(
    "extra_args, env, expected",
    [
        (["--mmproj-device", "CUDA0"], {}, False),
        (["-mmdev=none"], {}, False),
        ([], {"MTMD_BACKEND_DEVICE": "CUDA0"}, False),
        ([], {"LLAMA_ARG_NO_MMPROJ_OFFLOAD": "1"}, False),
        # Controls.
        (["--no-mmproj-offload"], {}, False),
        ([], {"LLAMA_ARG_MMPROJ_OFFLOAD": "1"}, False),
        ([], {}, True),
    ],
)
def test_cpu_fallback_leaves_an_explicitly_placed_projector_alone(extra_args, env, expected):
    intent = GgufLoadIntent(model_identifier = "m", gguf_path = "m.gguf")
    assert LlamaCppBackend._cpu_fallback_request_eligible(intent, extra_args, env) is expected


# --- draft CPU-MoE counts parse like std::stoi ------------------------------------------


@pytest.mark.parametrize(
    "extra_args, env, expected",
    [
        # Negative: arg.cpp throws (value < 0), so nothing is priceable; abstain.
        (["--spec-draft-n-cpu-moe", "-1"], {}, True),
        (["-ncmoed=-3"], {}, True),
        ([], {"LLAMA_ARG_SPEC_DRAFT_N_CPU_MOE": "-1"}, True),
        # std::stoi skips leading whitespace and ignores trailing text.
        (["-ncmoed", " 2"], {}, True),
        ([], {"LLAMA_ARG_SPEC_DRAFT_N_CPU_MOE": "4abc"}, True),
        # Controls: zero places nothing; positive places experts on the CPU.
        (["-ncmoed", "0"], {}, False),
        ([], {"LLAMA_ARG_SPEC_DRAFT_N_CPU_MOE": " 0"}, False),
        (["-ncmoed", "2"], {}, True),
        # is_truthy is a raw comparison: a padded value is not truthy upstream.
        ([], {"LLAMA_ARG_SPEC_DRAFT_CPU_MOE": " 1"}, False),
        ([], {"LLAMA_ARG_SPEC_DRAFT_CPU_MOE": "1"}, True),
        ([], {}, False),
    ],
)
def test_draft_cpu_moe_counts_follow_upstream_parsing(extra_args, env, expected):
    assert llama_cpp_module._draft_tensors_placed_off_card(extra_args, env) is expected


# --- the one per-device --fit-target reader ---------------------------------------------


@pytest.mark.parametrize(
    "extra_args, env, one_model_device, expected",
    [
        (["--fit-target", "0,16384"], {}, True, (0.0, True)),
        ([], {"LLAMA_ARG_FIT_TARGET": "512/8192"}, True, (512.0, True)),
        (["--fit-target", "0,16384"], {}, False, (16384.0, False)),
        (["--fit-target", "2048"], {}, False, (2048.0, True)),
        (["--fit-target", "2048,2048"], {}, False, (2048.0, True)),
        ([], {}, True, (None, True)),
    ],
)
def test_the_fit_target_reader_is_exact_only_where_the_entry_is_provable(
    extra_args, env, one_model_device, expected
):
    assert (
        llama_cpp_module._child_fit_target_margin_mib(
            extra_args, env, one_model_device = one_model_device
        )
        == expected
    )


# --- target and draft layer / CPU-MoE counts parse like std::stoi ------------------------


@pytest.mark.parametrize(
    "extra_args, env, expected",
    [
        # std::stoi (common/arg.cpp handler_int) reads the leading digits and ignores the rest.
        (["-ncmoe", "4abc"], {}, True),
        (["--n-cpu-moe", "999 layers"], {}, True),
        (["-ncmoe=+2"], {}, True),
        ([], {"LLAMA_ARG_N_CPU_MOE": "4abc"}, True),
        ([], {"LLAMA_ARG_N_CPU_MOE": " 3"}, True),
        # Controls: zero, a negative (arg.cpp throws) and no digits place nothing.
        (["-ncmoe", "0"], {}, False),
        (["-ncmoe", "0abc"], {}, False),
        (["-ncmoe", "-2"], {}, False),
        (["-ncmoe", "abc"], {}, False),
        ([], {"LLAMA_ARG_N_CPU_MOE": "0"}, False),
        ([], {}, False),
    ],
)
def test_target_cpu_moe_counts_follow_upstream_parsing(extra_args, env, expected):
    assert (
        llama_cpp_module._args_place_tensors_on_cpu(extra_args)
        or llama_cpp_module._env_places_tensors_on_cpu(env)
    ) is expected


@pytest.mark.parametrize(
    "extra_args, env, expected",
    [
        (["-ngld", "2 layers"], {}, True),
        (["--spec-draft-ngl", "3x"], {}, True),
        ([], {"LLAMA_ARG_N_GPU_LAYERS_DRAFT": "1abc"}, True),
        # Controls: the whole drafter, the 0 offload, and the auto / all spellings.
        (["-ngld", "2"], {}, True),
        (["-ngld", "5 layers"], {}, False),
        (["-ngld", "0 layers"], {}, False),
        (["-ngld", "auto"], {}, False),
        (["-ngld", "all"], {}, False),
    ],
)
def test_a_draft_layer_split_follows_upstream_parsing(extra_args, env, expected):
    assert (
        llama_cpp_module._draft_is_split_across_host(extra_args, env, n_draft_layers = 4) is expected
    )


@pytest.mark.parametrize(
    "extra_args, env, expected",
    [
        (["-ngld", "0 layers"], {}, True),
        (["-ngld", " 0"], {}, True),
        ([], {"LLAMA_ARG_N_GPU_LAYERS_DRAFT": "0abc"}, True),
        # A GPU tensor override puts drafter weights on the card whatever -ngld says
        # (llama-model.cpp applies the buffer-type overrides per tensor).
        (["-ngld", "0", "-otd", ".*=CUDA0"], {}, False),
        (["-ngld", "0", "--override-tensor-draft=exps=CPU,attn=CUDA0"], {}, False),
        (["-ngld", "0", "-otd", "exps=CPU", "-otd", "attn=Vulkan0"], {}, False),
        # parse_tensor_buffer_overrides throws without "=": no placement to price as CPU.
        (["-ngld", "0", "-otd", "exps"], {}, False),
        # Controls: CPU-only overrides and the device pin keep the drafter off the card.
        (["-ngld", "0", "-otd", ".*=CPU"], {}, True),
        (["-ngld", "0", "-otd", "exps=CPU,attn=CPU", "-cmoed"], {}, True),
        (["-ngld", "0"], {}, True),
        (["--spec-draft-device", "none"], {}, True),
        (["-ngld", "1"], {}, False),
        (["-ngld", "auto"], {}, False),
    ],
)
def test_a_cpu_drafter_follows_upstream_parsing_and_tensor_overrides(extra_args, env, expected):
    assert llama_cpp_module._extra_args_draft_offloaded_to_cpu(extra_args, env) is expected


# --- a zero-offload launch keeps GPUs visible for every projector device name ------------


@pytest.mark.parametrize(
    "extra_args, env, expected",
    [
        # arg.cpp parses each projector device as it reads it, env first then argv, so a
        # name before a final none must still resolve: masking the GPUs fails it.
        (["--mmproj-device", "CUDA0", "--mmproj-device", "none"], {}, True),
        (["-mmdev=CUDA0", "-mmdev", "none"], {}, True),
        (["--mmproj-device", "none"], {"MTMD_BACKEND_DEVICE": "CUDA0"}, True),
        # Controls: only none ever named, or no projector device at all.
        (["--mmproj-device", "none"], {}, False),
        ([], {"MTMD_BACKEND_DEVICE": "none"}, False),
        (["--no-mmproj-offload"], {}, False),
    ],
)
def test_a_zero_offload_launch_keeps_gpus_for_an_earlier_projector_device(
    extra_args, env, expected
):
    cmd = ["/fake/llama-server", "-m", "m.gguf", "--mmproj", "p.gguf", "-ngl", "0", *extra_args]
    assert LlamaCppBackend._zero_offload_keeps_gpu_visible(cmd, env) is expected
