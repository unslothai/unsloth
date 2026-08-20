# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The vision-projector placement policy: GPU, CPU, or not loaded at all.

Model + speculative decoding + projector + 4096 context in VRAM if they fit;
otherwise pin the projector to the CPU; if that is still not enough, drop
speculative decoding as well; and load no projector at all, on either device,
when vision is switched off. Also covers the two fields the client reseeds the
switch from, which no single-side test exercises.
"""

from __future__ import annotations

import inspect
import os
import struct
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from core.inference.llama_cpp import (
    GgufLoadIntent,
    LlamaCppBackend,
    _resolved_mmproj_offload,
)
from models.inference import InferenceStatusResponse, LoadResponse
from routes.inference import (
    _estimate_gguf_required_gb,
    _guard_chat_load_against_training,
    _llama_runtime_fields,
    _LoadPlacement,
)

MIB = 1024 * 1024
GIB = 1024**3
_REAL_POPEN = subprocess.Popen

# Model + projector fit at 4096 but not at the native length: the placement loop
# shrinks the context long before it would spill a layer, so pricing residency at
# the native length is the bug this policy exists to avoid.
NATIVE_CTX = 262144
KV_PER_TOKEN = 64 * 1024  # 4096 ctx -> 256 MiB, NATIVE_CTX -> 16 GiB


def _write_gguf(path: Path) -> Path:
    def string(value: str) -> bytes:
        data = value.encode()
        return struct.pack("<Q", len(data)) + data

    metadata = string("general.architecture") + struct.pack("<I", 8) + string("llama")
    path.write_bytes(struct.pack("<IIQQ", 0x46554747, 3, 0, 1) + metadata)
    return path


def _backend(
    tmp_path: Path,
    *,
    memory,
    model_bytes: int = 6 * GIB,
    mmproj_bytes: int = 1 * GIB,
    drafter_bytes: int = 0,
    native_ctx: int = NATIVE_CTX,
):
    """A GGUF vision load with the fit inputs pinned to known numbers.

    ``native_ctx`` is lowered for the drafter tests: the drop probe prices the
    reserve at the context the target alone would reach, so a 262144 native
    length makes the draft KV, not the placement, decide every one of them.
    """
    backend = LlamaCppBackend()
    gguf = _write_gguf(tmp_path / "model.gguf")
    mmproj = _write_gguf(tmp_path / "mmproj-F16.gguf")
    drafter = _write_gguf(tmp_path / "mtp.gguf")

    def read_metadata(_path):
        backend._context_length = native_ctx
        backend._n_layers = 32
        backend._n_heads = 32
        backend._n_kv_heads = 8
        backend._embedding_length = 4096
        backend._vocab_size = 32000

    backend._read_gguf_metadata = read_metadata
    backend._get_gpu_memory = lambda _binary = None, **_kw: list(memory)
    backend._get_gpu_free_memory = lambda _binary = None, **_kw: [
        (index, free) for index, free, _total in memory
    ]
    backend._estimate_kv_cache_bytes = lambda ctx, *_a, **_kw: max(0, ctx) * KV_PER_TOKEN
    backend._compute_buffer_ctx_bytes = lambda *_a, **_kw: 0
    backend._estimate_compute_buffer_bytes = lambda **_kw: 256 * MIB
    backend._get_gguf_size_bytes = lambda path: (
        drafter_bytes if Path(path).name == "mtp.gguf" else model_bytes
    )
    backend._mmproj_vram_bytes = lambda _path: mmproj_bytes
    backend._resolve_launch_mmproj_path = lambda **_kw: str(mmproj)
    # Only the speculative-decoding test asks for a drafter; everywhere else the
    # resolution has to come back empty or MTP engages behind the scenes.
    backend._resolve_launch_mtp_path = lambda **_kw: str(drafter) if drafter_bytes else None
    backend._apu_ram_shortfall_message = lambda *_a, **_kw: None
    backend._amd_apu_wants_unified_memory = lambda *_a, **_kw: False
    backend._find_llama_server_binary = lambda include_denied = False: "/fake/llama-server"
    backend._is_vulkan_backend = lambda _binary = None: False
    backend._wait_for_health = lambda timeout: True
    backend._detect_audio_type_strict = lambda: None
    backend._apply_detected_audio = lambda _detected: True
    backend.probe_server_capabilities = lambda _binary = None: {
        "supports_no_mmproj_offload": True,
        "mtp_token": "draft-mtp",
        "spec_draft_n_max_flag": "--spec-draft-n-max",
    }
    return backend, gguf


def _launch(backend, gguf, **load_kwargs):
    captured = {}

    def fake_popen(cmd, **kwargs):
        if not cmd or str(cmd[0]) != "/fake/llama-server":
            return _REAL_POPEN(cmd, **kwargs)
        captured["cmd"] = list(cmd)
        captured["env"] = kwargs.get("env") or dict(os.environ)
        return type(
            "Process",
            (),
            {
                "pid": 123,
                "stdout": (),
                "poll": lambda self: None,
                "terminate": lambda self: None,
                "wait": lambda self, timeout = None: 0,
                "kill": lambda self: None,
            },
        )()

    intent_kwargs = {
        "is_vision": True,
        # Auto context, the mode the policy is about: 0 resolves to the model's
        # native length and lets the placement loop shrink it. A request pinned at
        # 4096 would hide the whole floor question.
        "n_ctx": 0,
        **load_kwargs,
    }
    with patch.object(subprocess, "Popen", side_effect = fake_popen):
        assert backend.load_model(
            GgufLoadIntent(
                gguf_path = str(gguf),
                model_identifier = "test",
                **intent_kwargs,
            )
        )
    return captured


def test_projector_stays_on_gpu_when_it_fits_at_the_floor(tmp_path):
    """Model + projector fit at 4096 but not at the native 262144 context.

    The placement loop shrinks the context, it does not spill layers, so both
    arms are fully GPU-resident and pinning would buy nothing while costing
    ~8.8x on every image encode.
    """
    backend, gguf = _backend(tmp_path, memory = [(0, 12_000, 24_000)])

    cmd = _launch(backend, gguf)["cmd"]

    assert "--mmproj" in cmd
    assert "--no-mmproj-offload" not in cmd
    # The premise the pin would have been traded against: every layer is already
    # resident, so what the native length cost was context, not residency.
    assert cmd[cmd.index("--fit") + 1] == "off"
    assert int(cmd[cmd.index("-c") + 1]) < NATIVE_CTX


def test_projector_pinned_to_cpu_when_it_does_not_fit(tmp_path):
    """The 6 GiB model fits at 4096 on this card, the projector on top does not.

    Vision is preserved rather than dropped: --mmproj still goes out, only the
    offload is disabled. Sized so the _MMPROJ_VRAM_SAFETY surcharge decides it:
    the budget is 8692 - 3% of 16384 = 8200 MiB, the footprint at 4096 is 6144
    model + 1024 projector + 256 compute + 320 CUDA context + 256 KV = 8000 MiB,
    and only the 409 MiB a projector costs beyond its file size puts it over.
    """
    backend, gguf = _backend(tmp_path, memory = [(0, 8_692, 16_384)])

    cmd = _launch(backend, gguf)["cmd"]

    assert "--mmproj" in cmd
    assert "--no-mmproj-offload" in cmd
    # And the trade was paid for: the model alone is fully resident, which is the
    # only thing the slower image encode is bought with.
    assert cmd[cmd.index("--fit") + 1] == "off"


def test_user_owns_the_placement_when_they_name_either_spelling(tmp_path):
    """llama.cpp is last-wins on the placement pair, so an explicit
    --mmproj-offload must not be raced by the automatic pin."""
    backend, gguf = _backend(tmp_path, memory = [(0, 8_692, 16_384)])

    cmd = _launch(backend, gguf, extra_args = ["--mmproj-offload"])["cmd"]

    assert cmd.count("--no-mmproj-offload") == 0


@pytest.mark.parametrize(
    "memory,label",
    [
        ([], "apple_unified"),
        ([(0, 7_600, 0)], "amd_apu_or_igpu"),
    ],
)
def test_shared_memory_pools_are_not_charged_as_discrete(tmp_path, memory, label):
    """Apple Silicon enumerates no GPU and an APU / iGPU reports free SYSTEM RAM
    as its free VRAM with a total of 0. Moving the encoder inside one pool frees
    nothing, so the same shortfall that pins a discrete card must not pin here."""
    backend, gguf = _backend(tmp_path, memory = memory)

    cmd = _launch(backend, gguf)["cmd"]

    assert "--no-mmproj-offload" not in cmd, label


# The drafter tests share one shape: a small native context so the drop probe
# prices the reserve near the floor, and a 2 GiB drafter. Only the budget moves.
DRAFTER_NATIVE_CTX = 8192


def _drafter_backend(tmp_path, memory):
    return _backend(
        tmp_path,
        memory = memory,
        drafter_bytes = 2 * GIB,
        native_ctx = DRAFTER_NATIVE_CTX,
    )


def _launch_with_drafter(backend, gguf, tmp_path):
    return _launch(
        backend,
        gguf,
        mtp_draft_path = str(tmp_path / "mtp.gguf"),
        speculative_type = "auto",
    )["cmd"]


def test_the_projector_is_pinned_before_the_drafter_is_dropped(tmp_path):
    """The projector is the first thing given up, not the last.

    A bounded per-image cost is cheaper to concede than speculative decoding's
    per-token speedup, so the order is pin the projector, then drop the drafter
    only if that was not enough. Here it is enough: an unsized draft KV costs
    five points of fraction, so the budget is 12470 - 8% of 24000 = 10550 MiB,
    model + drafter + projector needs about 10682, and the pin gives back the
    projector's 1434 so the drafter survives.

    Deliberately tight. Every term of the drafter's charge is decisive at this
    budget, including its 224 MiB decode graph, so dropping any one of them from
    the predicate shows up here rather than passing on slack.
    """
    backend, gguf = _drafter_backend(tmp_path, [(0, 12_470, 24_000)])

    cmd = _launch_with_drafter(backend, gguf, tmp_path)

    assert "--no-mmproj-offload" in cmd
    assert "--model-draft" in cmd


def test_both_are_given_up_when_pinning_alone_is_not_enough(tmp_path):
    """Step 3: the drafter goes too, but only after the projector has moved and
    the load still does not fit. Budget 8200 MiB against about 9250 for model +
    drafter with the projector already pinned."""
    backend, gguf = _drafter_backend(tmp_path, [(0, 8_692, 16_384)])

    cmd = _launch_with_drafter(backend, gguf, tmp_path)

    assert "--no-mmproj-offload" in cmd
    assert "--model-draft" not in cmd


def test_the_drafters_vram_is_part_of_the_pin_decision(tmp_path):
    """The pin runs with speculative decoding still live, so the drafter has to
    be charged or the predicate answers for a machine that does not exist.

    Differential, on one budget: the only thing that changes between the two
    loads is whether a drafter is present. Model + projector alone is about 8410
    MiB against a 10080 MiB budget and fits comfortably, so a predicate that
    leaves the drafter out cannot pin either load, and the asymmetry below is
    exactly the drafter's roughly 2272 MiB of weights and draft graph.
    """
    memory = [(0, 12_000, 24_000)]

    with_drafter, gguf = _drafter_backend(tmp_path, memory)
    pinned = _launch_with_drafter(with_drafter, gguf, tmp_path)

    without_drafter, gguf2 = _backend(tmp_path, memory = memory, native_ctx = DRAFTER_NATIVE_CTX)
    unpinned = _launch(without_drafter, gguf2)["cmd"]

    assert "--no-mmproj-offload" in pinned
    assert "--model-draft" in pinned
    # Same card, same model, same projector: only the drafter differs.
    assert "--no-mmproj-offload" not in unpinned


_TOPOLOGIES = [
    ([(0, 7_600, 8_192)], True, "linux_nvidia_discrete"),
    ([(0, 7_600, 8_192)], True, "windows_nvidia_discrete"),
    ([(0, 7_600, 8_192)], True, "wsl_nvidia_discrete"),
    ([(0, 7_600, 8_192)], True, "linux_amd_discrete_rocm"),
    ([(0, 7_600, 8_192)], True, "windows_amd_discrete_vulkan"),
    ([(0, 7_600, 0)], False, "linux_amd_apu"),
    ([(0, 7_600, 0)], False, "windows_amd_igpu"),
    ([(0, 7_600, 0)], False, "wsl_amd_igpu"),
    ([(0, 7_600, 0)], False, "linux_intel_igpu"),
    ([], False, "mac_apple_silicon_unified"),
    ([], False, "linux_cpu_only"),
    ([], False, "windows_cpu_only"),
    ([], False, "wsl_cpu_only"),
    ([], False, "mac_cpu_only"),
]


@pytest.mark.parametrize("memory,expect_pin,label", _TOPOLOGIES)
def test_the_platform_and_gpu_matrix_pins_only_where_memory_is_discrete(
    tmp_path, memory, expect_pin, label
):
    """Moving the encoder out of a shared pool frees nothing, so only a card with
    its own memory may be charged for the projector. An APU, an iGPU and a
    virtualised Metal device all report free SYSTEM RAM as free VRAM."""
    backend, gguf = _backend(tmp_path, memory = memory)

    cmd = _launch(backend, gguf)["cmd"]

    assert ("--no-mmproj-offload" in cmd) is expect_pin, label
    # Vision survives either way: the pin moves the projector, it never drops it.
    assert "--mmproj" in cmd, label


@pytest.mark.parametrize(
    "memory,label",
    [
        ([(0, 7_600, 8_192), (1, 20_000, 24_576)], "big_card_second"),
        ([(0, 20_000, 24_576), (1, 7_600, 8_192)], "big_card_first"),
    ],
)
def test_a_heterogeneous_pair_is_ranked_before_the_projector_is_charged(tmp_path, memory, label):
    """Enumeration order must not decide placement: the same two cards in either
    order have to reach the same answer, or the pin is reading the device list
    rather than the memory on it."""
    backend, gguf = _backend(tmp_path, memory = memory)

    cmd = _launch(backend, gguf)["cmd"]

    assert "--mmproj" in cmd, label
    assert ("--no-mmproj-offload" in cmd) is False, label


def test_a_user_pinned_projector_is_not_charged_against_vram(tmp_path):
    """--no-mmproj-offload puts the projector in host RAM, so its bytes are not on
    the card. Charging them anyway shrank the context and spilled layers to make
    room for VRAM nothing occupies, which is worse placement than Studio's own."""
    backend, gguf = _backend(tmp_path, memory = [(0, 7_600, 8_192)])

    cmd = _launch(backend, gguf, extra_args = ["--no-mmproj-offload"])["cmd"]

    assert "--mmproj" in cmd
    # Fully placed, exactly as an unpinned load on this card is.
    assert "--fit" in cmd and cmd[cmd.index("--fit") + 1] == "off"
    assert cmd[cmd.index("-c") + 1] == "9984"


def test_a_user_demanding_gpu_offload_still_pays_for_it(tmp_path):
    """The mirror: --mmproj-offload asks for the projector ON the card, so its bytes
    stay in the budget and the context shrinks to fit them. Resolving the value is
    what separates this from the case above; merely detecting ownership cannot."""
    backend, gguf = _backend(tmp_path, memory = [(0, 7_600, 8_192)])

    cmd = _launch(backend, gguf, extra_args = ["--mmproj-offload"])["cmd"]

    assert cmd[cmd.index("-c") + 1] == "4096"


def test_the_last_placement_spelling_is_what_gets_budgeted(tmp_path):
    """llama.cpp folds the pair into one option and takes the last occurrence, so a
    list ending in the disable form must budget as disabled however it starts."""
    backend, gguf = _backend(tmp_path, memory = [(0, 7_600, 8_192)])

    cmd = _launch(
        backend,
        gguf,
        extra_args = ["--mmproj-offload", "--no-mmproj-offload"],
    )["cmd"]

    assert cmd[cmd.index("-c") + 1] == "9984"


def test_the_projector_probe_agrees_with_the_layer_loop_it_gates(tmp_path):
    """The probe answers "is the projector resident", and the layer placement that
    follows must not then contradict it by spilling model layers. The context-compute
    buffer is replicated per device, so a probe pricing it once could call a
    multi-GPU split resident and hand the fit a load it cannot place. Whatever the
    probe decides, the launch has to be self-consistent: either the projector went
    to the CPU, or the model is fully placed."""
    backend, gguf = _backend(
        tmp_path,
        memory = [(0, 6_000, 8_192), (1, 6_000, 8_192)],
        model_bytes = 9 * GIB,
    )

    cmd = _launch(backend, gguf)["cmd"]

    pinned = "--no-mmproj-offload" in cmd
    fitted = "--fit" in cmd and cmd[cmd.index("--fit") + 1] == "off"
    assert pinned or fitted, (
        "the probe left the projector on the GPU and the fit then could not place "
        f"the model: {[c for c in cmd if 'fit' in str(c) or 'mmproj' in str(c)]}"
    )


def test_an_explicit_context_is_priced_at_the_length_it_asked_for(tmp_path):
    """The same card the auto test above leaves alone, with the context pinned.

    Auto and explicit sizing give way differently, so the probe cannot ask the same
    question of both. Auto shrinks the CONTEXT and never spills a layer, which is
    why it is asked at the 4096 floor. An explicit context is honored verbatim by
    every branch of the placement loop, so the only give left is ``--fit on``, which
    offloads MODEL LAYERS. Priced at the floor this load answers "the projector
    fits" and then pays for it in the one currency the policy refuses to spend.

    Budget 12000 free - 3% of 24000 = 11280 MiB. At the requested 65536: 6144 model
    + 4096 KV + 256 compute + 320 CUDA context = 10816 resident, and the 1433 MiB
    the projector really costs puts it at 12249, over. So the projector goes to host
    RAM and every layer stays on the card.
    """
    backend, gguf = _backend(tmp_path, memory = [(0, 12_000, 24_000)])

    cmd = _launch(backend, gguf, n_ctx = 65536)["cmd"]

    assert "--mmproj" in cmd
    assert "--no-mmproj-offload" in cmd
    # What the pin bought, and the whole reason it was worth making: the requested
    # context survives intact with the model fully resident.
    assert cmd[cmd.index("--fit") + 1] == "off"
    assert cmd[cmd.index("-c") + 1] == "65536"


def test_an_explicit_context_that_fits_with_the_projector_keeps_it_on_the_gpu(tmp_path):
    """The floor is not simply replaced by "always pin under an explicit context".

    Same card, a context small enough that model + projector + KV all fit. Nothing
    is bought by moving the encoder off the GPU here, so it stays and image encode
    keeps its ~8.8x.
    """
    backend, gguf = _backend(tmp_path, memory = [(0, 12_000, 24_000)])

    cmd = _launch(backend, gguf, n_ctx = 8192)["cmd"]

    assert "--mmproj" in cmd
    assert "--no-mmproj-offload" not in cmd
    assert cmd[cmd.index("--fit") + 1] == "off"
    assert cmd[cmd.index("-c") + 1] == "8192"


def test_an_environment_owned_placement_is_not_reversed_by_the_pin(tmp_path, monkeypatch):
    """common/arg.cpp applies every set_env option BEFORE argv, so an appended
    --no-mmproj-offload does not lose to LLAMA_ARG_MMPROJ_OFFLOAD=1, it overwrites
    it, and llama.cpp says so only in a stderr warning nobody reads. A user who set
    the variable globally owns the placement exactly as one who passed the flag
    does, which is already how the CPU-recovery gate reads it."""
    monkeypatch.setenv("LLAMA_ARG_MMPROJ_OFFLOAD", "1")
    # The card that pins when nobody has claimed the placement.
    backend, gguf = _backend(tmp_path, memory = [(0, 8_692, 16_384)])

    cmd = _launch(backend, gguf)["cmd"]

    assert "--mmproj" in cmd
    assert "--no-mmproj-offload" not in cmd


def test_an_environment_pinned_projector_is_not_charged_against_vram(tmp_path, monkeypatch):
    """The mirror of the flag case: LLAMA_ARG_MMPROJ_OFFLOAD=0 puts the projector in
    host RAM just as --no-mmproj-offload does, so budgeting its bytes shrinks the
    context for VRAM nothing occupies."""
    monkeypatch.setenv("LLAMA_ARG_MMPROJ_OFFLOAD", "0")
    backend, gguf = _backend(tmp_path, memory = [(0, 7_600, 8_192)])

    cmd = _launch(backend, gguf)["cmd"]

    assert "--mmproj" in cmd
    assert cmd[cmd.index("--fit") + 1] == "off"
    # The same context the flag spelling earns on this card.
    assert cmd[cmd.index("-c") + 1] == "9984"


def test_the_negative_environment_spelling_pins_on_presence_alone(tmp_path, monkeypatch):
    """get_value_from_env checks the LLAMA_ARG_NO_ compatibility spelling first and
    forces falsey on getenv returning non-null, so an empty value still pins. Read
    any other way this charges VRAM the child never allocates."""
    monkeypatch.setenv("LLAMA_ARG_NO_MMPROJ_OFFLOAD", "")
    backend, gguf = _backend(tmp_path, memory = [(0, 7_600, 8_192)])

    cmd = _launch(backend, gguf)["cmd"]

    assert cmd[cmd.index("-c") + 1] == "9984"


@pytest.mark.parametrize(
    ("extras", "env", "expected"),
    [
        # Silence on both sides is the only None: nobody has placed it.
        ([], {}, None),
        ([], {"LLAMA_ARG_MMPROJ_OFFLOAD": "1"}, True),
        ([], {"LLAMA_ARG_MMPROJ_OFFLOAD": "enabled"}, True),
        ([], {"LLAMA_ARG_MMPROJ_OFFLOAD": "off"}, False),
        # arg.cpp raises on a value that is neither, so there is no side to budget.
        ([], {"LLAMA_ARG_MMPROJ_OFFLOAD": "yes"}, None),
        # Presence, not value, and it wins over the positive spelling.
        ([], {"LLAMA_ARG_NO_MMPROJ_OFFLOAD": ""}, False),
        ([], {"LLAMA_ARG_MMPROJ_OFFLOAD": "1", "LLAMA_ARG_NO_MMPROJ_OFFLOAD": "0"}, False),
        # argv is parsed after the environment, so the flag wins either direction.
        (["--mmproj-offload"], {"LLAMA_ARG_MMPROJ_OFFLOAD": "0"}, True),
        (["--no-mmproj-offload"], {"LLAMA_ARG_MMPROJ_OFFLOAD": "1"}, False),
        (["--mmproj-offload"], {"LLAMA_ARG_NO_MMPROJ_OFFLOAD": "1"}, True),
    ],
)
def test_the_resolved_placement_follows_arg_cpps_own_precedence(extras, env, expected):
    """Environment first, argv on top, the negative spelling short-circuiting on
    presence. Anything else and Studio budgets for a placement the child does not
    run."""
    assert _resolved_mmproj_offload(extras, env) is expected


def test_an_unparseable_environment_value_is_still_the_callers_placement(tmp_path, monkeypatch):
    """No side to budget for, but the variable is set, so Studio must not append its
    own spelling on top: common_params_parse throws on the value and the load fails
    naming the caller's variable, not a Studio flag they never chose."""
    monkeypatch.setenv("LLAMA_ARG_MMPROJ_OFFLOAD", "yes")
    backend, gguf = _backend(tmp_path, memory = [(0, 8_692, 16_384)])

    cmd = _launch(backend, gguf)["cmd"]

    assert "--no-mmproj-offload" not in cmd


def test_an_explicit_context_too_large_for_either_still_gives_the_projector_up_first(tmp_path):
    """Same card, a context that does not fit even with the projector in host RAM.

    The fit has to spill layers whatever happens here, which is exactly why the
    projector should not be holding 1433 MiB of the card while it does: every byte
    it gives back is a byte of model that stays resident. The order the policy is
    built on does not change just because the pin alone was not enough, and this is
    the same shape as dropping the drafter after the pin.
    """
    backend, gguf = _backend(tmp_path, memory = [(0, 12_000, 24_000)])

    cmd = _launch(backend, gguf, n_ctx = 131072)["cmd"]

    assert "--mmproj" in cmd
    assert "--no-mmproj-offload" in cmd
    assert cmd[cmd.index("--fit") + 1] == "on"


# The context-compute buffer, at the shape the real one has: linear in context, and
# _CTX_COMPUTE_SPLIT_MULT times larger PER DEVICE once the model is layer-split. The
# shared _backend stubs it to a flat 0, which is fine for the single-GPU cases above
# and is exactly why none of them can see a split-rate error.
_CC_PER_TOKEN = 1536  # 6 MiB at 4096, the rate the bundled estimator produces


def _split_rate_backend(tmp_path, *, memory, **kwargs):
    backend, gguf = _backend(tmp_path, memory = memory, **kwargs)
    backend._compute_buffer_ctx_bytes = (
        lambda n_ctx, n_ubatch = None, cache_type_kv = None, *, layer_split = False: (
            n_ctx * _CC_PER_TOKEN * (LlamaCppBackend._CTX_COMPUTE_SPLIT_MULT if layer_split else 1)
        )
    )
    return backend, gguf


def test_the_probe_prices_an_explicit_context_the_way_the_split_placement_does(tmp_path):
    """Two cards, an explicit 65536, and a footprint that only a layer split can hold.

    The explicit branch selects through `_select_gpus_split_aware`, charging the
    context-compute buffer in the per-device overhead as well as the total and handing
    the single-to-split step to the retry. A layer split does not just replicate that
    buffer per card, it replicates a bigger one, so pricing it once understates the
    real cost several-fold: 96 MiB charged against 768 MiB spent at this context on
    two devices. That gap is wider than the projector surcharge being decided, so a
    plain probe answers "the projector fits" on exactly the loads whose placement then
    falls back to `--fit on` and spills model layers around the projector it kept.
    """
    backend, gguf = _split_rate_backend(tmp_path, memory = [(0, 7_200, 8_200), (1, 7_200, 8_200)])

    cmd = _launch(backend, gguf, n_ctx = 65536)["cmd"]

    assert "--mmproj" in cmd
    assert "--no-mmproj-offload" in cmd
    # The point of the pin: the requested context is placed on the two cards rather
    # than the model being offloaded around a resident projector.
    assert cmd[cmd.index("--fit") + 1] == "off"
    assert cmd[cmd.index("-c") + 1] == "65536"


def test_a_single_card_explicit_load_is_untouched_by_the_split_rate(tmp_path):
    """No split, no replication, so the split-aware selector must reach the same
    answer the plain one did. Guards the tightening from leaking onto one-GPU loads,
    where `_select_gpus_split_aware` returns before its retry."""
    backend, gguf = _split_rate_backend(tmp_path, memory = [(0, 12_000, 24_000)])

    cmd = _launch(backend, gguf, n_ctx = 8192)["cmd"]

    assert "--mmproj" in cmd
    assert "--no-mmproj-offload" not in cmd
    assert cmd[cmd.index("-c") + 1] == "8192"


def test_auto_is_priced_at_the_split_rate_too_but_still_at_the_floor(tmp_path):
    """The split rate and the floor are separate questions and only the floor is Auto's.

    Auto's own loop charges `_cc_bytes(ctx, n_gpus)` and then makes every card hold its
    copy, so it is already stricter than plain `_select_gpus`; pricing this probe plainly
    leaves it more optimistic than the loop it gates. What keeps the pin honest under
    Auto is the FLOOR, not loose accounting: a subset that cannot hold the projector at
    4096 is one Auto cannot rescue by shrinking further, so `--fit on` was coming either
    way and giving the projector up is the cheaper half of it.

    Two cards where plain accounting says the projector fits at 4096 and the split rate
    says it does not.
    """
    backend, gguf = _split_rate_backend(tmp_path, memory = [(0, 4_900, 5_900), (1, 4_900, 5_900)])

    cmd = _launch(backend, gguf)["cmd"]

    assert "--mmproj" in cmd
    assert "--no-mmproj-offload" in cmd


def test_auto_still_leaves_a_roomy_split_alone(tmp_path):
    """The floor is what stops the split rate from turning into a blanket pin: on cards
    with room at 4096 the projector stays on the GPU and Auto pays for the native
    context in context, exactly as it does on one card."""
    backend, gguf = _split_rate_backend(tmp_path, memory = [(0, 12_000, 16_000), (1, 12_000, 16_000)])

    cmd = _launch(backend, gguf)["cmd"]

    assert "--mmproj" in cmd
    assert "--no-mmproj-offload" not in cmd


def test_the_probe_reserves_the_compute_buffer_on_every_split_device(tmp_path):
    """The buffer belongs in the per-device overhead as well as the total.

    That is how the explicit branch charges it, and it is a separate term from the
    retry's single-to-split step: the step re-prices the FOOTPRINT once the count is
    known, while this is what makes each individual card carry its own copy. Drop it
    and a subset whose cards cannot each hold one is accepted; the placement then makes
    every device hold one anyway and falls back to `--fit on`, spilling model layers
    around the projector the probe just kept. Verified reachable by sweeping the launch
    path: without this term these numbers go from pinned with `--fit off` to unpinned
    with `--fit on`.
    """
    backend, gguf = _split_rate_backend(tmp_path, memory = [(0, 6_000, 7_000), (1, 6_000, 7_000)])

    cmd = _launch(backend, gguf, n_ctx = 32768)["cmd"]

    assert "--mmproj" in cmd
    assert "--no-mmproj-offload" in cmd
    assert cmd[cmd.index("--fit") + 1] == "off"
    assert cmd[cmd.index("-c") + 1] == "32768"


def test_the_predicted_pin_is_reported_through_both_responses(tmp_path):
    """A projector the fit moved to host RAM must say so, on the same channel the
    startup-recovery pin uses.

    Both routes end at the same placement and the same user-visible cost, so the
    notice cannot depend on which one got there. Without this the predicted pin is
    silent outside the server log: image encoding simply gets slower with nothing in
    the UI to explain it, which is the complaint the recovery path exists to answer.
    The card here is the one the pin fires on.
    """
    backend, gguf = _backend(tmp_path, memory = [(0, 8_692, 16_384)])
    cmd = _launch(backend, gguf)["cmd"]

    # Premise: this really is the predicted pin, not a post-crash recovery.
    assert "--no-mmproj-offload" in cmd
    assert backend.mmproj_fallback_reason == "cpu_offload"

    fields = _llama_runtime_fields(backend)
    load = LoadResponse(
        status = "loaded", model = "test", display_name = "test", inference = {}, **fields
    ).model_dump()
    status = InferenceStatusResponse(active_model = "test", **fields).model_dump()
    for payload, where in ((load, "load"), (status, "status")):
        assert payload["mmproj_fallback_reason"] == "cpu_offload", where


def test_a_load_that_keeps_the_projector_on_the_gpu_reports_nothing(tmp_path):
    """The control: the reason is a report of something that happened, so a load that
    never moved the projector must leave it None rather than always claiming CPU."""
    backend, gguf = _backend(tmp_path, memory = [(0, 12_000, 24_000)])
    cmd = _launch(backend, gguf)["cmd"]

    assert "--no-mmproj-offload" not in cmd
    assert backend.mmproj_fallback_reason is None


@pytest.mark.parametrize(
    ("env", "expected_retry"),
    [
        # get_value_from_env checks the LLAMA_ARG_NO_ spelling first and forces falsey
        # on presence alone, so the projector is already in host RAM either way.
        ({"LLAMA_ARG_NO_MMPROJ_OFFLOAD": "1"}, False),
        ({"LLAMA_ARG_NO_MMPROJ_OFFLOAD": ""}, False),
        # It wins over the positive spelling, exactly as arg.cpp orders them.
        ({"LLAMA_ARG_MMPROJ_OFFLOAD": "1", "LLAMA_ARG_NO_MMPROJ_OFFLOAD": "0"}, False),
        # is_falsey accepts `disabled`; the spelling list alone does not.
        ({"LLAMA_ARG_MMPROJ_OFFLOAD": "disabled"}, False),
        # Still a GPU projector, so the recovery retry is real work.
        ({"LLAMA_ARG_MMPROJ_OFFLOAD": "enabled"}, True),
        ({}, True),
    ],
)
def test_the_recovery_retry_sees_every_environment_pin(env, expected_retry):
    """A projector the environment already pinned to CPU cannot be rescued by pinning
    it again: the retry respawns a command identical in effect, cannot clear the
    allocation failure, and costs the caller the real error, because the branch that
    surfaces that OOM only runs when this returns None."""
    cmd = ["llama-server", "-m", "/cache/model.gguf", "--mmproj", "/cache/mmproj.gguf"]
    retry = LlamaCppBackend._with_mmproj_offload_disabled(cmd, env)

    assert (retry is not None) is expected_retry
    if expected_retry:
        assert retry[-1] == "--no-mmproj-offload"


def test_the_speculative_reserve_is_normalized_before_anything_prices_it(tmp_path):
    """Ordering invariant: `_mtp_bytes` reads `mtp_overhead_fn` at call time, so the
    CPU-drafter normalization has to run before the first thing that prices it.

    It used to run after the projector probe. That is the wrong order on its face --
    a drafter pinned to the CPU allocates no VRAM, so charging the probe its full
    GPU footprint could pin the projector and cost ~8.8x per image encode for memory
    nothing was holding.

    Asserted on the source rather than on a launch deliberately. I could not construct
    a configuration where the two orders actually decide differently: on every route I
    could build, a separate CPU-pinned drafter is already excluded from the budget
    upstream, so `mtp_overhead_fn` is None at the probe either way and the block is a
    no-op. Reaching it needs a target that keeps its own reserve while a sidecar
    displaces an embedded head. So this locks the ordering, which is what was fixed,
    and does not claim a behavioural difference I could not demonstrate.
    """
    source = Path(inspect.getsourcefile(LlamaCppBackend)).read_text()
    normalize_at = source.index("if _draft_cpu_no_embedded and mtp_overhead_fn is not None:")
    probe_at = source.index("_mm_mtp_on_gpu = _mtp_will_engage and not _draft_cpu_no_embedded")
    assert (
        normalize_at < probe_at
    ), "the CPU-drafter reserve must be normalized before the projector probe prices it"


def test_a_shared_device_beside_a_discrete_one_does_not_veto_the_pin(tmp_path):
    """A laptop pairing a discrete card with an APU enumerates both.

    Requiring EVERY enumerated device to have its own budget refused the pin outright
    on that machine, so a model that would have fitted the discrete card once the
    projector moved kept it resident instead and either pulled the shared device into
    the split or went to `--fit on`. The shared device is dropped from the question
    rather than answering no for the whole host: bytes handed back land on a card that
    has its own pool.

    The discrete card here is the one the single-GPU pin test uses, with a shared
    device (total 0) alongside it.
    """
    backend, gguf = _backend(tmp_path, memory = [(0, 8_692, 16_384), (1, 7_600, 0)])

    cmd = _launch(backend, gguf)["cmd"]

    assert "--mmproj" in cmd
    assert "--no-mmproj-offload" in cmd


def test_a_host_that_is_only_shared_memory_still_never_pins(tmp_path):
    """The rule that survives: with no budgeted device anywhere, moving the projector
    shuffles bytes inside one pool and frees nothing, so the 8.8x image-encode cost
    buys exactly nothing. Unchanged by the mixed-host relaxation above."""
    backend, gguf = _backend(tmp_path, memory = [(0, 7_600, 0), (1, 7_600, 0)])

    cmd = _launch(backend, gguf)["cmd"]

    assert "--mmproj" in cmd
    assert "--no-mmproj-offload" not in cmd


def test_a_user_pinned_projector_still_costs_a_shared_pool(tmp_path):
    """--no-mmproj-offload on an APU moves the encoder inside one pool.

    On a discrete card the flag really does take those bytes off the device, so
    dropping them from the fit is right. A shared pool has nowhere to move them
    to: the projector sits in the very memory the context is being fitted
    against, and spending it on KV as well over-commits. The reference load
    charges the same bytes as model weights with no projector at all, which is
    exactly what a CPU-resident projector costs here, so the two must fit to the
    same context.
    """
    memory = [(0, 9_000, 0)]

    backend, gguf = _backend(tmp_path, memory = memory)
    cmd = _launch(backend, gguf, extra_args = ["--no-mmproj-offload"])["cmd"]
    pinned_ctx = int(cmd[cmd.index("-c") + 1])

    reference, ref_gguf = _backend(tmp_path, memory = memory, model_bytes = 7 * GIB, mmproj_bytes = 0)
    ref_cmd = _launch(reference, ref_gguf)["cmd"]
    reference_ctx = int(ref_cmd[ref_cmd.index("-c") + 1])

    assert pinned_ctx == reference_ctx


def test_a_pinned_projector_costs_the_shared_pool_beside_a_discrete_card(tmp_path):
    """The mixed host, where asking "is any device budgeted?" gets it backwards.

    _discrete_vram drops shared devices to decide whether the pin frees anything
    ANYWHERE, and one discrete card is enough for that. The fit is a different
    question: the selection walks prefixes of the enumerated list, so a shared
    device ranked first is a candidate subset on its own, and a context fitted
    against that pool without the pinned projector over-commits it. The discrete
    card here is nearly full, so the shared pool is what the load runs on.
    """
    memory = [(0, 9_000, 0), (1, 100, 16_384)]

    backend, gguf = _backend(tmp_path, memory = memory)
    cmd = _launch(backend, gguf, extra_args = ["--no-mmproj-offload"])["cmd"]
    pinned_ctx = int(cmd[cmd.index("-c") + 1])

    # Same reference as the single-device case: a CPU-resident projector in a
    # shared pool costs exactly what the same bytes cost as model weights.
    reference, ref_gguf = _backend(
        tmp_path, memory = memory, model_bytes = 7 * GIB, mmproj_bytes = 0
    )
    ref_cmd = _launch(reference, ref_gguf)["cmd"]

    assert pinned_ctx == int(ref_cmd[ref_cmd.index("-c") + 1])
