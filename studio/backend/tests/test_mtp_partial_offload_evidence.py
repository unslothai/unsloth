# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""What counts as evidence that a placement is partial, and what does not.

Auto stands the embedded Hybrid Mamba MTP head down by emitting
``--spec-type none`` when the placement is partial, because the recurrent
rollback copies then cost more layers than the drafting wins back. Getting the
EVIDENCE test wrong is expensive in both directions: too strict and Unsloth is
back to the 3.11 token/s of the reported regression, too loose and it gives up a
real speedup on a card that had room for every layer.

``--fit on`` alone is not evidence. ``use_fit`` starts True at its declaration,
every placement-planner branch is gated on a non-empty ``gpus``, and the except
path restores True having priced nothing -- so an unfitted ``--fit on`` means
"nobody looked" at least as often as it means "it does not fit". Only a planner
run that completed over a real device list and still could not fit the model is
a verdict; a concrete ``--gpu-layers`` count is independent evidence and needs
no planner at all.

Platform is simulated by patching sys.platform (llama_cpp.py reads it at call
time) and the accelerator by stubbing the probe, which is what actually differs
between hosts: _get_gpu_memory returns nvidia-smi output on Linux/Windows/WSL
with NVIDIA, amd-smi output on ROCm, ggml Vulkan ordinals on a Vulkan build, and
[] on a Metal Mac and on any CPU-only box (llama_cpp.py:6598-6680).
"""

import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

import pytest  # noqa: E402

# Reuse the module's dependency stubs, fixtures and launch harness.
from test_llama_cpp_placement import _hybrid_mtp_backend, _launch  # noqa: E402

from core.inference.llama_cpp import LlamaCppBackend  # noqa: E402


def test_auto_keeps_mtp_when_the_gpu_selector_raises():
    """GPUs enumerated, then _select_gpus throws.

    The handler logs "GPU selection failed, using --fit on" and restores
    `gpu_indices, use_fit = None, True`. _detected_gpus is already populated, so
    the GPU-evidence guard passes and a fit-only test would read that fallback
    True as a partial verdict -- but no placement was ever computed.

    This route is why a `bool(_detected_gpus)` guard would not be enough: the
    verdict has to be recorded where the planner returns, not inferred later.
    """
    with tempfile.TemporaryDirectory() as td:
        tmp_path = Path(td)
        backend, gguf = _hybrid_mtp_backend(tmp_path, partial_offload = False)

        def _boom(*args, **kwargs):
            raise RuntimeError("probe wedged")

        backend._select_gpus = _boom
        backend._select_gpus_split_aware = _boom

        result = _launch(
            backend,
            gguf,
            speculative_type = "auto",
            n_ctx = 4096,
            n_parallel = 4,
        )

    cmd = result["cmd"]
    spec = cmd[cmd.index("--spec-type") + 1] if "--spec-type" in cmd else None
    assert spec != "none", (
        "Auto stood MTP down off the exception fallback's --fit on, "
        f"fallback reason {backend.spec_fallback_reason!r}"
    )


def test_auto_keeps_mtp_when_the_planner_proved_full_offload():
    """The planner returned a fully offloaded placement.

    _select_gpus gives ([0], False) -- every layer fits -- and the user appends a
    last-wins `--fit on` in the extras. fit_is_effectively_on then reads True and
    the arm calls the placement partial, discarding a verdict that positively
    proved the opposite.
    """
    with tempfile.TemporaryDirectory() as td:
        tmp_path = Path(td)
        backend, gguf = _hybrid_mtp_backend(tmp_path, partial_offload = False)

        result = _launch(
            backend,
            gguf,
            speculative_type = "auto",
            n_ctx = 4096,
            n_parallel = 4,
            extra_args = ["--fit", "on"],
        )

    cmd = result["cmd"]
    spec = cmd[cmd.index("--spec-type") + 1] if "--spec-type" in cmd else None
    assert spec != "none", (
        "Auto stood MTP down despite the planner proving full offload, "
        f"fallback reason {backend.spec_fallback_reason!r}"
    )


def test_a_concrete_partial_layer_count_still_stands_mtp_down():
    """The control: independent evidence must keep working.

    42 of 65 blocks is partial whatever the planner did, so tightening the fit arm
    must not touch this one. Passes on the PR head and must keep passing.
    """
    with tempfile.TemporaryDirectory() as td:
        tmp_path = Path(td)
        backend, gguf = _hybrid_mtp_backend(tmp_path, partial_offload = False)

        result = _launch(
            backend,
            gguf,
            speculative_type = "auto",
            n_ctx = 4096,
            n_parallel = 4,
            extra_args = ["--gpu-layers", "42"],
        )

    cmd = result["cmd"]
    assert cmd[cmd.index("--spec-type") + 1] == "none"
    assert backend.spec_fallback_reason == "mtp_partial_offload"


# ─────────────── the platform x accelerator product ───────────────

# (id, sys.platform value, extra marker so WSL is distinguishable from Linux)
PLATFORMS = [
    ("linux", "linux"),
    ("wsl", "linux"),
    ("windows", "win32"),
    ("mac", "darwin"),
]

# (id, probe result, vulkan build?, device pin the user might supply)
ACCELERATORS = [
    ("nvidia", [(0, 12 * 1024, 24 * 1024)], False, None),
    ("amd", [(0, 12 * 1024, 24 * 1024)], False, None),
    ("vulkan", [(0, 12 * 1024, 24 * 1024)], True, "Vulkan0"),
    ("cpu_only", [], False, None),
]


def _spec_of(cmd):
    return cmd[cmd.index("--spec-type") + 1] if "--spec-type" in cmd else None


def _mtp_is_engaged(cmd):
    return _spec_of(cmd) in ("draft-mtp", "mtp")


@pytest.mark.parametrize("plat_id,plat", PLATFORMS, ids = [p[0] for p in PLATFORMS])
@pytest.mark.parametrize(
    "acc_id,memory,vulkan,device",
    ACCELERATORS,
    ids = [a[0] for a in ACCELERATORS],
)
def test_partial_layer_count_stands_mtp_down_everywhere(
    tmp_path, plat_id, plat, acc_id, memory, vulkan, device
):
    """A concrete partial `--gpu-layers` is placement evidence on every host.

    42 of 65 blocks is partial whatever the planner, the probe or the OS did, so
    this cell must stand MTP down uniformly -- except CPU-only, where there is no
    GPU to partially offload TO and the CPU MTP policy still applies.
    """
    backend, gguf = _hybrid_mtp_backend(tmp_path, partial_offload = False, memory = memory)
    backend._is_vulkan_backend = lambda _binary = None: vulkan
    extra = ["--gpu-layers", "42"]
    if device:
        extra = ["--device", device, *extra]

    with patch.object(sys, "platform", plat):
        result = _launch(
            backend,
            gguf,
            speculative_type = "auto",
            n_ctx = 4096,
            n_parallel = 4,
            extra_args = extra,
        )

    cmd = result["cmd"]
    if acc_id == "cpu_only" and not device:
        # No GPU anywhere: keep the CPU MTP policy (llama_cpp.py:15530 guard).
        assert not _mtp_is_engaged(cmd) or _spec_of(cmd) != "none"
    else:
        assert _spec_of(cmd) == "none", f"{plat_id}/{acc_id} did not stand MTP down"
        assert backend.spec_fallback_reason == "mtp_partial_offload"


@pytest.mark.parametrize("plat_id,plat", PLATFORMS, ids = [p[0] for p in PLATFORMS])
@pytest.mark.parametrize(
    "acc_id,memory,vulkan,device",
    ACCELERATORS,
    ids = [a[0] for a in ACCELERATORS],
)
def test_full_offload_keeps_mtp_everywhere(tmp_path, plat_id, plat, acc_id, memory, vulkan, device):
    """The planner proved every layer fits: MTP is the whole point, keep it.

    This is the regression direction that matters most -- the PR must not cost
    MTP to the users it already works for.
    """
    backend, gguf = _hybrid_mtp_backend(tmp_path, partial_offload = False, memory = memory)
    backend._is_vulkan_backend = lambda _binary = None: vulkan
    extra = ["--device", device] if device else None

    with patch.object(sys, "platform", plat):
        result = _launch(
            backend,
            gguf,
            speculative_type = "auto",
            n_ctx = 4096,
            n_parallel = 4,
            extra_args = extra,
        )

    cmd = result["cmd"]
    assert _spec_of(cmd) != "none", (
        f"{plat_id}/{acc_id} stood MTP down on a proven FULL offload "
        f"(reason {backend.spec_fallback_reason!r})"
    )


@pytest.mark.parametrize("plat_id,plat", PLATFORMS, ids = [p[0] for p in PLATFORMS])
@pytest.mark.parametrize(
    "acc_id,memory,vulkan,device",
    ACCELERATORS,
    ids = [a[0] for a in ACCELERATORS],
)
def test_cpu_only_layer_count_keeps_mtp_everywhere(
    tmp_path, plat_id, plat, acc_id, memory, vulkan, device
):
    """`--gpu-layers 0` is CPU-only, not partial: the rollback copies cost no VRAM."""
    backend, gguf = _hybrid_mtp_backend(tmp_path, partial_offload = False, memory = memory)
    backend._is_vulkan_backend = lambda _binary = None: vulkan
    extra = ["--gpu-layers", "0"]
    if device:
        extra = ["--device", device, *extra]

    with patch.object(sys, "platform", plat):
        result = _launch(
            backend,
            gguf,
            speculative_type = "auto",
            n_ctx = 4096,
            n_parallel = 4,
            extra_args = extra,
        )

    assert _spec_of(result["cmd"]) != "none", f"{plat_id}/{acc_id} stood down on -ngl 0"


@pytest.mark.parametrize("plat_id,plat", PLATFORMS, ids = [p[0] for p in PLATFORMS])
@pytest.mark.parametrize(
    "acc_id,memory,vulkan,device",
    ACCELERATORS,
    ids = [a[0] for a in ACCELERATORS],
)
def test_over_full_layer_count_keeps_mtp_everywhere(
    tmp_path, plat_id, plat, acc_id, memory, vulkan, device
):
    """`--gpu-layers 999` is full offload plus the output layer, never partial."""
    backend, gguf = _hybrid_mtp_backend(tmp_path, partial_offload = False, memory = memory)
    backend._is_vulkan_backend = lambda _binary = None: vulkan
    extra = ["--gpu-layers", "999"]
    if device:
        extra = ["--device", device, *extra]

    with patch.object(sys, "platform", plat):
        result = _launch(
            backend,
            gguf,
            speculative_type = "auto",
            n_ctx = 4096,
            n_parallel = 4,
            extra_args = extra,
        )

    assert _spec_of(result["cmd"]) != "none", f"{plat_id}/{acc_id} stood down on -ngl 999"


@pytest.mark.parametrize("plat_id,plat", PLATFORMS, ids = [p[0] for p in PLATFORMS])
def test_explicit_mtp_survives_partial_offload_everywhere(tmp_path, plat_id, plat):
    """A user who picks MTP by hand overrides the policy on every platform."""
    backend, gguf = _hybrid_mtp_backend(tmp_path, partial_offload = True)

    with patch.object(sys, "platform", plat):
        result = _launch(
            backend,
            gguf,
            speculative_type = "mtp",
            n_ctx = 4096,
            n_parallel = 4,
            extra_args = ["--gpu-layers", "42"],
        )

    assert _spec_of(result["cmd"]) != "none", f"{plat_id} overrode an explicit MTP choice"


@pytest.mark.parametrize("plat_id,plat", PLATFORMS, ids = [p[0] for p in PLATFORMS])
def test_a_metal_mac_style_empty_probe_keeps_mtp(tmp_path, plat_id, plat):
    """No probe result and no concrete layer count is not evidence of anything.

    A Metal Mac reaches llama_cpp.py:6598 with no nvidia-smi, no amd-smi and no
    torch.cuda, so the probe is [] and `--fit on` is the untouched default from
    llama_cpp.py:14127 rather than a planner verdict. Same shape as a failed
    Vulkan probe on Linux or Windows.
    """
    backend, gguf = _hybrid_mtp_backend(tmp_path, partial_offload = True, memory = [])

    with patch.object(sys, "platform", plat):
        result = _launch(
            backend,
            gguf,
            speculative_type = "auto",
            n_ctx = 4096,
            n_parallel = 4,
        )

    assert (
        _spec_of(result["cmd"]) != "none"
    ), f"{plat_id} stood MTP down with no GPU evidence and no planner verdict"


@pytest.mark.parametrize("plat_id,plat", PLATFORMS, ids = [p[0] for p in PLATFORMS])
def test_an_empty_probe_with_a_hand_pinned_device_keeps_mtp(tmp_path, plat_id, plat):
    """The b126194 hole: a device pin proves a GPU EXISTS, not that fit is partial.

    On a Metal Mac (`--device Metal0`) or after a failed Vulkan probe
    (`--device Vulkan0`) the planner never ran -- every branch of it is gated on a
    non-empty `gpus` -- so `--fit on` is still the default. Standing MTP down here
    costs a real speedup on a card that may have room for every layer.
    """
    backend, gguf = _hybrid_mtp_backend(tmp_path, partial_offload = True, memory = [])
    device = "Metal0" if plat == "darwin" else "Vulkan0"

    with patch.object(sys, "platform", plat):
        result = _launch(
            backend,
            gguf,
            speculative_type = "auto",
            n_ctx = 4096,
            n_parallel = 4,
            extra_args = ["--device", device],
        )

    assert _spec_of(result["cmd"]) != "none", (
        f"{plat_id} stood MTP down off a device pin with no planner verdict "
        f"(reason {backend.spec_fallback_reason!r})"
    )


# ───────────────────── binary + GGUF vintage ─────────────────────


def test_a_build_without_mtp_reports_the_binary_not_the_placement(tmp_path):
    """An old llama.cpp with no MTP spelling must not claim a placement policy.

    Its `--spec-type` enum may not even carry "none", so the emit path below has
    to name binary_no_mtp and keep the update affordance.
    """
    backend, gguf = _hybrid_mtp_backend(tmp_path, partial_offload = True)
    backend.probe_server_capabilities = lambda _binary = None: {
        "supports_ngram_mod": False,
        "spec_draft_n_max_flag": None,
    }

    result = _launch(
        backend,
        gguf,
        speculative_type = "auto",
        n_ctx = 4096,
        n_parallel = 4,
        extra_args = ["--gpu-layers", "42"],
    )

    assert backend.spec_fallback_reason != "mtp_partial_offload"


def test_an_old_gguf_without_ssm_group_count_is_priced_as_before(tmp_path):
    """A GGUF predating the ssm.group_count key must not get a garbage estimate.

    _mamba_recurrent_state_bytes returns 0 when any dimension is missing, so the
    load degrades to the pre-PR number instead of a wrong one.
    """
    b = LlamaCppBackend()
    for k, v in {
        "_n_layers": 65,
        "_nextn_predict_layers": 1,
        "_n_kv_heads": 4,
        "_n_heads": 24,
        "_embedding_length": 5120,
        "_kv_key_length": 256,
        "_kv_value_length": 256,
        "_full_attention_interval": 4,
        "_ssm_inner_size": 6144,
        "_ssm_state_size": 128,
        "_ssm_group_count": None,
        "_ssm_conv_kernel": None,
    }.items():
        setattr(b, k, v)

    assert b._mamba_recurrent_state_bytes() == 0
    assert b._mamba_recurrent_state_bytes(n_parallel = 4, n_rs_seq = 2) == 0
    # And the KV estimate is still the plain attention number.
    assert b._estimate_kv_cache_bytes(4096, "f16") == 16 * 4096 * 4 * (256 + 256) * 2


def test_a_gguf_without_nextn_is_untouched_by_the_layer_change(tmp_path):
    """No embedded head: block_count - 0 == block_count, byte for byte."""
    b = LlamaCppBackend()
    for k, v in {
        "_n_layers": 28,
        "_n_kv_heads": 8,
        "_n_heads": 16,
        "_embedding_length": 1024,
        "_kv_key_length": 128,
        "_kv_value_length": 128,
    }.items():
        setattr(b, k, v)

    assert b._estimate_kv_cache_bytes(4096, "f16") == 28 * 4096 * 8 * (128 + 128) * 2


@pytest.mark.parametrize("n_parallel", [1, 2, 4, 8])
@pytest.mark.parametrize("n_rs_seq", [0, 1, 2, 3, 16])
def test_recurrent_state_scales_linearly_in_slots_and_depth(n_parallel, n_rs_seq):
    """llama-memory-recurrent.cpp:99 allocates n_seq_max * (1 + n_rs_seq) rows."""
    b = LlamaCppBackend()
    for k, v in {
        "_n_layers": 65,
        "_nextn_predict_layers": 1,
        "_full_attention_interval": 4,
        "_ssm_inner_size": 6144,
        "_ssm_state_size": 128,
        "_ssm_group_count": 16,
        "_ssm_conv_kernel": 4,
    }.items():
        setattr(b, k, v)

    base = b._mamba_recurrent_state_bytes(n_parallel = 1, n_rs_seq = 0)
    assert b._mamba_recurrent_state_bytes(n_parallel, n_rs_seq) == base * n_parallel * (
        1 + n_rs_seq
    )


@pytest.mark.parametrize(
    "field",
    [
        "_n_layers",
        "_ssm_inner_size",
        "_ssm_state_size",
        "_ssm_group_count",
        "_ssm_conv_kernel",
        "_full_attention_interval",
    ],
)
def test_any_missing_recurrent_dimension_fails_closed(field):
    """A partially-populated header must return 0, never a partial product."""
    b = LlamaCppBackend()
    for k, v in {
        "_n_layers": 65,
        "_nextn_predict_layers": 1,
        "_full_attention_interval": 4,
        "_ssm_inner_size": 6144,
        "_ssm_state_size": 128,
        "_ssm_group_count": 16,
        "_ssm_conv_kernel": 4,
    }.items():
        setattr(b, k, v)
    setattr(b, field, None)

    assert b._mamba_recurrent_state_bytes(n_parallel = 4, n_rs_seq = 2) == 0


def test_zero_full_attention_interval_does_not_divide_by_zero():
    b = LlamaCppBackend()
    for k, v in {
        "_n_layers": 65,
        "_nextn_predict_layers": 1,
        "_full_attention_interval": 0,
        "_ssm_inner_size": 6144,
        "_ssm_state_size": 128,
        "_ssm_group_count": 16,
        "_ssm_conv_kernel": 4,
    }.items():
        setattr(b, k, v)

    # fai == 0 means every layer is attention, so nothing is recurrent.
    assert b._mamba_recurrent_state_bytes(n_parallel = 4) == 0


def test_the_reported_regression_is_still_fixed(tmp_path):
    """The whole point of the PR, guarded against every fix above.

    Qwen3.8-27B UD-IQ2_M, about 12 GiB free, Auto, four slots: nvidia-smi answers,
    the planner runs over a real device list and cannot fit the model, so --fit on
    IS a verdict here and the stand-down must fire. If a tightening of the fit arm
    ever breaks this, Unsloth is back to 3.11 token/s.

    See https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/discussions/18.
    """
    backend, gguf = _hybrid_mtp_backend(tmp_path, partial_offload = True)

    result = _launch(
        backend,
        gguf,
        speculative_type = "auto",
        n_ctx = 4096,
        n_parallel = 4,
    )

    cmd = result["cmd"]
    assert cmd[cmd.index("--fit") + 1] == "on"
    assert _spec_of(cmd) == "none"
    assert "draft-mtp" not in cmd
    assert backend.spec_fallback_reason == "mtp_partial_offload"


def test_the_stand_down_survives_a_64k_context(tmp_path):
    """Same verdict at the other context the PR measured."""
    backend, gguf = _hybrid_mtp_backend(tmp_path, partial_offload = True)

    result = _launch(
        backend,
        gguf,
        speculative_type = "auto",
        n_ctx = 65536,
        n_parallel = 4,
    )

    assert _spec_of(result["cmd"]) == "none"
    assert backend.spec_fallback_reason == "mtp_partial_offload"


@pytest.mark.parametrize("n_parallel", [1, 2, 4, 8])
def test_the_verdict_does_not_depend_on_slot_count(tmp_path, n_parallel):
    """The rollback reserve is per-slot, but the policy is not."""
    backend, gguf = _hybrid_mtp_backend(tmp_path, partial_offload = True)

    result = _launch(
        backend,
        gguf,
        speculative_type = "auto",
        n_ctx = 4096,
        n_parallel = n_parallel,
    )

    assert _spec_of(result["cmd"]) == "none"


def test_nextn_larger_than_block_count_does_not_go_negative():
    b = LlamaCppBackend()
    for k, v in {
        "_n_layers": 4,
        "_nextn_predict_layers": 99,
        "_full_attention_interval": 4,
        "_ssm_inner_size": 6144,
        "_ssm_state_size": 128,
        "_ssm_group_count": 16,
        "_ssm_conv_kernel": 4,
    }.items():
        setattr(b, k, v)

    assert b._mamba_recurrent_state_bytes(n_parallel = 4) >= 0
    assert b._estimate_kv_cache_bytes(4096, "f16") >= 0
