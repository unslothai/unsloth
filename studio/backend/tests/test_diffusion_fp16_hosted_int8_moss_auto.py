# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""AUTO on a Turing card runs MOSS-Image from the hosted int8 ConvRot W8A8 file where the host RAM covers its measured
peak; smaller hosts keep the at-load per-row build, and no other card moves."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from core.inference import diffusion_fp16_hosted_int8 as FH  # noqa: E402
from core.inference import diffusion_small_host as SH  # noqa: E402
from core.inference import diffusion_transformer_quant as tq  # noqa: E402
from core.inference.diffusion_families import detect_family  # noqa: E402

MOSS = "OpenMOSS-Team/MOSS-Image-SFT"
KREA = "krea/Krea-2-Turbo"
QWEN21 = "Qwen/Qwen-Image-2.1"
T4_FP32 = types.SimpleNamespace(device = "cuda", dtype = torch.float32, ordinal = 0)
T4_FP16 = types.SimpleNamespace(device = "cuda", dtype = torch.float16, ordinal = 0)
STANDARD = (12976, 11427)
KAGGLE = (32102, 29800)
HIGH_RAM = (52223, 50118)
HUGE = (1 << 20, 1 << 20)
SMALL_HOSTS = (STANDARD, KAGGLE)
OTHER_CAPS = ((8, 0), (8, 6), (8, 9), (9, 0), (10, 0), (12, 0))


def _fam(base, name):
    fam = detect_family(base, override = name)
    assert fam is not None and fam.name == name
    return fam


@pytest.fixture
def host(monkeypatch):
    for env in (
        FH.FP16_CARD_HOSTED_INT8_ENV,
        SH.SMALL_HOST_ENV,
        SH.HOST_RAM_LIMIT_ENV,
        SH.HOST_RAM_CHECK_ENV,
        "UNSLOTH_NATIVE_INT8_ACT",
        "UNSLOTH_NATIVE_INT8_ROT",
        "UNSLOTH_DIFFUSION_INT8_CONVROT",
        "UNSLOTH_DIFFUSION_FP16_CARD_INT8",
    ):
        monkeypatch.delenv(env, raising = False)
    cap = {"now": (7, 5)}
    monkeypatch.setattr(tq, "torch_is_rocm", lambda: False)
    monkeypatch.setattr(tq, "is_stubbed", lambda name: False)
    monkeypatch.setattr(tq, "_capability", lambda ordinal = None: cap["now"])
    monkeypatch.setattr(SH, "resolve_snapshot_dir", lambda *a, **k: None)
    monkeypatch.setattr(SH, "host_commit_mib", lambda: None)
    monkeypatch.setattr(FH, "fp16_card_hosted_int8_cached", lambda *a, **k: "/hub/x.safetensors")
    ram = {"now": HIGH_RAM}
    monkeypatch.setattr(SH, "host_ram_mib", lambda: ram["now"])
    return types.SimpleNamespace(ram = ram, cap = cap)


@pytest.mark.parametrize("ram", [HIGH_RAM, HUGE])
@pytest.mark.parametrize("target", [T4_FP32, T4_FP16])
def test_auto_reads_the_hosted_convrot_build_for_moss_on_a_host_that_covers_its_peak(
    host, ram, target
):
    host.ram["now"] = ram
    fam = _fam(MOSS, "moss-image")
    assert FH.fp16_card_hosted_int8_reason(fam, tq.TQ_INT8, target) is not None
    why = FH.small_host_hosted_int8_reason(fam, target, base = MOSS, require_cached = True)
    assert why is not None and "ConvRot" in why
    planned = FH.fp16_card_hosted_int8_planned_scheme(fam, target, None, kind = "pipeline", base = MOSS)
    assert planned == tq.TQ_INT8


def test_explicit_int8_reads_the_hosted_build_for_moss(host):
    fam = _fam(MOSS, "moss-image")
    planned = FH.fp16_card_hosted_int8_planned_scheme(
        fam, T4_FP32, "int8", kind = "pipeline", base = MOSS
    )
    assert planned == tq.TQ_INT8
    assert FH.fp16_card_hosted_int8_reason(fam, planned, T4_FP32) is not None


@pytest.mark.parametrize("cap", OTHER_CAPS)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_every_other_card_is_unchanged(host, cap, dtype):
    host.cap["now"] = cap
    target = types.SimpleNamespace(device = "cuda", dtype = dtype, ordinal = 0)
    fam = _fam(MOSS, "moss-image")
    assert FH.small_host_hosted_int8_reason(fam, target, base = MOSS) is None
    assert (
        FH.fp16_card_hosted_int8_planned_scheme(fam, target, None, kind = "pipeline", base = MOSS)
        is None
    )


def test_kill_switches_lora_and_offload_keep_the_at_load_route(host, monkeypatch):
    fam = _fam(MOSS, "moss-image")
    assert FH.small_host_hosted_int8_reason(fam, T4_FP32, base = MOSS, has_lora = True) is None
    assert (
        FH.fp16_card_hosted_int8_planned_scheme(
            fam, T4_FP32, None, kind = "pipeline", base = MOSS, forces_offload = True
        )
        is None
    )
    assert (
        FH.fp16_card_hosted_int8_planned_scheme(
            fam, T4_FP32, None, kind = "pipeline", speed_mode = "off", base = MOSS
        )
        is None
    )
    monkeypatch.setenv(FH.FP16_CARD_HOSTED_INT8_ENV, "0")
    assert FH.small_host_hosted_int8_reason(fam, T4_FP32, base = MOSS) is None
    monkeypatch.delenv(FH.FP16_CARD_HOSTED_INT8_ENV)
    monkeypatch.setenv("UNSLOTH_DIFFUSION_FP16_CARD_INT8", "0")
    assert FH.small_host_hosted_int8_reason(fam, T4_FP32, base = MOSS) is None


@pytest.mark.parametrize(
    "base", ["/models/MOSS-Image-SFT", "someone/MOSS-Image-SFT-finetune", None]
)
def test_only_the_released_moss_repo(host, base):
    assert FH.small_host_hosted_int8_reason(_fam(MOSS, "moss-image"), T4_FP32, base = base) is None


def test_auto_needs_the_file_in_the_cache_at_load(host, monkeypatch):
    monkeypatch.setattr(FH, "fp16_card_hosted_int8_cached", lambda *a, **k: None)
    fam = _fam(MOSS, "moss-image")
    assert FH.small_host_hosted_int8_reason(fam, T4_FP32, base = MOSS, require_cached = True) is None
    assert FH.small_host_hosted_int8_reason(fam, T4_FP32, base = MOSS) is not None


def test_krea2_has_no_hosted_build_and_stays_on_the_at_load_route(host):
    host.ram["now"] = STANDARD
    fam = _fam(KREA, "krea-2")
    assert FH.fp16_card_hosted_int8_reason(fam, tq.TQ_INT8, T4_FP32) is None
    assert FH.small_host_hosted_int8_reason(fam, T4_FP32, base = KREA) is None


def test_other_hosted_families_still_need_a_host_the_dense_dit_misses(host):
    fam = _fam(QWEN21, "qwen-image-2.1")
    host.ram["now"] = HUGE
    assert FH.small_host_hosted_int8_reason(fam, T4_FP16, base = QWEN21) is None
    host.ram["now"] = STANDARD
    assert FH.small_host_hosted_int8_reason(fam, T4_FP16, base = QWEN21) is not None


@pytest.mark.parametrize("ram", SMALL_HOSTS + ((None, None),))
@pytest.mark.parametrize("target", [T4_FP32, T4_FP16])
def test_a_host_below_the_measured_peak_keeps_the_per_row_build(host, ram, target):
    host.ram["now"] = ram
    fam = _fam(MOSS, "moss-image")
    assert FH.small_host_hosted_int8_reason(fam, target, base = MOSS) is None
    assert (
        FH.fp16_card_hosted_int8_planned_scheme(fam, target, None, kind = "pipeline", base = MOSS)
        is None
    )
    note = FH.hosted_auto_host_ram_shortfall(fam)
    assert note is not None
    if ram[1] is not None:
        assert f"{ram[1]} MiB available" in note and "per-row" in note
    # an explicit int8 request still reads the hosted file on any host: only AUTO's pick is gated
    assert (
        FH.fp16_card_hosted_int8_planned_scheme(fam, target, "int8", kind = "pipeline", base = MOSS)
        == tq.TQ_INT8
    )


def test_the_gate_is_the_measured_peak_plus_the_small_host_reserve(host):
    fam = _fam(MOSS, "moss-image")
    peak = FH.FP16_CARD_HOSTED_INT8_AUTO_HOST_PEAK_MIB["moss-image"]
    total = 52223
    need = peak + SH.host_reserve_mib(total)
    host.ram["now"] = (total, need)
    assert FH.hosted_auto_host_ram_shortfall(fam) is None
    assert FH.small_host_hosted_int8_reason(fam, T4_FP32, base = MOSS) is not None
    host.ram["now"] = (total, need - 1)
    assert FH.hosted_auto_host_ram_shortfall(fam) is not None
    assert FH.small_host_hosted_int8_reason(fam, T4_FP32, base = MOSS) is None


def test_the_gate_reads_the_small_host_helper(host, monkeypatch):
    host.ram["now"] = (29000, 29000)
    assert FH.hosted_auto_host_ram_shortfall(_fam(MOSS, "moss-image")) is not None


def test_other_families_have_no_ram_gate(host):
    host.ram["now"] = STANDARD
    assert FH.hosted_auto_host_ram_shortfall(_fam(QWEN21, "qwen-image-2.1")) is None
    assert FH.hosted_auto_host_ram_shortfall(_fam(KREA, "krea-2")) is None


def test_forced_small_host_engages_where_the_dense_load_fits():
    comps = {
        "text_encoder": SH.StoredComponent("text_encoder", 15000, "bfloat16"),
        "vae": SH.StoredComponent("vae", 300, "bfloat16"),
    }
    kw = dict(device = "cuda", host_total_mib = HIGH_RAM[0], host_available_mib = HIGH_RAM[1])
    assert not SH.decide_small_host(comps, torch.float32, **kw).engaged
    forced = SH.decide_small_host(comps, torch.float32, force = True, **kw)
    assert forced.engaged and forced.storage_dtypes.get("text_encoder") == "bfloat16"
    assert "forced by the load" in forced.reason and SH.SMALL_HOST_ENV not in forced.reason


def test_auto_hosted_moss_load_forces_the_small_host_route_for_its_encoders():
    src = (BACKEND / "core" / "inference" / "diffusion.py").read_text(encoding = "utf-8")
    branch = src.index("elif fam.name == MOSS_IMAGE_FAMILY_NAME:")
    call = src.index("self._small_host_decision(", branch)
    end = src.index("pipe = load_moss_image_pipeline(", call)
    assert "force = small_host_hosted is not None" in src[call:end]
    assert "and hosted_native_scheme is not None" in src[call:end]


def test_status_records_the_pick_and_the_ram_decline():
    from core.inference import diffusion as d

    pipe = types.SimpleNamespace()
    SH.mark(pipe, {"reason": "test", "components": {"transformer": "int8 weights"}})
    d._note_small_host_hosted_int8(pipe, "too small", picked = False)
    info = SH.engaged_on(pipe)
    assert info["hosted_int8"] == {"picked": False, "reason": "too small"}
    assert info["components"] == {"transformer": "int8 weights"}
