# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Component sizes from safetensors headers (diffusion_component_sizes) against the hand tables they back up."""

from __future__ import annotations

import json
import os
import struct
from pathlib import Path

import pytest

from core.inference import diffusion_component_sizes as dcs
from core.inference import video_stream_residency as vsr
from core.inference.video_families import _FAMILIES as VIDEO_FAMILIES

_GB = 1000.0**3


def _real_hub_cache():
    """The host's HF hub cache, read at collection (before conftest points every cache root at an empty dir)."""
    try:
        from huggingface_hub import constants
        return os.environ.get("HF_HUB_CACHE") or constants.HF_HUB_CACHE
    except Exception:  # noqa: BLE001
        return None


_REAL_HUB_CACHE = _real_hub_cache()


def _write_header(path: Path, tensors: dict) -> None:
    """A header-only safetensors file: the size readers never touch the data section."""
    header: dict = {"__metadata__": {"format": "pt"}}
    offset = 0
    for name, (dtype, numel) in tensors.items():
        width = {"BF16": 2, "F16": 2, "F32": 4, "I64": 8, "F8_E4M3": 1, "F8_E5M2": 1}[dtype]
        header[name] = {
            "dtype": dtype,
            "shape": [numel],
            "data_offsets": [offset, offset + numel * width],
        }
        offset += numel * width
    raw = json.dumps(header).encode()
    path.parent.mkdir(parents = True, exist_ok = True)
    path.write_bytes(struct.pack("<Q", len(raw)) + raw)


def _component(
    root: Path,
    folder: str,
    gb: float,
    dtype: str = "BF16",
    *,
    stem: str = "diffusion_pytorch_model",
):
    width = 4 if dtype == "F32" else 2
    _write_header(
        root / folder / f"{stem}.safetensors", {"w": (dtype, int(round(gb * _GB / width)))}
    )


# ---- parser ----


def test_parser_rejects_implausible_and_non_object_headers(tmp_path):
    bad = tmp_path / "bad.safetensors"
    bad.write_bytes(struct.pack("<Q", 1 << 40) + b"{}")
    assert dcs.read_safetensors_header(bad) is None
    arr = tmp_path / "arr.safetensors"
    arr.write_bytes(struct.pack("<Q", 2) + b"[]")
    assert dcs.read_safetensors_header(arr) is None
    short = tmp_path / "short.safetensors"
    short.write_bytes(b"\x01\x02")
    assert dcs.read_safetensors_header(short) is None
    capped = tmp_path / "capped.safetensors"
    _write_header(capped, {"a": ("BF16", 4)})
    assert dcs.read_safetensors_header(capped) is not None
    assert dcs.read_safetensors_header(capped, max_bytes = 8) is None


def test_merged_callers_read_through_the_shared_parser(tmp_path):
    from core.inference.diffusion import DiffusionBackend
    from core.inference.diffusion_memory import safetensors_prefix_mib
    from core.inference.diffusion_small_host import _safetensors_bytes_by_dtype

    f = tmp_path / "m.safetensors"
    _write_header(f, {"blk.a": ("F32", 1 << 20), "blk.b": ("BF16", 1 << 20), "other": ("I64", 16)})
    assert safetensors_prefix_mib(f, "blk.") == 6
    assert _safetensors_bytes_by_dtype(f) == {"float32": 4 << 20, "bfloat16": 2 << 20, "i64": 128}
    assert DiffusionBackend._safetensors_param_count(f) == (2 << 20) + 16
    # mixed widths: the half load does not narrow (the fp32 tensors are pinned), sizes as stored
    assert DiffusionBackend._safetensors_cast_bytes(f, 2) == (4 << 20) + (2 << 20) + 128
    bad = tmp_path / "bad.safetensors"
    bad.write_bytes(b"\x00" * 4)
    assert safetensors_prefix_mib(bad, "blk.") is None
    assert _safetensors_bytes_by_dtype(bad) == {}
    assert DiffusionBackend._safetensors_param_count(bad) == 0
    assert DiffusionBackend._safetensors_cast_bytes(bad, 2) is None


# ---- file selection ----


def test_local_sizes_pick_default_variant_and_complete_index(tmp_path):
    root = tmp_path / "pipe"
    # sharded transformer + a stray fp16 variant the loader never reads
    for i in (1, 2):
        _write_header(
            root / "transformer" / f"diffusion_pytorch_model-0000{i}-of-00002.safetensors",
            {f"t{i}": ("F32", 1000)},
        )
    (root / "transformer" / "diffusion_pytorch_model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "t1": "diffusion_pytorch_model-00001-of-00002.safetensors",
                    "t2": "diffusion_pytorch_model-00002-of-00002.safetensors",
                }
            }
        )
    )
    _write_header(
        root / "transformer" / "diffusion_pytorch_model.fp16.safetensors", {"x": ("F16", 999_999)}
    )
    # text_encoder with a complete transformers index and a stale diffusers index naming absent shards (LTX-2 layout)
    _write_header(root / "text_encoder" / "model-00001-of-00001.safetensors", {"e": ("F32", 500)})
    (root / "text_encoder" / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"e": "model-00001-of-00001.safetensors"}})
    )
    (root / "text_encoder" / "diffusion_pytorch_model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"e": "diffusion_pytorch_model-00001-of-00009.safetensors"}})
    )
    # a partially downloaded component: its index lists a shard that is not there -> unknown, never smaller
    _write_header(
        root / "vae" / "diffusion_pytorch_model-00001-of-00002.safetensors", {"v": ("BF16", 10)}
    )
    (root / "vae" / "diffusion_pytorch_model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "v": "diffusion_pytorch_model-00001-of-00002.safetensors",
                    "w": "diffusion_pytorch_model-00002-of-00002.safetensors",
                }
            }
        )
    )
    sizes = dcs.local_component_sizes(root)
    assert sizes is not None and sizes.names() == ("text_encoder", "transformer")
    t = sizes.get("transformer")
    assert t.float_params == 2000 and t.stored_bytes == 8000 and t.resident_bytes(2) == 4000
    assert len(t.files) == 2 and all(".fp16." not in f for f in t.files)
    assert sizes.get("text_encoder").files == ("model-00001-of-00001.safetensors",)


def test_resident_cast_keeps_non_float_tensors(tmp_path):
    _write_header(tmp_path / "c" / "model.safetensors", {"f": ("F32", 100), "ids": ("I64", 10)})
    size = dcs.local_component_sizes(tmp_path).get("c")
    assert size.resident_bytes(2) == 200 + 80
    assert size.resident_bytes(4) == 400 + 80
    assert dict(size.bytes_by_dtype) == {"F32": 400}


# ---- roles and reconciliation ----


def _fam(name):
    return next(f for f in VIDEO_FAMILIES if f.name == name)


def test_video_roles(tmp_path):
    ltx25 = tmp_path / "ltx25"
    for folder, gb in (
        ("transformer", 37.98),
        ("text_encoder", 23.92),
        ("connectors", 6.34),
        ("vae", 1.45),
        ("audio_vae", 0.11),
        ("vocoder", 0.26),
        ("latent_upsampler", 1.0),
        ("prompt_enhancer", 10.2),  # not loaded by the pipeline: outside the table
    ):
        _component(
            ltx25,
            folder,
            gb,
            stem = "model"
            if folder in ("text_encoder", "prompt_enhancer")
            else "diffusion_pytorch_model",
        )
    triple = dcs.video_header_components_gb(_fam("ltx-2.5"), dcs.local_component_sizes(ltx25))
    assert triple == pytest.approx((37.98, 23.92, 6.34 + 1.45 + 0.11 + 0.26 + 1.0), abs = 1e-6)
    # an audio family without its audio VAE is incomplete
    (ltx25 / "audio_vae" / "diffusion_pytorch_model.safetensors").unlink()
    assert dcs.video_header_components_gb(_fam("ltx-2.5"), dcs.local_component_sizes(ltx25)) is None

    wan = tmp_path / "wan"
    _component(wan, "transformer", 5.68, "F32")
    _component(wan, "text_encoder", 22.72, "F32", stem = "model")
    _component(wan, "vae", 0.508, "F32")
    fam = _fam("wan2.1-t2v-1.3b")
    assert fam.vae_force_fp32
    # DiT and encoder cast to bf16, the fp32-pinned VAE stays at fp32
    assert dcs.video_header_components_gb(fam, dcs.local_component_sizes(wan)) == pytest.approx(
        (2.84, 11.36, 0.508), abs = 1e-6
    )

    moe = _fam("wan2.2-t2v-a14b")
    assert (
        dcs.video_header_components_gb(moe, dcs.local_component_sizes(wan)) is None
    )  # one expert only
    _component(wan, "transformer_2", 5.68, "F32")
    assert dcs.video_header_components_gb(moe, dcs.local_component_sizes(wan))[0] == pytest.approx(
        5.68
    )


def test_reconcile_rules(monkeypatch):
    monkeypatch.setattr(dcs, "_MISMATCHES", {})
    table = (10.0, 11.4, 2.8)
    assert dcs.reconcile_with_table(None, table, is_base = True) == table
    assert dcs.reconcile_with_table((9.9996, 11.3618, 2.8188), table, is_base = True) == table
    assert dcs.reconcile_with_table((9.9996, 11.3618, 2.8188), None, is_base = True) == (
        10.0,
        11.4,
        2.8,
    )
    # base repo: a real disagreement keeps the table and is recorded for review
    assert dcs.reconcile_with_table((12.0, 11.4, 2.8), table, is_base = True, key = ("f", "b")) == table
    assert dcs.table_mismatches()[("f", "b")] == ((12.0, 11.4, 2.8), table)
    # a variant the table does not describe: the larger of its headers and the table, per term
    assert dcs.reconcile_with_table((12.04, 11.4, 2.8), table, is_base = False) == (12.0, 11.4, 2.8)
    assert dcs.reconcile_with_table((8.0, 12.0, 1.0), table, is_base = False) == (10.0, 12.0, 2.8)
    assert dcs.reconcile_scalar(0.9957, 1.0, is_base = True) == 1.0
    assert dcs.reconcile_scalar(None, 1.0, is_base = True) == 1.0
    assert dcs.reconcile_scalar(1.5, 1.0, is_base = False) == 1.5
    assert dcs.reconcile_scalar(0.5, 1.0, is_base = False) == 1.0


def _stub_for_table(root: Path, fam) -> None:
    """A snapshot whose headers reproduce ``fam.bf16_components_gb`` exactly (stub for un-cached families)."""
    denoiser, text, vae = fam.bf16_components_gb
    experts = ("transformer", "transformer_2") if fam.is_moe else ("transformer",)
    for name in experts:
        _component(root, name, denoiser / len(experts))
    _component(root, "text_encoder", text, stem = "model")
    audio = 0.1 if fam.has_audio else 0.0
    if fam.has_audio:
        _component(root, "audio_vae", audio)
    _component(root, "vae", vae - audio, "F32" if fam.vae_force_fp32 else "BF16")


_TABLE_FAMILIES = [f for f in VIDEO_FAMILIES if f.bf16_components_gb and not f.modular_workflow]


@pytest.mark.parametrize("fam", _TABLE_FAMILIES, ids = lambda f: f.name)
def test_stub_headers_reproduce_every_video_table(tmp_path, monkeypatch, fam):
    monkeypatch.setattr(dcs, "_MISMATCHES", {})
    _stub_for_table(tmp_path, fam)
    sizes = dcs.local_component_sizes(tmp_path)
    assert dcs.video_header_components_gb(fam, sizes) == pytest.approx(
        fam.bf16_components_gb, abs = 1e-6
    )
    # the base repo resolving to these headers keeps the table exactly
    monkeypatch.setattr(dcs, "local_snapshot_dir", lambda repo, cache_dir = None: tmp_path)
    assert dcs.video_components_gb(fam) == fam.bf16_components_gb
    # a larger denoiser on the base repo is reported and the table still answers
    _component(tmp_path, "transformer", fam.bf16_components_gb[0] + 2.0)
    if fam.is_moe:
        (tmp_path / "transformer_2").rename(tmp_path / "unused")
        _component(tmp_path, "transformer_2", 0.0001)
    assert dcs.video_components_gb(fam) == fam.bf16_components_gb
    assert (fam.name, fam.base_repo) in dcs.table_mismatches()
    # the same headers on a variant repo price the variant
    variant = dcs.video_components_gb(fam, str(tmp_path))
    assert variant[0] == pytest.approx(
        round(fam.bf16_components_gb[0] + 2.0 + (0.0001 if fam.is_moe else 0), 1)
    )


def test_no_headers_falls_back_to_the_table(monkeypatch):
    monkeypatch.setattr(dcs, "local_snapshot_dir", lambda repo, cache_dir = None: None)
    monkeypatch.setattr(
        dcs, "hub_component_sizes", lambda *a, **k: pytest.fail("hub read for a tabled base")
    )
    for fam in VIDEO_FAMILIES:
        assert dcs.video_components_gb(fam, allow_hub = True) == fam.bf16_components_gb


def test_untabled_or_variant_repo_reads_the_hub_before_download(monkeypatch, tmp_path):
    fam = _fam("wan2.2-ti2v-5b")
    _stub_for_table(tmp_path, fam)
    stub = dcs.local_component_sizes(tmp_path)
    calls = []
    monkeypatch.setattr(dcs, "local_snapshot_dir", lambda repo, cache_dir = None: None)
    monkeypatch.setattr(dcs, "hub_component_sizes", lambda repo, **k: calls.append(repo) or stub)
    assert (
        dcs.video_components_gb(fam, "someone/finetune", allow_hub = True) == fam.bf16_components_gb
    )
    assert (
        dcs.video_components_gb(fam, "someone/finetune", allow_hub = False) == fam.bf16_components_gb
    )
    assert calls == ["someone/finetune"]


def test_hub_reads_are_disabled_offline(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    assert dcs.hub_component_sizes("Wan-AI/Wan2.2-TI2V-5B-Diffusers") is None
    monkeypatch.delenv("HF_HUB_OFFLINE")
    monkeypatch.setenv("UNSLOTH_DIFFUSION_HEADER_FETCH", "0")
    assert dcs.hub_component_sizes("Wan-AI/Wan2.2-TI2V-5B-Diffusers") is None


def test_ltx25_latent_upsampler_from_header(monkeypatch, tmp_path):
    from core.inference.video_ltx25 import LTX25_LATENT_UPSAMPLER_GB, ltx25_latent_upsampler_gb

    monkeypatch.setattr(dcs, "local_snapshot_dir", lambda repo, cache_dir = None: None)
    assert ltx25_latent_upsampler_gb() == LTX25_LATENT_UPSAMPLER_GB
    _component(tmp_path, "latent_upsampler", 0.995743320)  # the pack's 995,743,320 bytes
    assert ltx25_latent_upsampler_gb(str(tmp_path)) == LTX25_LATENT_UPSAMPLER_GB
    _component(tmp_path, "latent_upsampler", 1.6)
    assert ltx25_latent_upsampler_gb(str(tmp_path)) == pytest.approx(1.6)
    _component(tmp_path, "latent_upsampler", 0.4)
    assert ltx25_latent_upsampler_gb(str(tmp_path)) == LTX25_LATENT_UPSAMPLER_GB


# ---- the real local HF cache: headers vs tables ----

# Disagreements beyond rounding, measured from the Hub headers (2026-10-13). The table keeps answering for the base repo;
# each needs a human decision before a table changes.
KNOWN_TABLE_DELTAS = {
    # The 42.0 is the single-file DiT (diffusion_models/ltx-2.5-22b-*-transformer-bf16.safetensors, 42,018,190,584
    # bytes), which carries ~4.04 GB of connector weights. The Diffusers pack's transformer/ is 37.98 GB and its
    # connectors/ (6.34) are already in the companion term, so the table over-states the pack by ~4.0 GB.
    ("ltx-2.5", 0): 4.03,
    # Qwen2.5-VL 7B (14.14) + ByT5 (0.44 at bf16) = 14.58, table 14.8; VAE 2.52 at bf16 (5.04 stored fp32), table 2.4.
    ("hunyuanvideo-1.5", 1): -0.22,
    ("hunyuanvideo-1.5", 2): 0.12,
    ("hunyuanvideo-1.5-720p", 1): -0.22,
    ("hunyuanvideo-1.5-720p", 2): 0.12,
    # image: Qwen-Image-2.1's VAE is stored fp32 (1.35 GB) and loads at bf16 (0.68 GB); the table holds the stored size.
    ("qwen-image-2.1", 2): -0.72,
}


def _check_against_table(name: str, header, table) -> list:
    problems = []
    for i, (h, t) in enumerate(zip(header, table)):
        known = KNOWN_TABLE_DELTAS.get((name, i))
        if known is not None:
            if abs((t - h) - known) > 0.06 and abs((h - t) - known) > 0.06:
                problems.append((name, i, h, t, "known delta moved"))
        elif abs(h - t) > dcs.TABLE_ROUNDING_GB:
            problems.append((name, i, h, t))
    return problems


def test_cached_video_families_headers_match_tables():
    checked, problems = [], []
    for fam in _TABLE_FAMILIES:
        snap = dcs.local_snapshot_dir(fam.base_repo, _REAL_HUB_CACHE)
        header = (
            dcs.video_header_components_gb(fam, dcs.local_component_sizes(snap)) if snap else None
        )
        if header is None:
            continue
        checked.append(fam.name)
        problems += _check_against_table(fam.name, header, fam.bf16_components_gb)
    if not checked:
        pytest.skip("no video family is fully cached on this host")
    assert not problems, problems


def test_cached_image_families_headers_match_tables():
    from core.inference.diffusion_auto_policy import family_bf16_components_gb
    from core.inference.diffusion_families import _FAMILIES as IMAGE_FAMILIES

    checked, problems = [], []
    for fam in IMAGE_FAMILIES:
        table = family_bf16_components_gb(fam, fam.base_repo)
        snap = dcs.local_snapshot_dir(fam.base_repo, _REAL_HUB_CACHE)
        header = (
            dcs.image_header_components_gb(dcs.local_component_sizes(snap))
            if snap and table
            else None
        )
        if header is None:
            continue
        checked.append(fam.name)
        problems += _check_against_table(fam.name, header, table)
    if not checked:
        pytest.skip("no image family is fully cached on this host")
    assert not problems, problems


@pytest.mark.allow_network
@pytest.mark.skipif(
    os.environ.get("UNSLOTH_TEST_HUB_HEADERS") != "1",
    reason = "Hub ranged header reads (set UNSLOTH_TEST_HUB_HEADERS=1)",
)
@pytest.mark.parametrize("fam", _TABLE_FAMILIES, ids = lambda f: f.name)
def test_hub_headers_match_tables(fam):
    sizes = dcs.hub_component_sizes(fam.base_repo, timeout_seconds = 120)
    header = dcs.video_header_components_gb(fam, sizes)
    assert header is not None
    assert not _check_against_table(fam.name, header, fam.bf16_components_gb)


# ---- runtime activation calibration, any family ----


class _Module:
    pass


class _Pipe:
    def __init__(self):
        self.transformer = _Module()


def test_calibration_records_and_reuses_for_any_family(monkeypatch):
    import torch

    monkeypatch.setattr(vsr, "_PEAKS", {})
    allocated = {"now": 1000 << 20, "peak": 0}
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda dev = None: None)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda dev = None: allocated["now"])
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda dev = None: allocated["peak"])

    key = vsr.calibration_key("ltx-2.5", "Lightricks/LTX-2.5-Diffusers", "group", "cuda")
    pipe = _Pipe()
    assert vsr.arm_request(pipe, key = key, width = 64, height = 32, frames = 9, device = "cuda:0")
    allocated["peak"] = (1000 + 3000) << 20
    assert vsr.record_request_peak(pipe) == 3000
    work = 64 * 32 * 9
    assert vsr.measured_extra_mib(key, work) == 3000
    assert vsr.measured_extra_mib(key, work - 1) == 3000  # a smaller request is covered
    assert vsr.measured_extra_mib(key, work + 1) is None  # a larger one is not
    # a reload of the same model (fresh module) reuses the process-wide record
    reloaded = _Pipe()
    reloaded.transformer._unsloth_video_calibration_key = key
    assert vsr._measured_extra_mib(reloaded.transformer, work) == 3000
    # rearm=False (fit_for_request armed it) only tags the module
    other = _Pipe()
    assert vsr.arm_request(
        other, key = key, width = 1, height = 1, frames = 1, device = "cuda:0", rearm = False
    )
    assert getattr(other.transformer, "_unsloth_video_pending", None) is None
    assert not vsr.arm_request(_Pipe(), key = key, width = 1, height = 1, frames = 1, device = "cpu")
    monkeypatch.setenv(vsr.VIDEO_DIT_RESIDENT_MEASURED_ENV, "0")
    assert vsr.measured_extra_mib(key, work) is None


def test_shortfall_uses_a_measured_peak_only_when_smaller():
    from core.inference.video import video_offload_shortfall_message

    kwargs = dict(
        family = "ltx-2.5", floor_mib = 20_000, placement = "x", width = 1280, height = 704, frames = 121
    )
    base = video_offload_shortfall_message(available_mib = 1, **kwargs)
    needed = float(base.split("needs about ")[1].split(" GiB")[0])
    # unchanged without a measurement
    assert (
        video_offload_shortfall_message(available_mib = 1, measured_extra_mib = None, **kwargs) == base
    )
    # a large measured peak never raises the bar
    assert (
        video_offload_shortfall_message(available_mib = 1, measured_extra_mib = 10**6, **kwargs) == base
    )
    small = video_offload_shortfall_message(available_mib = 1, measured_extra_mib = 2000, **kwargs)
    assert float(small.split("needs about ")[1].split(" GiB")[0]) < needed
    fits = 20_000 + int(2000 * vsr.MEASURED_PEAK_MARGIN) + vsr.MEASURED_SLACK_MIB
    assert (
        video_offload_shortfall_message(available_mib = fits, measured_extra_mib = 2000, **kwargs)
        is None
    )


# ---- review follow-ups ----


def test_float8_tensors_are_floating_parameters(tmp_path):
    _write_header(
        tmp_path / "transformer" / "diffusion_pytorch_model.safetensors",
        {"a": ("F8_E4M3", 1000), "b": ("F8_E5M2", 500), "ids": ("I64", 2)},
    )
    size = dcs.local_component_sizes(tmp_path).get("transformer")
    assert size.float_params == 1500 and size.nonfloat_bytes == 16
    assert size.resident_bytes(2) == 1500 * 2 + 16  # a float8 store loaded at bf16 / fp16
    for dtype in ("F8_E4M3", "F8_E5M2", "F8_E4M3FNUZ", "F8_E5M2FNUZ", "F8_E8M0"):
        assert dcs.FLOAT_WIDTH[dtype] == 1


def test_optional_folders_do_not_send_a_cached_model_to_the_hub(tmp_path, monkeypatch):
    fam = _fam(
        "wan2.2-ti2v-5b"
    )  # non-MoE, no audio: no transformer_2 / audio_vae / connectors / vocoder
    _stub_for_table(tmp_path, fam)
    monkeypatch.setattr(dcs, "local_snapshot_dir", lambda repo, cache_dir = None: tmp_path)
    monkeypatch.setattr(
        dcs, "hub_component_sizes", lambda *a, **k: pytest.fail("hub read for a fully cached model")
    )
    assert (
        dcs.video_components_gb(fam, "someone/finetune", allow_hub = True) == fam.bf16_components_gb
    )
    assert dcs.video_required_roles(fam) == ("transformer", "text_encoder", "vae")
    assert dcs.video_required_roles(_fam("wan2.2-t2v-a14b")) == (
        "transformer",
        "transformer_2",
        "text_encoder",
        "vae",
    )
    assert dcs.video_required_roles(_fam("ltx-2.5"))[-1] == "audio_vae"
    # optional folders are still counted when present
    ltx = _fam("ltx-2.5")
    root = tmp_path / "ltx"
    _stub_for_table(root, ltx)
    _component(root, "connectors", 1.0)
    monkeypatch.setattr(dcs, "local_snapshot_dir", lambda repo, cache_dir = None: root)
    got = dcs.video_components_gb(ltx, "someone/ltx-finetune", allow_hub = True)
    assert got[2] == pytest.approx(round(ltx.bf16_components_gb[2] + 1.0, 1))
    # a missing required folder still asks the Hub (with the load's token)
    (root / "vae" / "diffusion_pytorch_model.safetensors").unlink()
    calls = []
    monkeypatch.setattr(
        dcs, "hub_component_sizes", lambda repo, **k: calls.append((repo, k.get("token")))
    )
    dcs.video_components_gb(ltx, "someone/ltx-finetune", allow_hub = True, token = "hf_secret")
    assert calls == [("someone/ltx-finetune", "hf_secret")]


class _Info:
    def __init__(self, sha):
        self.sha = sha
        self.siblings = ()


def _fake_hub(monkeypatch, shas, reads, tokens):
    import huggingface_hub

    class _Api:
        def __init__(self, token = None):
            tokens.append(token)

        def model_info(
            self,
            repo_id,
            revision = None,
            files_metadata = False,
            timeout = None,
        ):
            return _Info(shas[0])

    monkeypatch.setattr(huggingface_hub, "HfApi", _Api)
    monkeypatch.setattr(
        dcs,
        "_read_hub_sizes",
        lambda repo, info, commit, *a, **k: reads.append(commit)
        or dcs.ComponentSizes({}, "hub", commit),
    )


def test_hub_cache_revalidates_mutable_refs_by_commit(monkeypatch):
    monkeypatch.setattr(dcs, "_HUB_CACHE", {})
    monkeypatch.setattr(dcs, "_HUB_SHA_CACHE", {})
    monkeypatch.delenv("HF_HUB_OFFLINE", raising = False)
    monkeypatch.delenv("UNSLOTH_DIFFUSION_HEADER_FETCH", raising = False)
    shas, reads, tokens = ["a" * 40], [], []
    _fake_hub(monkeypatch, shas, reads, tokens)
    clock = [1000.0]
    import time as _time

    monkeypatch.setattr(_time, "monotonic", lambda: clock[0])
    assert dcs.hub_component_sizes("o/r", token = "t1").revision == "a" * 40
    assert dcs.hub_component_sizes("o/r", token = "t1").revision == "a" * 40  # fresh: no call
    assert reads == ["a" * 40] and tokens == ["t1"]
    clock[0] += dcs.HUB_REFRESH_SECONDS + 1
    # stale ref, same commit: one metadata call, no header re-read
    assert dcs.hub_component_sizes("o/r", token = "t1").revision == "a" * 40
    assert reads == ["a" * 40] and tokens == ["t1", "t1"]
    clock[0] += dcs.HUB_REFRESH_SECONDS + 1
    shas[0] = "b" * 40  # the branch moved
    assert dcs.hub_component_sizes("o/r", token = "t1").revision == "b" * 40
    assert reads == ["a" * 40, "b" * 40]
    # a pinned commit never revalidates
    pinned = "c" * 40
    shas[0] = pinned
    dcs.hub_component_sizes("o/r", revision = pinned)
    clock[0] += 10 * dcs.HUB_REFRESH_SECONDS
    n = len(tokens)
    dcs.hub_component_sizes("o/r", revision = pinned)
    assert len(tokens) == n
    # another token is a separate cache partition (and is the one sent)
    dcs.hub_component_sizes("o/r", token = "t2")
    assert tokens[-1] == "t2"


def test_calibration_key_separates_load_configurations():
    base = dict(
        transformer_quant = "int8",
        text_encoder_quant = None,
        dtype = "bfloat16",
        attention_backend = "sdpa",
    )
    k = vsr.calibration_key("ltx-2", "r", "group", "cuda:0", gguf_filename = "a-Q4_K_M.gguf", **base)
    assert k == vsr.calibration_key(
        "ltx-2", "r", "group", "cuda:0", gguf_filename = "a-Q4_K_M.gguf", **base
    )
    assert k != vsr.calibration_key(
        "ltx-2", "r", "group", "cuda:0", gguf_filename = "a-Q8_0.gguf", **base
    )
    assert k != vsr.calibration_key(
        "ltx-2",
        "r",
        "group",
        "cuda:0",
        gguf_filename = "a-Q4_K_M.gguf",
        **{**base, "dtype": "float32"},
    )
    assert k != vsr.calibration_key(
        "ltx-2",
        "r",
        "group",
        "cuda:0",
        gguf_filename = "a-Q4_K_M.gguf",
        **{**base, "attention_backend": "flash"},
    )
    assert k != vsr.calibration_key(
        "ltx-2", "r", "group", "cuda:1", gguf_filename = "a-Q4_K_M.gguf", **base
    )


def test_generate_keys_calibration_on_the_load_configuration():
    import inspect

    from core.inference import video

    src = inspect.getsource(video)
    call = src[src.index("video_stream_residency.calibration_key(") :]
    call = call[: call.index("\n                    )")]
    for field in (
        "gguf_filename",
        "dtype",
        "attention_backend",
        "speed_mode",
        "transformer_cache",
        "transformer_quant",
        "text_encoder_quant",
        "kind",
        "variant_id",
    ):
        assert f'"{field}"' in call, field
