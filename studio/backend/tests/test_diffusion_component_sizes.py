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
    # a first-expert-only checkpoint is a complete single denoiser (video._transformer_names)
    assert dcs.video_header_components_gb(moe, dcs.local_component_sizes(wan))[0] == pytest.approx(2.84)
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


def _stub_for_table(root: Path, fam) -> None:
    """A snapshot whose headers reproduce ``fam.bf16_components_gb`` exactly (stub for un-cached families)."""
    denoiser, text, vae = fam.bf16_components_gb
    experts = ("transformer", "transformer_2") if fam.is_moe else ("transformer",)
    for name in experts:
        _component(root, name, denoiser / len(experts))
    # every optional folder the family loads (connectors / vocoder / upsampler / a second text encoder) is present,
    # so the stub is a COMPLETE snapshot; their small sizes come out of the same table term
    extra_text = [n for n in dcs.video_loaded_roles(fam) if n.startswith("text_encoder_")]
    for n in extra_text:
        _component(root, n, 0.1, stem = "model")
    _component(root, "text_encoder", text - 0.1 * len(extra_text), stem = "model")
    companions = [n for n in dcs.video_loaded_roles(fam) if n not in ("vae", "text_encoder") and n not in extra_text and not n.startswith("transformer")]
    for n in companions:
        _component(root, n, 0.1)
    _component(root, "vae", vae - 0.1 * len(companions), "F32" if fam.vae_force_fp32 else "BF16")


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


# Header-vs-table disagreements (Hub, 2026-10-13); each needs a human decision before a table changes.
KNOWN_TABLE_DELTAS = {
    # 42.0 is the single-file DiT incl. ~4 GB connectors, already in the companion term: over-states the pack.
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
    # the ref still points at the cached snapshot, so only the folder rule is under test
    monkeypatch.setattr(dcs, "resolve_ref", lambda *a, **k: tmp_path.name)
    monkeypatch.setattr(
        dcs, "hub_component_sizes", lambda *a, **k: pytest.fail("hub read for a fully cached model")
    )
    assert (
        dcs.video_components_gb(fam, "someone/finetune", allow_hub = True) == fam.bf16_components_gb
    )
    assert dcs.video_required_roles(fam) == ("transformer", "text_encoder", "vae")
    # the second expert is loaded only when shipped (present or in the manifest), so never required
    assert dcs.video_required_roles(_fam("wan2.2-t2v-a14b")) == ("transformer", "text_encoder", "vae")
    assert dcs.video_required_roles(_fam("ltx-2.5"))[-1] == "audio_vae"
    # optional folders are still counted when present
    ltx = _fam("ltx-2.5")
    root = tmp_path / "ltx"
    _stub_for_table(root, ltx)
    _component(root, "connectors", 1.0)
    monkeypatch.setattr(dcs, "local_snapshot_dir", lambda repo, cache_dir = None: root)
    monkeypatch.setattr(dcs, "resolve_ref", lambda *a, **k: root.name)
    got = dcs.video_components_gb(ltx, "someone/ltx-finetune", allow_hub = True)
    assert got[2] == pytest.approx(round(ltx.bf16_components_gb[2] - 0.1 + 1.0, 1))  # stub connectors 0.1 -> 1.0
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
        "transformer_quant",
        "text_encoder_quant",
        "kind",
        "variant_id",
    ):
        assert f'"{field}"' in call, field
    # the cache mode is the one this request runs (see test_auto_cache_decision_is_shared_and_retags_the_calibration)
    assert "transformer_cache = request_cache_mode" in call




def _index(path: Path, shards: list) -> None:
    path.write_text(json.dumps({"weight_map": {f"t{i}": s for i, s in enumerate(shards)}}))


def test_incomplete_preferred_index_is_unknown_even_beside_a_complete_one(tmp_path):
    # diffusers folder: the preferred diffusion_pytorch_model index is incomplete, a stale model index is complete
    t = tmp_path / "transformer"
    _write_header(t / "model-00001-of-00001.safetensors", {"a": ("BF16", 10)})
    _index(t / "model.safetensors.index.json", ["model-00001-of-00001.safetensors"])
    _write_header(t / "diffusion_pytorch_model-00001-of-00002.safetensors", {"b": ("BF16", 10)})
    _index(
        t / "diffusion_pytorch_model.safetensors.index.json",
        ["diffusion_pytorch_model-00001-of-00002.safetensors", "diffusion_pytorch_model-00002-of-00002.safetensors"],
    )
    # transformers folder: the preferred model index is incomplete, a diffusers index beside it is complete
    e = tmp_path / "text_encoder"
    _write_header(e / "diffusion_pytorch_model-00001-of-00001.safetensors", {"c": ("BF16", 10)})
    _index(e / "diffusion_pytorch_model.safetensors.index.json", ["diffusion_pytorch_model-00001-of-00001.safetensors"])
    _write_header(e / "model-00001-of-00002.safetensors", {"d": ("BF16", 10)})
    _index(e / "model.safetensors.index.json", ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"])
    assert dcs.local_component_sizes(tmp_path) is None
    # the preferred stem's unindexed file wins over a lower-priority index
    v = tmp_path / "vae"
    _write_header(v / "diffusion_pytorch_model.safetensors", {"v": ("BF16", 7)})
    _write_header(v / "model-00001-of-00001.safetensors", {"w": ("BF16", 999)})
    _index(v / "model.safetensors.index.json", ["model-00001-of-00001.safetensors"])
    assert dcs.local_component_sizes(tmp_path).get("vae").files == ("diffusion_pytorch_model.safetensors",)


class _Sib:
    def __init__(self, name, size = 100):
        self.rfilename, self.size = name, size


def _hub_with(monkeypatch, siblings, header_for):
    import huggingface_hub

    class _Api:
        def __init__(self, token = None):
            pass

        def model_info(self, repo_id, revision = None, files_metadata = False, timeout = None):
            info = _Info("d" * 40)
            info.siblings = [_Sib(n) for n in siblings]
            return info

    monkeypatch.setattr(huggingface_hub, "HfApi", _Api)
    monkeypatch.setattr(dcs, "_hub_header", lambda repo, filename, *a, **k: header_for(filename))
    monkeypatch.setattr(dcs, "_HUB_CACHE", {})
    monkeypatch.setattr(dcs, "_HUB_SHA_CACHE", {})
    monkeypatch.delenv("HF_HUB_OFFLINE", raising = False)
    monkeypatch.delenv("UNSLOTH_DIFFUSION_HEADER_FETCH", raising = False)


def test_hub_result_with_a_failed_header_is_rejected(monkeypatch):
    files = [
        "transformer/diffusion_pytorch_model.safetensors",
        "text_encoder/model.safetensors",
        "vae/diffusion_pytorch_model.safetensors",
        "connectors/diffusion_pytorch_model.safetensors",
    ]
    ok = {"__metadata__": {}, "w": {"dtype": "BF16", "shape": [10], "data_offsets": [0, 20]}}
    _hub_with(monkeypatch, files, lambda f: ok)
    assert dcs.hub_component_sizes("o/r").names() == ("connectors", "text_encoder", "transformer", "vae")
    # an optional-but-loaded folder whose header does not read invalidates the whole answer (and is not cached)
    _hub_with(monkeypatch, files, lambda f: None if f.startswith("connectors/") else ok)
    assert dcs.hub_component_sizes("o/r") is None
    assert not dcs._HUB_SHA_CACHE
    # an index that cannot be read is unknown too, not a skipped folder
    _hub_with(monkeypatch, files[:3] + ["vocoder/diffusion_pytorch_model.safetensors.index.json"], lambda f: ok)
    monkeypatch.setattr(dcs, "_fetch_capped_text", lambda *a, **k: (_ for _ in ()).throw(OSError("refused")))
    assert dcs.hub_component_sizes("o/r") is None


class _Resp:
    def __init__(self, chunks, length = None):
        self.chunks, self.headers, self.pulled, self.closed = chunks, {}, 0, False
        if length is not None:
            self.headers["Content-Length"] = str(length)

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size = 1):
        for c in self.chunks:
            self.pulled += 1
            yield c

    def close(self):
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


class _Session:
    def __init__(self, resp):
        self.resp, self.kwargs = resp, None

    def get(self, url, **kwargs):
        self.kwargs = kwargs
        return self.resp


class _HttpxClient:
    """huggingface_hub 1.x: an httpx Client streams via ``Client.stream`` and has no ``stream=`` keyword."""

    def __init__(self, resp):
        self.resp, self.args = resp, None

    def stream(self, method, url, **kwargs):
        self.args = (method, kwargs)
        return self.resp

    def get(self, url, **kwargs):
        if "stream" in kwargs:
            raise TypeError("Client.get() got an unexpected keyword argument 'stream'")
        return self.resp


def test_index_fetch_works_on_the_httpx_client():
    resp = _Resp([b'{"a"', b": 1}"])
    resp.iter_bytes = resp.iter_content
    client = _HttpxClient(resp)
    assert dcs._fetch_capped_text(client, "u", {}, 1.0, float("inf")) == '{"a": 1}'
    assert client.args[0] == "GET" and client.args[1]["follow_redirects"] is True and resp.closed


def test_index_fetch_is_streamed_under_the_cap():
    good = _Session(_Resp([b'{"weight_map"', b": {}}"]))
    assert dcs._fetch_capped_text(good, "u", {}, 1.0, float("inf")) == '{"weight_map": {}}'
    assert good.kwargs["stream"] is True and good.resp.closed
    declared = _Session(_Resp([b"x"], length = 100))
    with pytest.raises(ValueError):
        dcs._fetch_capped_text(declared, "u", {}, 1.0, float("inf"), cap = 10)
    assert declared.resp.pulled == 0  # refused on Content-Length, nothing read
    lying = _Session(_Resp([b"x" * 6] * 1000))
    with pytest.raises(ValueError):
        dcs._fetch_capped_text(lying, "u", {}, 1.0, float("inf"), cap = 10)
    assert lying.resp.pulled == 2 and lying.resp.closed  # stopped as soon as the body passed the cap


def test_tagged_module_reads_only_its_own_mode(monkeypatch):
    monkeypatch.setattr(vsr, "_PEAKS", {})
    t2v = vsr.calibration_key("ltx-2", "r", "group", "cuda:0", fps = 24, conditioning = "t2v", cfg = True)
    i2v = vsr.calibration_key("ltx-2", "r", "group", "cuda:0", fps = 24, conditioning = "i2v", cfg = True)
    fps50 = vsr.calibration_key("ltx-2", "r", "group", "cuda:0", fps = 50, conditioning = "t2v", cfg = True)
    assert len({t2v, i2v, fps50}) == 3
    vsr._PEAKS[t2v] = {1000: 3000}
    module = _Module()
    module._unsloth_video_peaks = {1000: 9999}  # mode-blind own table: ignored once tagged
    module._unsloth_video_calibration_key = i2v
    assert vsr._measured_extra_mib(module, 1000) is None
    module._unsloth_video_calibration_key = t2v
    assert vsr._measured_extra_mib(module, 1000) == 3000
    # a fresh module after a reload, tagged before the fit, finds the process-wide peak
    reloaded = _Pipe()
    assert vsr.arm_request(reloaded, key = t2v, width = 1, height = 1, frames = 1, device = "cuda:0", rearm = False)
    assert vsr._measured_extra_mib(reloaded.transformer, 1000) == 3000


def test_generate_tags_before_the_fit_and_keys_the_request_mode():
    import inspect

    from core.inference import video

    src = inspect.getsource(video)
    key_at = src.index("video_stream_residency.calibration_key(")
    tag_at = src.index("rearm = False", key_at)
    fit_at = src.index("video_stream_residency.fit_for_request(", key_at)
    assert key_at < tag_at < fit_at
    call = src[key_at : src.index("\n                    )", key_at)]
    for field in ("fps", "conditioning", "first_frame", "last_frame", "references", "cfg"):
        assert f"{field} = " in call, field




def test_local_files_only_scope_disables_hub_reads(monkeypatch):
    from core.inference.diffusion_prequant import _LOCAL_FILES_ONLY

    monkeypatch.delenv("HF_HUB_OFFLINE", raising = False)
    monkeypatch.delenv("UNSLOTH_DIFFUSION_HEADER_FETCH", raising = False)
    assert not dcs.hub_disabled()
    token = _LOCAL_FILES_ONLY.set(True)
    try:
        assert dcs.hub_disabled()
        assert dcs.hub_component_sizes("o/r") is None
    finally:
        _LOCAL_FILES_ONLY.reset(token)


def test_load_pipeline_passes_local_only_to_the_sizing_path():
    import inspect

    from core.inference import video

    src = inspect.getsource(video.VideoBackend.load_pipeline)
    call = src[src.index("components = video_components_gb(") :]
    call = call[: call.index("\n        )")]
    assert "allow_hub = not local_files_only" in call


def test_snapshot_lookup_uses_the_active_cache_root(monkeypatch, tmp_path):
    from huggingface_hub import constants

    from utils import hf_cache_settings

    custom = tmp_path / "custom_hub"
    snap = custom / "models--o--r" / "snapshots" / ("e" * 40)
    _component(snap, "transformer", 0.001)
    (snap / "model_index.json").write_text("{}")
    (custom / "models--o--r" / "refs").mkdir(parents = True)
    (custom / "models--o--r" / "refs" / "main").write_text("e" * 40)
    assert constants.HF_HUB_CACHE != str(custom)  # huggingface_hub's own default does not see it
    monkeypatch.setattr(hf_cache_settings, "active_hf_hub_cache", lambda: str(custom))
    assert dcs.local_snapshot_dir("o/r") == snap
    assert dcs.component_sizes("o/r").names() == ("transformer",)


def test_shards_without_an_index_are_unknown(tmp_path):
    t = tmp_path / "transformer"
    _write_header(t / "diffusion_pytorch_model-00001-of-00002.safetensors", {"a": ("BF16", 10)})
    _write_header(t / "diffusion_pytorch_model-00002-of-00002.safetensors", {"b": ("BF16", 10)})
    assert dcs.local_component_sizes(tmp_path) is None
    assert dcs.select_weight_files(["model-00001-of-00001.safetensors"], folder = "text_encoder") == []
    # the unsharded file alone is a complete answer, a stray shard beside it is ignored
    _write_header(t / "diffusion_pytorch_model.safetensors", {"c": ("BF16", 30)})
    assert dcs.local_component_sizes(tmp_path).get("transformer").files == ("diffusion_pytorch_model.safetensors",)


def test_mixed_format_variant_keeps_its_known_larger_terms(tmp_path, monkeypatch):
    monkeypatch.setattr(dcs, "_MISMATCHES", {})
    fam = _fam("wan2.2-ti2v-5b")
    table = fam.bf16_components_gb
    _component(tmp_path, "transformer", table[0] + 3.0)
    (tmp_path / "text_encoder").mkdir()
    (tmp_path / "text_encoder" / "pytorch_model.bin").write_bytes(b"\0" * 16)  # header-less: unknown term
    _component(tmp_path, "vae", 0.1, "F32")
    terms = dcs.video_header_terms(fam, dcs.local_component_sizes(tmp_path))
    assert terms[1] is None and terms[0] == pytest.approx(table[0] + 3.0)
    got = dcs.video_components_gb(fam, str(tmp_path))
    # known larger denoiser wins, unknown text encoder and smaller VAE take the table
    assert got == (round(table[0] + 3.0, 1), table[1], table[2])
    # the base repo keeps the table on every term
    monkeypatch.setattr(dcs, "local_snapshot_dir", lambda repo, cache_dir = None: tmp_path)
    assert dcs.video_components_gb(fam) == table
    # without a table an unknown term means no answer
    assert dcs.reconcile_with_table((1.0, None, 1.0), None, is_base = False) is None


def test_transformer_2_counts_only_for_moe(tmp_path):
    _component(tmp_path, "transformer", 2.0)
    _component(tmp_path, "transformer_2", 2.0)
    _component(tmp_path, "text_encoder", 1.0, stem = "model")
    _component(tmp_path, "vae", 0.5)
    sizes = dcs.local_component_sizes(tmp_path)
    assert dcs.video_header_terms(_fam("wan2.2-ti2v-5b"), sizes)[0] == pytest.approx(2.0)
    assert dcs.video_header_terms(_fam("wan2.2-t2v-a14b"), sizes)[0] == pytest.approx(4.0)


# ---- review round 4 ----


def test_pytorch_model_stem_and_its_index_are_recognised(tmp_path):
    t = tmp_path / "transformer"
    for i in (1, 2):
        _write_header(t / f"pytorch_model-0000{i}-of-00002.safetensors", {f"p{i}": ("BF16", 100)})
    _index(
        t / "pytorch_model.safetensors.index.json",
        ["pytorch_model-00001-of-00002.safetensors", "pytorch_model-00002-of-00002.safetensors"],
    )
    size = dcs.local_component_sizes(tmp_path).get("transformer")
    assert size.float_params == 200 and len(size.files) == 2
    # the stems match the loader's default weight names
    from core.inference.diffusion import _DEFAULT_PIPELINE_WEIGHT_INDEX_RE, _DEFAULT_PIPELINE_WEIGHT_RE

    for name in ("pytorch_model-00001-of-00002.safetensors", "pytorch_model.safetensors"):
        assert dcs._DEFAULT_WEIGHT_RE.match(name) and _DEFAULT_PIPELINE_WEIGHT_RE.match(name)
    assert dcs._INDEX_RE.match("pytorch_model.safetensors.index.json")
    assert _DEFAULT_PIPELINE_WEIGHT_INDEX_RE.match("pytorch_model.safetensors.index.json")
    # an incomplete pytorch_model index is unknown, like the others
    (t / "pytorch_model-00002-of-00002.safetensors").unlink()
    assert dcs.local_component_sizes(tmp_path) is None


def test_complete_local_snapshot_is_revalidated_online(tmp_path, monkeypatch):
    fam = _fam("wan2.2-ti2v-5b")
    snap = tmp_path / ("a" * 40)
    _stub_for_table(snap, fam)
    _component(snap, "transformer", fam.bf16_components_gb[0] + 5.0)  # the cached (old) commit is larger
    monkeypatch.setattr(dcs, "local_snapshot_dir", lambda repo, cache_dir = None: snap)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising = False)
    monkeypatch.delenv("UNSLOTH_DIFFUSION_HEADER_FETCH", raising = False)
    fresh_root = tmp_path / "fresh"
    _stub_for_table(fresh_root, fam)
    fresh = dcs.local_component_sizes(fresh_root)
    hub_calls, ref_calls = [], []
    monkeypatch.setattr(dcs, "hub_component_sizes", lambda repo, **k: hub_calls.append(repo) or fresh)
    head = ["a" * 40]
    monkeypatch.setattr(dcs, "resolve_ref", lambda repo, revision = None, **k: ref_calls.append(repo) or head[0])
    expected_old = round(fam.bf16_components_gb[0] + 5.0, 1)
    # the ref still points at the cached snapshot: local answer, no header reads
    assert dcs.video_components_gb(fam, "o/variant", allow_hub = True)[0] == expected_old
    assert hub_calls == [] and ref_calls == ["o/variant"]
    # the ref moved: hub sizing
    head[0] = "b" * 40
    assert dcs.video_components_gb(fam, "o/variant", allow_hub = True) == fam.bf16_components_gb
    assert hub_calls == ["o/variant"]
    # offline / local-only / allow_hub=False stay local without resolving
    ref_calls.clear()
    assert dcs.video_components_gb(fam, "o/variant", allow_hub = False)[0] == expected_old
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    assert dcs.video_components_gb(fam, "o/variant", allow_hub = True)[0] == expected_old
    assert ref_calls == []


def test_resolve_ref_is_ttl_cached(monkeypatch):
    import huggingface_hub

    monkeypatch.setattr(dcs, "_REF_CACHE", {})
    monkeypatch.delenv("HF_HUB_OFFLINE", raising = False)
    monkeypatch.delenv("UNSLOTH_DIFFUSION_HEADER_FETCH", raising = False)
    calls, sha = [], ["c" * 40]

    class _Api:
        def __init__(self, token = None):
            pass

        def model_info(self, repo_id, revision = None, timeout = None, files_metadata = False):
            calls.append(files_metadata)
            return _Info(sha[0])

    monkeypatch.setattr(huggingface_hub, "HfApi", _Api)
    clock = [0.0]
    import time as _time

    monkeypatch.setattr(_time, "monotonic", lambda: clock[0])
    assert dcs.resolve_ref("o/r") == "c" * 40
    assert dcs.resolve_ref("o/r") == "c" * 40
    assert calls == [False]  # one cheap call (no files metadata) inside the TTL
    clock[0] += dcs.HUB_REFRESH_SECONDS + 1
    sha[0] = "d" * 40
    assert dcs.resolve_ref("o/r") == "d" * 40 and len(calls) == 2
    assert dcs.resolve_ref("o/r", "e" * 40) == "e" * 40 and len(calls) == 2  # pinned: no call


def test_companion_folders_follow_the_family_or_manifest(tmp_path):
    wan = _fam("wan2.2-ti2v-5b")  # no audio, not LTX
    _stub_for_table(tmp_path, wan)
    base = dcs.video_header_terms(wan, dcs.local_component_sizes(tmp_path))[2]
    for folder in ("connectors", "vocoder", "audio_vae"):
        _component(tmp_path, folder, 1.0)
    # leftover companion folders are not counted for a family that does not load them
    assert dcs.video_header_terms(wan, dcs.local_component_sizes(tmp_path))[2] == pytest.approx(base)
    # ... unless model_index.json names them
    (tmp_path / "model_index.json").write_text(
        json.dumps({"_class_name": "X", "vae": ["diffusers", "A"], "vocoder": ["transformers", "B"], "connectors": [None, None]})
    )
    sizes = dcs.local_component_sizes(tmp_path)
    assert sizes.manifest == frozenset({"vae", "vocoder"})
    assert dcs.video_header_terms(wan, sizes)[2] == pytest.approx(base + 1.0)
    # the LTX family loads connectors, an audio family audio_vae and vocoder
    ltx = _fam("ltx-2")
    ltx_root = tmp_path / "ltx"
    _stub_for_table(ltx_root, ltx)
    before = dcs.video_header_terms(ltx, dcs.local_component_sizes(ltx_root))[2]
    _component(ltx_root, "connectors", 1.0)
    _component(ltx_root, "vocoder", 0.5)
    # the stub already holds 0.1 GB of each; they grew to 1.0 and 0.5
    assert dcs.video_header_terms(ltx, dcs.local_component_sizes(ltx_root))[2] == pytest.approx(before + 0.9 + 0.4)


def test_sub_byte_floats_count_parameters_by_bit_width(tmp_path):
    # safetensors packs sub-byte dtypes: shape is the logical element count, bytes = numel * bits / 8
    header = {
        "__metadata__": {},
        "f4": {"dtype": "F4", "shape": [4, 16], "data_offsets": [0, 32]},
        "f6a": {"dtype": "F6_E2M3", "shape": [8], "data_offsets": [32, 38]},
        "f6b": {"dtype": "F6_E3M2", "shape": [16], "data_offsets": [38, 50]},
        "scale": {"dtype": "F8_E8M0", "shape": [4], "data_offsets": [50, 54]},
    }
    raw = json.dumps(header).encode()
    path = tmp_path / "transformer" / "diffusion_pytorch_model.safetensors"
    path.parent.mkdir(parents = True)
    path.write_bytes(struct.pack("<Q", len(raw)) + raw)
    size = dcs.local_component_sizes(tmp_path).get("transformer")
    assert size.float_params == 64 + 8 + 16 + 4
    assert size.nonfloat_bytes == 0
    assert size.resident_bytes(2) == (64 + 8 + 16 + 4) * 2  # a bf16 load budgets 2 bytes per param
    assert dcs.FLOAT_BITS["F4"] == 4 and dcs.FLOAT_BITS["F6_E2M3"] == dcs.FLOAT_BITS["F6_E3M2"] == 6


# ---- review round 5 ----


def test_unresolved_ref_probes_the_hub_and_falls_back_to_local(tmp_path, monkeypatch):
    fam = _fam("wan2.2-ti2v-5b")
    snap = tmp_path / ("a" * 40)
    _stub_for_table(snap, fam)
    _component(snap, "transformer", fam.bf16_components_gb[0] + 5.0)
    monkeypatch.setattr(dcs, "local_snapshot_dir", lambda repo, cache_dir = None: snap)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising = False)
    monkeypatch.delenv("UNSLOTH_DIFFUSION_HEADER_FETCH", raising = False)
    monkeypatch.setattr(dcs, "resolve_ref", lambda *a, **k: None)
    fresh_root = tmp_path / "fresh"
    _stub_for_table(fresh_root, fam)
    hub = [dcs.local_component_sizes(fresh_root)]
    calls = []
    monkeypatch.setattr(dcs, "hub_component_sizes", lambda repo, **k: calls.append(repo) or hub[0])
    # head unknown: the Hub answers
    assert dcs.video_components_gb(fam, "o/variant", allow_hub = True) == fam.bf16_components_gb
    assert calls == ["o/variant"]
    # head unknown and the Hub probe fails too: the cached local sizes stand (no refusal on a network blip)
    hub[0] = None
    got = dcs.video_components_gb(fam, "o/variant", allow_hub = True)
    assert got[0] == round(fam.bf16_components_gb[0] + 5.0, 1) and len(calls) == 2


def test_extra_text_encoders_need_the_family_or_manifest(tmp_path):
    wan = _fam("wan2.2-ti2v-5b")
    _stub_for_table(tmp_path, wan)
    base = dcs.video_header_terms(wan, dcs.local_component_sizes(tmp_path))[1]
    _component(tmp_path, "text_encoder_2", 1.0, stem = "model")
    _component(tmp_path, "text_encoder_3", 2.0, stem = "model")
    assert dcs.video_header_terms(wan, dcs.local_component_sizes(tmp_path))[1] == pytest.approx(base)
    (tmp_path / "model_index.json").write_text(
        json.dumps({"text_encoder": ["t", "A"], "text_encoder_2": ["t", "B"], "text_encoder_3": [None, None]})
    )
    assert dcs.video_header_terms(wan, dcs.local_component_sizes(tmp_path))[1] == pytest.approx(base + 1.0)
    # HunyuanVideo-1.5 loads its ByT5 text_encoder_2 without a manifest
    hv = _fam("hunyuanvideo-1.5")
    root = tmp_path / "hv"
    _stub_for_table(root, hv)
    before = dcs.video_header_terms(hv, dcs.local_component_sizes(root))[1]
    _component(root, "text_encoder_2", 0.44, stem = "model")
    assert dcs.video_header_terms(hv, dcs.local_component_sizes(root))[1] == pytest.approx(before - 0.1 + 0.44)


def test_same_stem_order_follows_the_loader(tmp_path):
    for folder in ("text_encoder", "transformer"):
        d = tmp_path / folder
        stem = "model" if folder == "text_encoder" else "diffusion_pytorch_model"
        _write_header(d / f"{stem}.safetensors", {"u": ("BF16", 10)})
        _write_header(d / f"{stem}-00001-of-00001.safetensors", {"s": ("BF16", 999)})
        _index(d / f"{stem}.safetensors.index.json", [f"{stem}-00001-of-00001.safetensors"])
    sizes = dcs.local_component_sizes(tmp_path)
    # transformers folder: the unsharded file first; diffusers folder: the index first
    assert sizes.get("text_encoder").files == ("model.safetensors",)
    assert sizes.get("transformer").files == ("diffusion_pytorch_model-00001-of-00001.safetensors",)


# ---- review round 6 ----


def test_hub_skips_folders_the_manifest_does_not_load(monkeypatch):
    files = [
        "model_index.json",
        "transformer/diffusion_pytorch_model.safetensors",
        "text_encoder/model.safetensors",
        "vae/diffusion_pytorch_model.safetensors",
        "connectors/diffusion_pytorch_model.safetensors",
        "vocoder/diffusion_pytorch_model.safetensors",
        "latent_upsampler/diffusion_pytorch_model.safetensors",
    ]
    ok = {"__metadata__": {}, "w": {"dtype": "BF16", "shape": [10], "data_offsets": [0, 20]}}
    # vocoder is null in the manifest and its header fails: not loaded, so not fatal
    _hub_with(monkeypatch, files, lambda f: None if f.startswith("vocoder/") else ok)
    manifest = json.dumps(
        {"transformer": ["d", "T"], "text_encoder": ["t", "E"], "vae": ["d", "V"], "connectors": ["l", "C"], "vocoder": [None, None]}
    )
    monkeypatch.setattr(dcs, "_fetch_capped_text", lambda *a, **k: manifest)
    sizes = dcs.hub_component_sizes("o/r")
    # latent_upsampler is attached by Studio outside the manifest, so it is still read
    assert sizes is not None and sizes.names() == ("connectors", "latent_upsampler", "text_encoder", "transformer", "vae")
    # a LOADED folder that fails stays fatal
    _hub_with(monkeypatch, files, lambda f: None if f.startswith("connectors/") else ok)
    monkeypatch.setattr(dcs, "_fetch_capped_text", lambda *a, **k: manifest)
    assert dcs.hub_component_sizes("o/r") is None


def test_loaded_role_with_unreadable_size_makes_its_term_unknown(tmp_path, monkeypatch):
    ltx = _fam("ltx-2")  # loads connectors and vocoder
    _stub_for_table(tmp_path, ltx)
    _component(tmp_path, "connectors", 1.0)
    import shutil

    shutil.rmtree(tmp_path / "vocoder")  # the stub's safetensors vocoder becomes a .bin one
    (tmp_path / "vocoder").mkdir()
    (tmp_path / "vocoder" / "diffusion_pytorch_model.bin").write_bytes(b"\0" * 8)
    sizes = dcs.local_component_sizes(tmp_path)
    assert "vocoder" in sizes.unknown
    assert dcs.video_header_terms(ltx, sizes)[2] is None  # not the remaining folders summed
    # a loaded extra text encoder stored as .bin: the text term is unknown
    hv = _fam("hunyuanvideo-1.5")
    root = tmp_path / "hv"
    _stub_for_table(root, hv)
    shutil.rmtree(root / "text_encoder_2")
    (root / "text_encoder_2").mkdir()
    (root / "text_encoder_2" / "pytorch_model.bin").write_bytes(b"\0" * 8)
    assert dcs.video_header_terms(hv, dcs.local_component_sizes(root))[1] is None
    # an unloaded leftover .bin folder changes nothing
    wan = _fam("wan2.2-ti2v-5b")
    w = tmp_path / "wan"
    _stub_for_table(w, wan)
    (w / "vocoder").mkdir()
    (w / "vocoder" / "diffusion_pytorch_model.bin").write_bytes(b"\0" * 8)
    assert None not in dcs.video_header_terms(wan, dcs.local_component_sizes(w))
    # end to end: a variant whose loaded companion is unknown takes the table for that term
    monkeypatch.setattr(dcs, "local_snapshot_dir", lambda repo, cache_dir = None: tmp_path)
    monkeypatch.setattr(dcs, "resolve_ref", lambda *a, **k: tmp_path.name)
    monkeypatch.setattr(dcs, "hub_component_sizes", lambda *a, **k: None)
    # ... and the readable siblings stay a lower bound: never below their known sum
    known = dcs.video_header_parts(ltx, dcs.local_component_sizes(tmp_path))[2]
    assert known[1] is False
    got = dcs.video_components_gb(ltx, "o/ltx2-variant", allow_hub = True)[2]
    assert got == max(ltx.bf16_components_gb[2], round(known[0], 1))


def test_no_index_fallback_takes_only_loader_names(tmp_path):
    for name in ("adapter_model.safetensors", "weights.safetensors", "diffusion_pytorch_model.fp16.safetensors"):
        assert dcs.select_weight_files([name], folder = "transformer") == []
    _write_header(tmp_path / "transformer" / "adapter_model.safetensors", {"a": ("BF16", 10)})
    assert dcs.local_component_sizes(tmp_path) is None
    for stem in ("diffusion_pytorch_model", "model", "pytorch_model"):
        assert dcs.select_weight_files([f"{stem}.safetensors", "adapter_model.safetensors"]) == [f"{stem}.safetensors"]


def test_local_index_read_is_capped(tmp_path, monkeypatch):
    t = tmp_path / "transformer"
    _write_header(t / "diffusion_pytorch_model-00001-of-00001.safetensors", {"a": ("BF16", 10)})
    _index(t / "diffusion_pytorch_model.safetensors.index.json", ["diffusion_pytorch_model-00001-of-00001.safetensors"])
    assert dcs.local_component_sizes(tmp_path) is not None
    monkeypatch.setattr(dcs, "MAX_INDEX_BYTES", 8)  # the index is now over the cap: never read, unknown
    reads = []
    real = Path.read_text
    monkeypatch.setattr(Path, "read_text", lambda self, *a, **k: reads.append(self.name) or real(self, *a, **k))
    sizes = dcs.local_component_sizes(tmp_path)
    assert sizes is None and "diffusion_pytorch_model.safetensors.index.json" not in reads


# ---- review round 7 ----


def test_snapshot_missing_a_loaded_role_is_incomplete(tmp_path, monkeypatch):
    import shutil

    for name, folder, term in (
        ("ltx-2.5", "latent_upsampler", 2),  # Studio-attached, never in model_index.json
        ("ltx-2", "vocoder", 2),  # family rule
        ("hunyuanvideo-1.5", "text_encoder_2", 1),  # family rule
    ):
        fam = _fam(name)
        root = tmp_path / name
        _stub_for_table(root, fam)
        assert folder in dcs.video_loaded_roles(fam)
        shutil.rmtree(root / folder)
        # the absent loaded role makes its term unknown (table), never the rest summed
        assert dcs.video_header_terms(fam, dcs.local_component_sizes(root))[term] is None
        monkeypatch.setattr(dcs, "local_snapshot_dir", lambda repo, cache_dir = None, root = root: root)
        monkeypatch.setattr(dcs, "resolve_ref", lambda *a, root = root, **k: root.name)
        calls = []
        monkeypatch.setattr(dcs, "hub_component_sizes", lambda repo, **k: calls.append(repo))
        got = dcs.video_components_gb(fam, "o/variant", allow_hub = True)
        assert calls == ["o/variant"], name  # incomplete local snapshot: the Hub is probed
        # the Hub failed too: the known part plus an allowance for the absent folder, never below the table
        known = dcs.video_header_parts(fam, dcs.local_component_sizes(root))[term][0]
        allowance = 1.0 if folder == "latent_upsampler" else fam.bf16_components_gb[term]
        assert got[term] == max(fam.bf16_components_gb[term], round(known + allowance, 1)), name
        assert got[term] >= fam.bf16_components_gb[term]
    # a manifest-loaded extra counts too
    wan = _fam("wan2.2-ti2v-5b")
    w = tmp_path / "wan"
    _stub_for_table(w, wan)
    (w / "model_index.json").write_text(json.dumps({"vae": ["d", "V"], "connectors": ["l", "C"]}))
    assert "connectors" in dcs.video_loaded_roles(wan, dcs.local_component_sizes(w))
    assert dcs.video_header_terms(wan, dcs.local_component_sizes(w))[2] is None


def test_index_fetch_is_bounded_by_the_deadline_while_a_peer_trickles():
    import threading as _threading
    import time as _time

    release = _threading.Event()

    class _Trickle(_Resp):
        def iter_content(self, chunk_size = 1):
            yield b"{"
            release.wait(30)  # a peer that never completes the chunk
            yield b"}"

    resp = _Trickle([])
    start = _time.monotonic()
    with pytest.raises(TimeoutError):
        dcs._fetch_capped_text(_Session(resp), "u", {}, 30.0, _time.monotonic() + 0.5)
    assert _time.monotonic() - start < 5  # the whole fetch, not one 30 s read
    assert resp.closed  # interrupted (no urllib3 raw here, so closed)
    release.set()


# ---- review round 8 ----


def test_ltx_bundled_connector_tensors_leave_the_denoiser_term(tmp_path):
    fam = _fam("ltx-2.5")
    _stub_for_table(tmp_path, fam)
    clean = dcs.video_header_terms(fam, dcs.local_component_sizes(tmp_path))[0]
    # a variant whose transformer/ also carries the connector tensors (single-file style keys, both prefixes)
    _write_header(
        tmp_path / "transformer" / "diffusion_pytorch_model.safetensors",
        {
            "w": ("BF16", int(round(fam.bf16_components_gb[0] * _GB / 2))),
            "video_embeddings_connector.x": ("BF16", 1_000_000_000),
            "model.diffusion_model.text_embedding_projection.y": ("BF16", 500_000_000),
            "connectors.z": ("F32", 250_000_000),
        },
    )
    size = dcs.local_component_sizes(tmp_path).get("transformer")
    assert size.bundled_connector_float_params == 1_750_000_000
    assert dcs.video_header_terms(fam, dcs.local_component_sizes(tmp_path))[0] == pytest.approx(clean)
    # the same split for LTX-2; a non-LTX family keeps every tensor
    assert dcs.video_header_terms(_fam("ltx-2"), dcs.local_component_sizes(tmp_path))[0] == pytest.approx(clean)
    wan = dcs.video_header_terms(_fam("wan2.2-ti2v-5b"), dcs.local_component_sizes(tmp_path))[0]
    assert wan == pytest.approx(clean + 3.5)
    # base plans are unchanged: the table still answers for the base repo
    from unittest import mock

    with mock.patch.object(dcs, "local_snapshot_dir", lambda repo, cache_dir = None: tmp_path):
        assert dcs.video_components_gb(fam) == fam.bf16_components_gb


def test_latent_upsampler_is_loaded_only_when_attached(tmp_path, monkeypatch):
    import shutil

    from core.inference import video_ltx25

    fam = _fam("ltx-2.5")
    _stub_for_table(tmp_path, fam)
    monkeypatch.setattr(video_ltx25, "two_stage_enabled", lambda: True)
    assert "latent_upsampler" in dcs.video_loaded_roles(fam)
    assert "latent_upsampler" not in dcs.video_loaded_roles(fam, dev = True)
    shutil.rmtree(tmp_path / "latent_upsampler")
    sizes = dcs.local_component_sizes(tmp_path)
    # two-stage distilled: the absent upsampler makes the term unknown; dev: it is not loaded, the term is known
    assert dcs.video_header_terms(fam, sizes)[2] is None
    assert dcs.video_header_terms(fam, sizes, dev = True)[2] is not None
    monkeypatch.setattr(video_ltx25, "two_stage_enabled", lambda: False)
    assert "latent_upsampler" not in dcs.video_loaded_roles(fam)
    assert dcs.video_header_terms(fam, sizes)[2] is not None


def test_load_pipeline_passes_the_dev_flag():
    import inspect

    from core.inference import video

    src = inspect.getsource(video.VideoBackend.load_pipeline)
    call = src[src.index("components = video_components_gb(") :]
    call = call[: call.index("\n        )")]
    assert "dev = _ltx25_dev_load(" in call


def test_moe_second_expert_is_optional_unless_shipped(tmp_path):
    moe = _fam("wan2.2-t2v-a14b")
    _component(tmp_path, "transformer", 28.6)
    _component(tmp_path, "text_encoder", 11.4, stem = "model")
    _component(tmp_path, "vae", 0.254, "F32")
    sizes = dcs.local_component_sizes(tmp_path)
    assert "transformer_2" not in dcs.video_loaded_roles(moe, sizes)
    assert dcs.video_header_terms(moe, sizes)[0] == pytest.approx(28.6)  # the known first expert is kept
    # named by the manifest but absent: loaded, so the snapshot is incomplete and the term unknown
    (tmp_path / "model_index.json").write_text(json.dumps({"transformer": ["d", "W"], "transformer_2": ["d", "W"]}))
    sizes = dcs.local_component_sizes(tmp_path)
    assert "transformer_2" in dcs.video_loaded_roles(moe, sizes)
    assert dcs.video_header_terms(moe, sizes)[0] is None
    # present: both experts count
    _component(tmp_path, "transformer_2", 28.6)
    assert dcs.video_header_terms(moe, dcs.local_component_sizes(tmp_path))[0] == pytest.approx(57.2)
    # a non-MoE family never counts a stray transformer_2
    assert dcs.video_header_terms(_fam("wan2.2-ti2v-5b"), dcs.local_component_sizes(tmp_path))[0] == pytest.approx(28.6)


# ---- review round 9 ----


def test_partial_term_keeps_its_known_sum_as_a_lower_bound(tmp_path, monkeypatch):
    import shutil

    ltx = _fam("ltx-2")
    _stub_for_table(tmp_path, ltx)
    _component(tmp_path, "vae", ltx.bf16_components_gb[2] + 4.0)  # readable siblings far above the table term
    shutil.rmtree(tmp_path / "vocoder")
    (tmp_path / "vocoder").mkdir()
    (tmp_path / "vocoder" / "diffusion_pytorch_model.bin").write_bytes(b"\0" * 8)
    known, complete = dcs.video_header_parts(ltx, dcs.local_component_sizes(tmp_path))[2]
    assert not complete and known > ltx.bf16_components_gb[2]
    monkeypatch.setattr(dcs, "local_snapshot_dir", lambda repo, cache_dir = None: tmp_path)
    monkeypatch.setattr(dcs, "resolve_ref", lambda *a, **k: tmp_path.name)
    monkeypatch.setattr(dcs, "hub_component_sizes", lambda *a, **k: None)
    got = dcs.video_components_gb(ltx, "o/variant", allow_hub = True)
    assert got[2] == round(known, 1)  # never below what did read
    assert dcs.reconcile_with_table((None, None, None), (1.0, 1.0, 1.0), is_base = False, lower = (None, 2.04, None)) == (1.0, 2.0, 1.0)


def test_dev_table_term_drops_the_upsampler_before_reconciling(monkeypatch, tmp_path):
    from core.inference import video_ltx25

    fam = _fam("ltx-2.5")
    table = fam.bf16_components_gb
    monkeypatch.setattr(dcs, "local_snapshot_dir", lambda repo, cache_dir = None: None)
    monkeypatch.setattr(video_ltx25, "two_stage_enabled", lambda: True)
    assert dcs.video_components_gb(fam) == table
    # dev: the table's companion term loses the upsampler constant, so it compares with a header term without it
    dev = dcs.video_components_gb(fam, dev = True)
    assert dev == (table[0], table[1], round(table[2] - video_ltx25.LTX25_LATENT_UPSAMPLER_GB, 3))
    monkeypatch.setattr(video_ltx25, "two_stage_enabled", lambda: False)
    assert dcs.video_components_gb(fam) == dev
    # a dev variant whose headers equal the table minus the upsampler agrees, no mismatch recorded
    monkeypatch.setattr(dcs, "_MISMATCHES", {})
    _stub_for_table(tmp_path, fam)  # two-stage off: no upsampler folder in the stub
    _component(tmp_path, "vae", 0.1 + 0.0)  # shrink the stub's companion term by the upsampler the table includes
    vae_known = dcs.video_header_parts(fam, dcs.local_component_sizes(tmp_path))[2][0]
    _component(tmp_path, "vae", 0.1 + dev[2] - vae_known)
    monkeypatch.setattr(dcs, "local_snapshot_dir", lambda repo, cache_dir = None: tmp_path)
    monkeypatch.setattr(dcs, "resolve_ref", lambda *a, **k: tmp_path.name)
    got = dcs.video_components_gb(fam, "o/ltx25-variant", allow_hub = True, dev = True)
    assert got[2] == pytest.approx(dev[2], abs = 0.06) and not dcs.table_mismatches()


def test_load_pipeline_no_longer_subtracts_the_upsampler_itself():
    import inspect

    from core.inference import video

    src = inspect.getsource(video.VideoBackend.load_pipeline)
    assert "LTX25_LATENT_UPSAMPLER_GB" not in src and "term_part_gb" not in src
    assert "dev = _ltx25_dev_load(" in src


def test_dev_load_never_probes_the_upsampler_on_the_hub(monkeypatch):
    from core.inference import video_ltx25

    fam = _fam("ltx-2.5")
    monkeypatch.setattr(dcs, "local_snapshot_dir", lambda repo, cache_dir = None: None)
    seen = []
    monkeypatch.setattr(dcs, "hub_component_sizes", lambda repo, components = None, **k: seen.append(tuple(components)))
    monkeypatch.setattr(video_ltx25, "two_stage_enabled", lambda: True)
    dcs.video_components_gb(fam, "o/v", allow_hub = True, dev = True)
    dcs.video_components_gb(fam, "o/v", allow_hub = True, dev = False)
    assert "latent_upsampler" not in seen[0] and "latent_upsampler" in seen[1]
    monkeypatch.setattr(video_ltx25, "two_stage_enabled", lambda: False)
    dcs.video_components_gb(fam, "o/v", allow_hub = True)
    assert "latent_upsampler" not in seen[2]


def test_manifest_is_authoritative_for_the_second_expert(tmp_path):
    moe = _fam("wan2.2-t2v-a14b")
    _component(tmp_path, "transformer", 28.6)
    _component(tmp_path, "transformer_2", 28.6)  # a stale folder
    _component(tmp_path, "text_encoder", 11.4, stem = "model")
    _component(tmp_path, "vae", 0.254, "F32")
    assert dcs.video_header_terms(moe, dcs.local_component_sizes(tmp_path))[0] == pytest.approx(57.2)  # no manifest
    (tmp_path / "model_index.json").write_text(json.dumps({"transformer": ["d", "W"], "transformer_2": [None, None]}))
    sizes = dcs.local_component_sizes(tmp_path)
    assert "transformer_2" not in dcs.video_loaded_roles(moe, sizes)
    assert dcs.video_header_terms(moe, sizes)[0] == pytest.approx(28.6)


def test_checkpoint_fingerprint_tracks_the_exact_weights(tmp_path, monkeypatch):
    root = tmp_path / "pipe"
    _component(root, "transformer", 0.001)
    (root / "transformer" / "config.json").write_text('{"a": 1}')
    (root / "model_index.json").write_text("{}")
    first = dcs.checkpoint_fingerprint(str(root))
    assert first and first == dcs.checkpoint_fingerprint(str(root))
    (root / "transformer" / "config.json").write_text('{"a": 2}')
    assert dcs.checkpoint_fingerprint(str(root)) != first
    # a repo id resolves to its cache snapshot commit
    from utils import hf_cache_settings

    cache = tmp_path / "hub"
    snap = cache / "models--o--r" / "snapshots" / ("f" * 40)
    snap.mkdir(parents = True)
    (snap / "model_index.json").write_text("{}")
    (cache / "models--o--r" / "refs").mkdir()
    (cache / "models--o--r" / "refs" / "main").write_text("f" * 40)
    monkeypatch.setattr(hf_cache_settings, "active_hf_hub_cache", lambda: str(cache))
    assert dcs.checkpoint_fingerprint("o/r") == "f" * 40
    assert dcs.checkpoint_fingerprint("o/missing") is None


def test_calibration_key_carries_the_revision():
    import inspect

    from core.inference import video

    src = inspect.getsource(video)
    key_at = src.index("video_stream_residency.calibration_key(")
    call = src[key_at : src.index("\n                    )", key_at)]
    assert "revision = _calibration_revision(state)" in call
    a = vsr.calibration_key("ltx-2", "r", "group", "cuda:0", revision = "a" * 40)
    assert a != vsr.calibration_key("ltx-2", "r", "group", "cuda:0", revision = "b" * 40)


# ---- review round 10 ----


def test_manifest_predicate_matches_the_loader(tmp_path):
    import inspect

    from core.inference import diffusion

    assert dcs.manifest_entry_loaded("vae", ["diffusers", "AutoencoderKL"])
    for spec in (["diffusers"], ["diffusers", None], [None, "X"], [None, None], "diffusers", None, []):
        assert not dcs.manifest_entry_loaded("vae", spec), spec
    assert not dcs.manifest_entry_loaded("_class_name", ["a", "b"])
    (tmp_path / "model_index.json").write_text(
        json.dumps({"_class_name": "P", "vae": ["d", "V"], "vocoder": ["d", None], "connectors": ["l"], "text_encoder": [None, "T"]})
    )
    assert dcs._manifest_names(tmp_path) == frozenset({"vae"})
    # the loader's staging selection uses the same predicate (no second copy to drift)
    src = inspect.getsource(diffusion)
    assert "manifest_entry_loaded(name, spec)" in src
    assert "len(spec) >= 2" not in src


def test_auto_cache_decision_is_shared_and_retags_the_calibration():
    import inspect

    from core.inference import diffusion_cache, video

    assert diffusion_cache.auto_step_cache_mode(diffusion_cache.FBCACHE_MIN_STEPS) == diffusion_cache.TC_FBCACHE
    assert diffusion_cache.auto_step_cache_mode(diffusion_cache.FBCACHE_MIN_STEPS - 1) is None
    assert "auto_step_cache_mode(steps)" in inspect.getsource(diffusion_cache.maybe_toggle_step_cache)
    src = inspect.getsource(video)
    key_at = src.index("video_stream_residency.calibration_key(")
    call = src[key_at : src.index("\n                    )", key_at)]
    assert "transformer_cache = request_cache_mode" in call
    assert src.index("request_cache_mode = auto_step_cache_mode(steps)") < key_at
    toggle_at = src.index("toggled = maybe_toggle_step_cache(")
    assert src.index('video_stream_residency.retag(state.pipe, "transformer_cache", toggled)') > toggle_at
    # the retag rewrites only that entry, so the recorded peak lands under the mode that ran
    pipe = _Pipe()
    key = vsr.calibration_key("wan", "r", "group", "cuda:0", transformer_cache = "fbcache", dtype = "bf16")
    pipe.transformer._unsloth_video_calibration_key = key
    new = vsr.retag(pipe, "transformer_cache", None)
    assert new == vsr.calibration_key("wan", "r", "group", "cuda:0", transformer_cache = None, dtype = "bf16")
    assert pipe.transformer._unsloth_video_calibration_key == new
    assert vsr.retag(_Pipe(), "transformer_cache", None) is None


def test_retagged_peak_is_recorded_under_the_mode_that_ran(monkeypatch):
    import torch

    monkeypatch.setattr(vsr, "_PEAKS", {})
    alloc = {"now": 0, "peak": 0}
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda dev = None: None)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda dev = None: alloc["now"])
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda dev = None: alloc["peak"])
    predicted = vsr.calibration_key("wan", "r", "group", "cuda:0", transformer_cache = "fbcache")
    pipe = _Pipe()
    vsr.arm_request(pipe, key = predicted, width = 8, height = 8, frames = 1, device = "cuda:0")
    vsr.retag(pipe, "transformer_cache", None)  # the runner-aware toggle ran uncached
    alloc["peak"] = 500 << 20
    vsr.record_request_peak(pipe)
    ran = vsr.with_extra(predicted, "transformer_cache", None)
    assert vsr.measured_extra_mib(ran, 64) == 500
    assert vsr.measured_extra_mib(predicted, 64) is None


# ---- review round 11 ----


def test_pre_download_plan_threads_the_dev_flag():
    import inspect

    from core.inference import video

    seed = inspect.getsource(video._video_seed_stays_resident)
    assert "dev = ltx25_dev" in seed
    planned = inspect.getsource(video.VideoBackend._video_planned_auto_denoiser_scheme)
    assert planned.count("ltx25_dev = ltx25_dev,") == 2  # both seed checks
    for caller in (video.VideoBackend._run_load, video.VideoBackend.download_plan):
        src = inspect.getsource(caller)
        call = src[src.index("_video_planned_auto_denoiser_scheme(") :]
        assert "ltx25_dev = ltx25_dev" in call[: call.index("DENOISER_SEED_DECLINED")], caller.__name__


def test_unknown_loaded_folder_gets_an_allowance(tmp_path, monkeypatch):
    import shutil

    ltx = _fam("ltx-2")
    _stub_for_table(tmp_path, ltx)
    _component(tmp_path, "vae", ltx.bf16_components_gb[2] + 4.0)  # known siblings above the table term
    shutil.rmtree(tmp_path / "vocoder")
    (tmp_path / "vocoder").mkdir()
    (tmp_path / "vocoder" / "diffusion_pytorch_model.bin").write_bytes(b"\0" * 300_000_000)  # 0.3 GB on disk
    sizes = dcs.local_component_sizes(tmp_path)
    assert dict(sizes.unknown_bytes)["vocoder"] == 300_000_000
    known = dcs.video_header_parts(ltx, sizes)[2][0]
    monkeypatch.setattr(dcs, "local_snapshot_dir", lambda repo, cache_dir = None: tmp_path)
    monkeypatch.setattr(dcs, "resolve_ref", lambda *a, **k: tmp_path.name)
    monkeypatch.setattr(dcs, "hub_component_sizes", lambda *a, **k: None)
    got = dcs.video_components_gb(ltx, "o/variant", allow_hub = True)
    assert got[2] == round(known + 0.3, 1)  # the .bin's stored bytes are its allowance, not dropped
    # no file and no own constant: one whole table term for the unestimated folders
    assert dcs._term_lower_bound(2.0, ("a", "b"), sizes, 5.0) == 7.0
    assert dcs._term_lower_bound(2.0, ("vocoder",), sizes, 5.0) == pytest.approx(2.3)
    assert dcs._term_lower_bound(2.0, ("latent_upsampler",), sizes, 5.0) == pytest.approx(3.0)
    assert dcs._term_lower_bound(2.0, ("a",), sizes, None) is None


def test_single_file_loads_size_companions_from_the_resolved_base():
    import inspect

    from core.inference import video

    src = inspect.getsource(video.VideoBackend.load_pipeline)
    call = src[src.index("components = video_components_gb(") :]
    call = call[: call.index("\n        )")]
    assert 'repo_id if kind == "pipeline" else base,' in call


def test_calibration_revision_includes_the_companion_base(monkeypatch):
    from types import SimpleNamespace

    from core.inference import video

    fps = {"u/ltx-gguf": "g" * 40, "Lightricks/LTX-2": "b" * 40, "someone/ltx2-base": "c" * 40}
    monkeypatch.setattr(dcs, "checkpoint_fingerprint", lambda repo, filename = None: fps.get(repo))

    def rev(kind, base):
        state = SimpleNamespace(pipe = SimpleNamespace(), repo_id = "u/ltx-gguf", gguf_filename = "x.gguf", kind = kind, base_repo = base)
        return video._calibration_revision(state)

    assert rev("pipeline", "Lightricks/LTX-2") == "g" * 40
    gguf_default = rev("gguf", "Lightricks/LTX-2")
    assert gguf_default == f"{'g' * 40}|base=Lightricks/LTX-2@{'b' * 40}"
    assert rev("gguf", "someone/ltx2-base") != gguf_default


def test_transformers_folders_recognise_only_transformers_stems(tmp_path):
    for name in (
        "diffusion_pytorch_model.safetensors",
        "pytorch_model.safetensors",
    ):
        assert dcs._pick_files([name], lambda n: "", "text_encoder") == [], name
        assert dcs._pick_files([name], lambda n: "", "transformer") == [name], name
    assert dcs._pick_files(["model.safetensors"], lambda n: "", "image_encoder") == ["model.safetensors"]
    # pytorch_model.bin: what transformers reads next, not sized here -> unknown, with its bytes as the allowance
    te = tmp_path / "text_encoder"
    te.mkdir()
    (te / "pytorch_model.bin").write_bytes(b"\0" * 1000)
    _write_header(te / "diffusion_pytorch_model.safetensors", {"x": ("BF16", 10)})
    _component(tmp_path, "vae", 0.1)
    sizes = dcs.local_component_sizes(tmp_path)
    assert sizes.get("text_encoder") is None and "text_encoder" in sizes.unknown
