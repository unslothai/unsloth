# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Pre-quantized SAM 3.1 detector files for Mask from text: the precision choice (UNSLOTH_SAM3_PRECISION, auto, CUDA
only), the per-precision download specs and their bf16 fallback when a file is not published, which weights a load
uses, the header checks, and a real build -> save -> load round trip on a tiny random Sam3Model (dequantized on the
CPU; quantized kernels on CUDA when a card is present)."""

from __future__ import annotations

import dataclasses
import json
import re
from pathlib import Path

import pytest

from core.inference import diffusion_segment as seg
from core.inference import diffusion_segment_quant as sq


@pytest.fixture(autouse = True)
def _clean_env(monkeypatch):
    for name in (sq.ENV_PRECISION, sq.ENV_INT8_CONVROT, seg.ENV_PATH_OVERRIDE, seg.ENV_VARIANT):
        monkeypatch.delenv(name, raising = False)
    monkeypatch.setattr(seg, "_BROKEN_PREQUANT", set())


def _cuda(monkeypatch, capability):
    """Pretend ``cuda:N`` cards have ``capability`` (None: no CUDA) and torchao can read the files."""
    monkeypatch.setattr(sq, "_cuda_capability", lambda device: capability)
    from core.inference import diffusion_transformer_quant as tq
    from core.inference import prequant_safetensors as pst

    monkeypatch.setattr(tq, "torchao_unavailable_reason", lambda: None)
    monkeypatch.setattr(pst, "safetensors_prequant_supported", lambda: True)


# ── precision choice ───────────────────────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "value, expected",
    [
        ("", "auto"),
        ("auto", "auto"),
        (" BF16 ", "bf16"),
        ("bfloat16", "bf16"),
        ("fp8", "fp8"),
        ("INT8", "int8"),
        ("nf4", "auto"),
    ],
)
def test_requested_precision(monkeypatch, value, expected):
    monkeypatch.setenv(sq.ENV_PRECISION, value)
    assert sq.requested_precision() == expected


def test_auto_follows_the_gated_order_on_a_tight_cuda_card(monkeypatch):
    import torch

    _cuda(monkeypatch, (9, 0))
    assert sq.AUTO_ORDER, "the gate passed at least one quantized arm"
    tight = dict(memory_tight = True)
    assert sq.resolve_precision("cuda:0", torch.bfloat16, **tight)[0] == sq.AUTO_ORDER[0]
    # Only what is on disk (or published) counts.
    assert sq.resolve_precision("cuda:0", available = (), **tight)[0] == "bf16"
    assert (
        sq.resolve_precision("cuda:0", available = (sq.AUTO_ORDER[-1],), **tight)[0]
        == sq.AUTO_ORDER[-1]
    )
    monkeypatch.setattr(sq, "AUTO_ORDER", ())
    precision, why = sq.resolve_precision("cuda:0", **tight)
    assert precision == "bf16" and "gate" in why


def test_auto_runs_bf16_while_the_card_has_room(monkeypatch):
    """The 8-bit files are about 2.2x slower per Find than bf16 (model card, B200), so auto only trades speed for
    memory when the card is short of it. An explicit int8 / fp8 is honoured either way."""
    import torch

    _cuda(monkeypatch, (9, 0))
    for args in ((), (torch.bfloat16,)):
        precision, why = sq.resolve_precision("cuda:0", *args)
        assert precision == "bf16" and "auto" in why
        assert sq.resolve_precision("cuda:0", *args, memory_tight = False)[0] == "bf16"
    for value in ("int8", "fp8"):
        monkeypatch.setenv(sq.ENV_PRECISION, value)
        assert sq.resolve_precision("cuda:0", torch.bfloat16)[0] == value
        assert sq.resolve_precision("cuda:0", torch.bfloat16, memory_tight = True)[0] == value
    monkeypatch.setenv(sq.ENV_PRECISION, "bf16")
    assert sq.resolve_precision("cuda:0", torch.bfloat16, memory_tight = True)[0] == "bf16"


def test_auto_memory_tight_reads_the_card_only_once_cuda_is_up(monkeypatch):
    import torch

    reads = []

    def free(ordinal):
        reads.append(ordinal)
        return free_total

    monkeypatch.setattr(seg, "_cuda_free_mib", free)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    free_total = (1024, 80_000)
    # No CUDA context in this process yet: nothing of ours is on a card, and planning must not create one.
    assert seg.auto_memory_tight("cuda:0") is False and reads == []
    assert seg.auto_memory_tight("cpu") is False and seg.auto_memory_tight("mps") is False
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    # 80 GB card: tight below 20% (16000 MiB) free once bf16's 2048 MiB are placed.
    free_total = (60_000, 80_000)
    assert seg.auto_memory_tight("cuda:1") is False and reads == [1]
    free_total = (16_000 + seg._BF16_RESIDENT_MIB - 1, 80_000)
    assert seg.auto_memory_tight("cuda:1") is True
    free_total = (16_000 + seg._BF16_RESIDENT_MIB, 80_000)
    assert seg.auto_memory_tight("cuda:1") is False
    # Small card: the 6 GiB floor.
    free_total = (seg._TIGHT_FREE_MIB + seg._BF16_RESIDENT_MIB - 1, 12_000)
    assert seg.auto_memory_tight("cuda:0") is True
    free_total = (None, None)
    assert seg.auto_memory_tight("cuda:0") is False


def test_a_bf16_pick_is_not_unloaded_as_tight_after_its_first_find(monkeypatch):
    # mem_get_info free drop with bf16 resident after a Find, A100 + torch 2.11 (dequantized INT8 file, the larger).
    import torch

    measured_drop_mib = 2282
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    for total in (12_000, 24_000, 40_000, 80_000):
        floor = max(seg._TIGHT_FREE_MIB, int(total * seg._TIGHT_FREE_FRACTION))
        free = floor + seg._BF16_RESIDENT_MIB
        monkeypatch.setattr(seg, "_cuda_free_mib", lambda ordinal, f = free, t = total: (f, t))
        assert seg.auto_memory_tight("cuda:0") is False
        assert not seg.vram_is_tight(free - measured_drop_mib, total), total


def test_a_loaded_segmenter_keeps_its_memory_reading(monkeypatch):
    """Loaded on a card, the free memory read on the next call no longer counts the segmenter itself: the choice made
    at load is reused, so a bf16 model is not swapped for int8 (or back) on every call."""
    seen = []
    plan = seg.WeightsPlan("bf16", "/m")
    monkeypatch.setattr(
        seg, "plan_weights", lambda *a, memory_tight, **k: seen.append(memory_tight) or plan
    )
    monkeypatch.setattr(seg, "auto_memory_tight", lambda device: True)
    segmenter = seg.MaskSegmenter(keep_alive_seconds = 0)
    segmenter._model, segmenter._path, segmenter.device, segmenter._dtype = (
        object(),
        "/m",
        "cuda:0",
        "bf16",
    )
    segmenter._weights_key, segmenter._memory_tight = ("/m", None, False), False
    segmenter._load("/m", "cuda:0", "bf16")
    assert seen == [False]
    # Another device (or nothing loaded) reads the card again.
    segmenter._model = None
    segmenter._memory_tight = None
    monkeypatch.setattr(segmenter, "_load_weights", lambda plan, *a: (object(), plan))
    import transformers

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: object())
    segmenter._load("/m", "cuda:0", "bf16")
    assert seen == [False, True] and segmenter._memory_tight is True
    segmenter._unload_locked("test")
    assert segmenter._memory_tight is None


def test_quantized_kernels_are_cuda_only(monkeypatch):
    import torch

    _cuda(monkeypatch, (10, 0))
    for value in ("auto", "fp8", "int8"):
        monkeypatch.setenv(sq.ENV_PRECISION, value)
        for device in ("cpu", "mps"):
            precision, why = sq.resolve_precision(device, torch.float32)
            assert precision == "bf16" and (value == "auto" or "CUDA only" in why), (value, device)
    # A card without bf16 runs the segmenter in fp16: dense.
    monkeypatch.setenv(sq.ENV_PRECISION, "int8")
    assert sq.resolve_precision("cuda:0", torch.float16)[0] == "bf16"


def test_capability_floors(monkeypatch):
    _cuda(monkeypatch, (8, 0))  # Ampere: int8 yes, fp8 no
    monkeypatch.setenv(sq.ENV_PRECISION, "fp8")
    precision, why = sq.resolve_precision("cuda:0")
    assert precision == "bf16" and "8.9" in why
    monkeypatch.setenv(sq.ENV_PRECISION, "int8")
    assert sq.resolve_precision("cuda:0")[0] == "int8"
    _cuda(monkeypatch, (8, 9))
    monkeypatch.setenv(sq.ENV_PRECISION, "fp8")
    assert sq.resolve_precision("cuda:0")[0] == "fp8"
    _cuda(monkeypatch, (7, 5))  # Turing
    monkeypatch.setenv(sq.ENV_PRECISION, "int8")
    assert sq.resolve_precision("cuda:0")[0] == "bf16"
    monkeypatch.setenv(sq.ENV_PRECISION, "bf16")
    _cuda(monkeypatch, (10, 0))
    assert sq.resolve_precision("cuda:0") == ("bf16", f"{sq.ENV_PRECISION}=bf16")


def test_a_missing_torchao_keeps_bf16(monkeypatch):
    _cuda(monkeypatch, (10, 0))
    from core.inference import diffusion_transformer_quant as tq

    monkeypatch.setattr(tq, "torchao_unavailable_reason", lambda: "no torchao")
    monkeypatch.setenv(sq.ENV_PRECISION, "int8")
    assert sq.resolve_precision("cuda:0") == ("bf16", "int8: no torchao")


# ── the variant table ──────────────────────────────────────────────────────────────────────────────────────────────
def test_sam31_lists_one_file_per_scheme_and_sam3_none():
    sam31, sam3 = seg.SEGMENTER_VARIANTS["sam3.1"], seg.SEGMENTER_VARIANTS["sam3"]
    names = {(a.precision, a.rotated): a.filename for a in sam31.prequant}
    assert names == {
        ("int8", True): "sam3.1-INT8-ConvRot.safetensors",
        ("int8", False): "sam3.1-INT8.safetensors",
        ("fp8", False): "sam3.1-FP8.safetensors",
    }
    for artifact in sam31.prequant:
        assert artifact.filename == sq.artifact_filename(
            "sam3.1", artifact.precision, rotated = artifact.rotated
        )
        # Every pre-quantized file is smaller than the bf16 one it replaces.
        assert 0 < artifact.size_bytes < seg.SAM31_SIZE_BYTES - seg.SAM31_CONFIG_BYTES
    assert sam3.prequant == () and sam3.prequant_revision is None
    rev = sam31.prequant_revision
    assert rev is None or rev == "main" or re.fullmatch(r"[0-9a-f]{40}", rev)


def test_each_file_has_a_download_spec_that_falls_back_to_bf16():
    sam31 = seg.SEGMENTER_VARIANTS["sam3.1"]
    for artifact in sam31.prequant:
        spec_id = seg.prequant_spec_id(sam31, artifact)
        spec = seg.SEGMENTER_SPECS[spec_id]
        assert spec.files == seg.SEGMENTER_CONFIG_FILES + (artifact.filename,)
        assert seg.SEGMENTER_WEIGHTS_FILE not in spec.files  # only the chosen weights file
        assert (spec.repo, spec.revision, spec.fallback) == (
            sam31.repo,
            sam31.prequant_revision,
            "sam3.1",
        )
        assert seg.split_spec_id(spec_id) == (sam31, artifact)
        assert seg._resolve_segmenter_id(spec_id) == spec_id
    assert seg.SEGMENTER_SPECS["sam3.1"].fallback is None
    with pytest.raises(ValueError):
        seg._resolve_segmenter_id("sam3.1@nf4")


def test_int8_is_the_convrot_file_unless_turned_off(monkeypatch):
    sam31 = seg.SEGMENTER_VARIANTS["sam3.1"]
    assert seg.variant_artifact(sam31, "int8").filename == "sam3.1-INT8-ConvRot.safetensors"
    monkeypatch.setenv(sq.ENV_INT8_CONVROT, "0")
    assert seg.variant_artifact(sam31, "int8").filename == "sam3.1-INT8.safetensors"
    assert seg.variant_artifact(sam31, "fp8").filename == "sam3.1-FP8.safetensors"
    assert seg.variant_artifact(seg.SEGMENTER_VARIANTS["sam3"], "int8") is None


def _published(monkeypatch, revision = "main"):
    """sam3.1 with its pre-quantized files published at ``revision``."""
    variant = dataclasses.replace(seg.SEGMENTER_VARIANTS["sam3.1"], prequant_revision = revision)
    monkeypatch.setitem(seg.SEGMENTER_VARIANTS, "sam3.1", variant)
    return variant


def test_download_plan_fetches_one_weights_file(monkeypatch):
    _cuda(monkeypatch, (9, 0))
    unpublished = dataclasses.replace(seg.SEGMENTER_VARIANTS["sam3.1"], prequant_revision = None)
    # Not published: the bf16 files, whatever the host.
    plan = seg.download_plan(unpublished, "cuda:0")
    assert (plan["spec"], plan["precision"], plan["size_bytes"]) == (
        "sam3.1",
        "bf16",
        seg.SAM31_SIZE_BYTES,
    )
    variant = _published(monkeypatch)
    # A card with room fetches bf16 (the faster path); a tight one the smaller gated file.
    assert seg.download_plan(variant, "cuda:0")["spec"] == "sam3.1"
    assert seg.download_plan(variant, "cuda:0", memory_tight = False)["spec"] == "sam3.1"
    plan = seg.download_plan(variant, "cuda:0", memory_tight = True)
    artifact = seg.variant_artifact(variant, sq.AUTO_ORDER[0])
    assert (
        plan["spec"] == seg.prequant_spec_id(variant, artifact)
        and plan["precision"] == sq.AUTO_ORDER[0]
    )
    assert plan["file"] == artifact.filename
    assert plan["size_bytes"] == artifact.size_bytes + seg.SAM31_CONFIG_BYTES < seg.SAM31_SIZE_BYTES
    # CPU / Mac hosts fetch bf16.
    assert seg.download_plan(variant, "cpu", memory_tight = True)["spec"] == "sam3.1"
    assert seg.download_plan(variant, "mps", memory_tight = True)["spec"] == "sam3.1"
    # An explicit int8 fetches the int8 file whatever the memory.
    monkeypatch.setenv(sq.ENV_PRECISION, "int8")
    assert seg.download_plan(variant, "cuda:0")["spec"] == seg.prequant_spec_id(variant, artifact)
    monkeypatch.setenv(sq.ENV_PRECISION, "bf16")
    assert seg.download_plan(variant, "cuda:0", memory_tight = True)["spec"] == "sam3.1"


def test_start_download_uses_the_plan(monkeypatch):
    _cuda(monkeypatch, (9, 0))
    variant = _published(monkeypatch)
    monkeypatch.setattr(seg, "SEGMENTER_SPECS", seg._build_specs())
    monkeypatch.setattr(seg, "_host_device", lambda: "cuda:0")
    started = []

    class _State:
        def start(self, model_id, token):
            started.append(model_id)
            return "id"

    monkeypatch.setattr(seg, "_get_download_state", lambda: _State())
    seg.start_download(None, "sam3.1")
    assert started == ["sam3.1"]
    monkeypatch.setattr(seg, "auto_memory_tight", lambda device: True)
    seg.start_download(None, "sam3.1")
    assert started[-1] == seg.prequant_spec_id(
        variant, seg.variant_artifact(variant, sq.AUTO_ORDER[0])
    )
    # A spec id passes through untouched.
    seg.start_download(None, "sam3.1@fp8")
    assert started[-1] == "sam3.1@fp8"


class _Meta:
    def __init__(self, commit):
        self.commit_hash, self.size, self.etag = commit, 10, "e"


def _download_state(monkeypatch, missing_exc):
    """A download state whose Hub has every file except the pre-quantized ones (they raise ``missing_exc``)."""
    from core.inference import stt_mtmd_sidecar as mtmd

    seen, commit = [], "c" * 40

    def _meta(url, token = None):
        seen.append(url.rsplit("/", 1)[-1])
        if "INT8" in url or "FP8" in url:
            raise missing_exc
        asked = url.split("/resolve/", 1)[1].split("/", 1)[0]
        return _Meta(asked if re.fullmatch(r"[0-9a-f]{40}", asked) else commit)

    def _stop(repo):
        seen.append(("claim", repo))
        raise RuntimeError("stop before the transfer")

    monkeypatch.setattr("huggingface_hub.get_hf_file_metadata", _meta)
    monkeypatch.setattr(mtmd, "_claim_stt_repository", _stop)
    specs = seg._build_specs()
    state = mtmd._MtmdDownloadState(
        specs = specs,
        cached_paths = lambda *a, **k: None,
        resolve = seg._resolve_segmenter_id,
        label = "Mask segmenter",
        describe_error = lambda model_id, error: None,
    )
    return state, seen


def test_an_unpublished_file_downloads_bf16_instead(monkeypatch, tmp_path):
    from huggingface_hub.errors import EntryNotFoundError

    _published(monkeypatch)
    state, seen = _download_state(
        monkeypatch, EntryNotFoundError("404 Client Error: Entry Not Found")
    )
    state._model_id = "sam3.1@int8-convrot"
    state._run("sam3.1@int8-convrot", None, hub_cache = tmp_path)
    # The quantized file was asked for first, then every bf16 file, and the bf16 run went on to claim the repo.
    assert seen[0] == "config.json" and "sam3.1-INT8-ConvRot.safetensors" in seen
    assert seg.SEGMENTER_WEIGHTS_FILE in seen and ("claim", seg.SAM31_REPO) in seen
    assert state._model_id == "sam3.1"


def test_other_hub_errors_do_not_fall_back(monkeypatch, tmp_path):
    _published(monkeypatch)
    state, seen = _download_state(monkeypatch, ConnectionError("network down"))
    state._model_id = "sam3.1@fp8"
    state._run("sam3.1@fp8", None, hub_cache = tmp_path)
    assert seg.SEGMENTER_WEIGHTS_FILE not in seen and state._model_id == "sam3.1@fp8"
    assert state.status()["error"]


def test_cached_snapshot_finds_a_prequant_spec(tmp_path, monkeypatch):
    from core.inference import stt_sidecar

    monkeypatch.setattr(stt_sidecar, "_read_revision_record", lambda repo: None)
    variant = _published(monkeypatch, "d" * 40)
    monkeypatch.setattr(seg, "SEGMENTER_SPECS", seg._build_specs())
    snapshot = tmp_path / "models--unsloth--sam3.1" / "snapshots" / ("d" * 40)
    snapshot.mkdir(parents = True)
    for name in seg.SEGMENTER_CONFIG_FILES:
        (snapshot / name).write_text("{}")
    spec_id = "sam3.1@fp8"
    assert seg._cached_snapshot(spec_id, hub_cache = tmp_path) is None
    (snapshot / "sam3.1-FP8.safetensors").write_bytes(b"")
    assert seg._cached_snapshot(spec_id, hub_cache = tmp_path) == str(snapshot)
    # The bf16 spec still needs model.safetensors.
    assert seg._cached_snapshot("sam3.1", hub_cache = tmp_path) is None
    # A variant with only a pre-quantized file cached is downloaded, when the precision setting may load that file:
    # fp8 failed the gate, so under auto it is ignored and the auto file is offered for download instead.
    monkeypatch.setattr(
        seg, "_cached_snapshot", lambda model_id, **k: "/snap" if model_id == spec_id else None
    )
    assert seg._variant_snapshot("sam3.1") is None
    monkeypatch.setenv(sq.ENV_PRECISION, "fp8")
    assert seg._variant_snapshot("sam3.1") == "/snap"
    monkeypatch.setenv(sq.ENV_PRECISION, "bf16")
    assert seg._variant_snapshot("sam3.1") is None
    monkeypatch.delenv(sq.ENV_PRECISION)
    convrot_id = "sam3.1@int8-convrot"
    monkeypatch.setattr(
        seg, "_cached_snapshot", lambda model_id, **k: "/snap" if model_id == convrot_id else None
    )
    assert seg._variant_snapshot("sam3.1") == "/snap"
    assert variant.id == "sam3.1"


# ── which weights a load uses ──────────────────────────────────────────────────────────────────────────────────────
def _folder(tmp_path, *names):
    tmp_path.mkdir(parents = True, exist_ok = True)
    for name in ("config.json", "tokenizer.json", *names):
        (tmp_path / name).write_bytes(b"{}")
    return str(tmp_path)


def test_plan_weights(monkeypatch, tmp_path):
    import torch

    _cuda(monkeypatch, (9, 0))
    sam31 = seg.SEGMENTER_VARIANTS["sam3.1"]
    every = [a.filename for a in sam31.prequant]
    full = _folder(tmp_path / "full", "model.safetensors", *every)
    # auto with room: the bf16 file, though the quantized ones are on disk.
    plan = seg.plan_weights(full, "cuda:0", torch.bfloat16, variant = sam31, use_cache = False)
    assert (plan.precision, plan.weights, plan.dequantize) == ("bf16", None, False)
    plan = seg.plan_weights(
        full, "cuda:0", torch.bfloat16, variant = sam31, use_cache = False, memory_tight = True
    )
    first = seg.variant_artifact(sam31, sq.AUTO_ORDER[0])
    assert (plan.precision, plan.weights, plan.dequantize) == (
        sq.AUTO_ORDER[0],
        str(Path(full) / first.filename),
        False,
    )
    # The CPU (OOM retry, a busy card) and Metal keep the bf16 file.
    for device in ("cpu", "mps"):
        plan = seg.plan_weights(full, device, torch.float32, variant = sam31, use_cache = False)
        assert (plan.precision, plan.weights) == ("bf16", None)
    # bf16 asked for.
    monkeypatch.setenv(sq.ENV_PRECISION, "bf16")
    assert seg.plan_weights(full, "cuda:0", variant = sam31, use_cache = False).weights is None
    monkeypatch.setenv(sq.ENV_PRECISION, "fp8")
    assert seg.plan_weights(full, "cuda:0", variant = sam31, use_cache = False).weights.endswith(
        "FP8.safetensors"
    )
    # Not yet uploaded: only model.safetensors on disk -> bf16, no error.
    monkeypatch.setenv(sq.ENV_PRECISION, "int8")
    dense = _folder(tmp_path / "dense", "model.safetensors")
    plan = seg.plan_weights(dense, "cuda:0", variant = sam31, use_cache = False)
    assert (plan.precision, plan.weights) == ("bf16", None) and "not available" in plan.reason
    # Only the plain INT8 file while ConvRot is preferred: never used as a stand-in.
    plain = _folder(tmp_path / "plain", "model.safetensors", "sam3.1-INT8.safetensors")
    assert seg.plan_weights(plain, "cuda:0", variant = sam31, use_cache = False).precision == "bf16"
    monkeypatch.setenv(sq.ENV_INT8_CONVROT, "0")
    assert seg.plan_weights(plain, "cuda:0", variant = sam31, use_cache = False).precision == "int8"


def test_only_a_quantized_file_on_a_cpu_is_dequantized(monkeypatch, tmp_path):
    _cuda(monkeypatch, (9, 0))
    sam31 = seg.SEGMENTER_VARIANTS["sam3.1"]
    only = _folder(tmp_path, "sam3.1-INT8-ConvRot.safetensors")
    plan = seg.plan_weights(only, "cpu", variant = sam31, use_cache = False)
    assert (
        plan.precision == "bf16"
        and plan.dequantize
        and plan.weights.endswith("INT8-ConvRot.safetensors")
    )
    # A card with room runs it dequantized (bf16 speed); a tight one runs it quantized.
    plan = seg.plan_weights(only, "cuda:0", variant = sam31, use_cache = False)
    assert plan.precision == "bf16" and plan.dequantize
    plan = seg.plan_weights(only, "cuda:0", variant = sam31, use_cache = False, memory_tight = True)
    assert plan.precision == "int8" and not plan.dequantize


def test_a_file_the_selection_rejected_is_never_dequantized(monkeypatch, tmp_path):
    """FP8 downloaded earlier, then back on auto (FP8 failed the gate): not run, not dequantized, not downloaded."""
    _cuda(monkeypatch, (9, 0))
    sam31 = seg.SEGMENTER_VARIANTS["sam3.1"]
    only = _folder(tmp_path, "sam3.1-FP8.safetensors")
    for device in ("cuda:0", "cpu"):
        plan = seg.plan_weights(only, device, variant = sam31, use_cache = False)
        assert (plan.precision, plan.weights, plan.dequantize) == ("bf16", None, False)
    # bf16 asked for: an int8 file on disk is not a bf16 model either.
    monkeypatch.setenv(sq.ENV_PRECISION, "bf16")
    convrot = _folder(tmp_path / "convrot", "sam3.1-INT8-ConvRot.safetensors")
    assert seg.plan_weights(convrot, "cpu", variant = sam31, use_cache = False).weights is None
    # Asked for explicitly, the fp8 file is what runs (and on the CPU it is dequantized).
    monkeypatch.setenv(sq.ENV_PRECISION, "fp8")
    plan = seg.plan_weights(only, "cpu", variant = sam31, use_cache = False)
    assert plan.dequantize and plan.weights.endswith("FP8.safetensors")


def test_rocm_never_runs_the_quantized_files(monkeypatch):
    from core import _torchao_stub

    _cuda(monkeypatch, (9, 4))
    assert sq.unsupported_reason("int8", "cuda:0") is None
    monkeypatch.setattr(_torchao_stub, "torch_is_rocm", lambda: True)
    assert "ROCm" in sq.unsupported_reason("int8", "cuda:0")
    assert sq.resolve_precision("cuda:0", available = ("int8", "fp8"))[0] == "bf16"
    sam31 = seg.SEGMENTER_VARIANTS["sam3.1"]
    assert seg.download_plan(sam31, "cuda:0")["precision"] == "bf16"


def test_an_override_folder_holds_the_reported_variants_file(tmp_path, monkeypatch):
    """UNSLOTH_SAM3_VARIANT=sam3 with only a SAM 3.1 file in the override folder: nothing sam3 can load."""
    _folder(tmp_path, "sam3.1-INT8-ConvRot.safetensors")
    monkeypatch.setenv(seg.ENV_PATH_OVERRIDE, str(tmp_path))
    assert seg.model_dir() == str(tmp_path)
    monkeypatch.setenv(seg.ENV_VARIANT, "sam3")
    assert seg.model_dir() is None


def test_no_prequant_reader_means_no_quantized_file_counts(tmp_path, monkeypatch):
    """No torchao (or the Windows ROCm stub): a cached INT8 file cannot even be dequantized, so it is not a
    download and bf16 is offered."""
    from core.inference import prequant_safetensors as pst

    _cuda(monkeypatch, (9, 0))
    sam31 = seg.SEGMENTER_VARIANTS["sam3.1"]
    only = _folder(tmp_path, "sam3.1-INT8-ConvRot.safetensors")
    assert seg.plan_weights(only, "cpu", variant = sam31, use_cache = False).dequantize
    monkeypatch.setattr(pst, "safetensors_prequant_supported", lambda: False)
    plan = seg.plan_weights(only, "cpu", variant = sam31, use_cache = False)
    assert (plan.weights, plan.dequantize) == (None, False)
    monkeypatch.setenv(seg.ENV_PATH_OVERRIDE, str(tmp_path))
    assert seg.model_dir() is None


def test_a_rotated_file_at_another_group_is_refused():
    from core.inference.diffusion_convrot import rotation_metadata
    from core.inference.diffusion_prequant import PREQUANT_FORMAT_ROTATED

    def meta(group):
        return {
            "family": sq.SEGMENTER_QUANT_FAMILY,
            "scheme": "int8",
            "min_features": sq.SEGMENTER_MIN_FEATURES,
            "exclude_name_tokens": list(sq.exclude_tokens("int8", True)),
            **rotation_metadata(group, ["vision_encoder.backbone.layers.9.mlp.fc1"]),
        }

    check = lambda m, **k: sq.checkpoint_problem(
        PREQUANT_FORMAT_ROTATED, m, scheme = "int8", rotated = True, **k
    )  # noqa: E731
    assert check(meta(sq.SEGMENTER_CONVROT_GROUP)) is None
    assert "ConvRot group 16" in check(meta(16))
    assert check(meta(16), convrot_group = 16) is None


def test_an_override_folder_with_only_a_rejected_file_is_not_downloaded(tmp_path, monkeypatch):
    _folder(tmp_path, "sam3.1-FP8.safetensors")
    monkeypatch.setenv(seg.ENV_PATH_OVERRIDE, str(tmp_path))
    assert seg.model_dir() is None and seg.status()["downloaded"] is False
    monkeypatch.setenv(sq.ENV_PRECISION, "fp8")
    assert seg.model_dir() == str(tmp_path)
    monkeypatch.setenv(sq.ENV_PRECISION, "bf16")
    assert seg.model_dir() is None


def test_an_override_folder_with_only_a_quantized_file_is_downloaded(tmp_path, monkeypatch):
    _folder(tmp_path, "sam3.1-INT8-ConvRot.safetensors")
    monkeypatch.setenv(seg.ENV_PATH_OVERRIDE, str(tmp_path))
    assert seg.model_dir() == str(tmp_path)
    (tmp_path / "sam3.1-INT8-ConvRot.safetensors").unlink()
    assert seg.model_dir() is None


def test_status_reports_the_precision_and_its_size(monkeypatch):
    _cuda(monkeypatch, (9, 0))
    monkeypatch.setattr(seg, "transformers_supports_sam3", lambda: True)
    monkeypatch.setattr(seg, "_cached_snapshot", lambda *a, **k: None)
    monkeypatch.setattr(seg, "_host_device", lambda: "cuda:0")
    unpublished = dataclasses.replace(seg.SEGMENTER_VARIANTS["sam3.1"], prequant_revision = None)
    monkeypatch.setitem(seg.SEGMENTER_VARIANTS, "sam3.1", unpublished)
    body = seg.status()
    assert (body["precision"], body["weights_file"], body["size_bytes"]) == (
        "bf16",
        "model.safetensors",
        seg.SAM31_SIZE_BYTES,
    )
    assert body["precision_requested"] == "auto"
    variant = _published(monkeypatch)
    body = seg.status()
    assert (body["precision"], body["weights_file"]) == ("bf16", "model.safetensors")
    monkeypatch.setattr(seg, "auto_memory_tight", lambda device: True)
    body = seg.status()
    artifact = seg.variant_artifact(variant, sq.AUTO_ORDER[0])
    assert (body["precision"], body["weights_file"]) == (sq.AUTO_ORDER[0], artifact.filename)
    assert body["size_bytes"] == artifact.size_bytes + seg.SAM31_CONFIG_BYTES
    monkeypatch.setattr(seg, "_host_device", lambda: "cpu")
    assert seg.status()["precision"] == "bf16"


# ── header checks ──────────────────────────────────────────────────────────────────────────────────────────────────
def _meta(
    scheme = "int8",
    rotated = False,
    **extra,
):
    meta = {
        "family": sq.SEGMENTER_QUANT_FAMILY,
        "scheme": scheme,
        "base_model_id": "unsloth/sam3.1",
        "min_features": sq.SEGMENTER_MIN_FEATURES,
        "exclude_name_tokens": list(sq.exclude_tokens(scheme, rotated)),
    }
    meta.update(extra)
    return meta


def test_checkpoint_problem():
    from core.inference.diffusion_convrot import rotation_metadata
    from core.inference.diffusion_prequant import PREQUANT_FORMAT, PREQUANT_FORMAT_ROTATED

    ok = dict(scheme = "int8", rotated = False, base_model_id = "unsloth/sam3.1")
    assert sq.checkpoint_problem(PREQUANT_FORMAT, _meta(), **ok) is None
    assert "family" in sq.checkpoint_problem(PREQUANT_FORMAT, _meta(family = "flux.1"), **ok)
    assert "scheme" in sq.checkpoint_problem(PREQUANT_FORMAT, _meta("fp8"), **ok)
    assert "built from" in sq.checkpoint_problem(PREQUANT_FORMAT, _meta(base_model_id = "x/y"), **ok)
    assert "min_features" in sq.checkpoint_problem(PREQUANT_FORMAT, _meta(min_features = 256), **ok)
    assert "excludes" in sq.checkpoint_problem(
        PREQUANT_FORMAT, _meta(exclude_name_tokens = ["something_else"]), **ok
    )
    rotated = _meta(
        rotated = True, **rotation_metadata(256, ["vision_encoder.backbone.layers.9.mlp.fc1"])
    )
    rok = dict(ok, rotated = True)
    assert sq.checkpoint_problem(PREQUANT_FORMAT_ROTATED, rotated, **rok) is None
    # A rotation needs the v2 tag both ways, and the file must be the kind the plan asked for.
    assert "disagree" in sq.checkpoint_problem(PREQUANT_FORMAT, rotated, **rok)
    assert "disagree" in sq.checkpoint_problem(PREQUANT_FORMAT_ROTATED, _meta(), **ok)
    assert "unexpected rotation" in sq.checkpoint_problem(PREQUANT_FORMAT_ROTATED, rotated, **ok)
    assert sq.checkpoint_problem(PREQUANT_FORMAT, "x", **ok) == "no metadata"
    # Each file carries its own measured exclusions: a plain INT8 header does not pass as the ConvRot one.
    assert sq.exclude_tokens("int8") != sq.exclude_tokens("int8", rotated = True)
    plain_as_rotated = _meta(**rotation_metadata(256, ["vision_encoder.backbone.layers.9.mlp.fc1"]))
    assert "excludes" in sq.checkpoint_problem(PREQUANT_FORMAT_ROTATED, plain_as_rotated, **rok)


def test_measured_exclusions_and_gate_verdicts():
    """The early ViT blocks stay bf16 (the scan's most sensitive group); auto loads only the files that passed."""
    assert sq.exclude_tokens("int8", rotated = True) == tuple(
        f"backbone.layers.{i}." for i in range(8)
    )
    assert sq.exclude_tokens("int8") == tuple(f"backbone.layers.{i}." for i in range(4))
    assert sq.exclude_tokens("fp8") == tuple(f"backbone.layers.{i}." for i in range(8))
    assert sq.GATE_PASSED == {"fp8": False, "int8": True, "int8-convrot": True}
    assert sq.AUTO_ORDER == ("int8",)
    # A token names one ViT block, never a later one sharing its digits, nor a text-encoder layer.
    from core.inference.diffusion_transformer_quant import make_filter_fn
    import torch

    keep = make_filter_fn(512, exclude_name_tokens = sq.exclude_tokens("fp8"))
    lin = torch.nn.Linear(1024, 1024)
    assert not keep(lin, "vision_encoder.backbone.layers.1.attention.q_proj")
    assert keep(lin, "vision_encoder.backbone.layers.10.attention.q_proj")
    assert keep(lin, "text_encoder.text_model.encoder.layers.1.self_attn.q_proj")


def test_auto_skips_a_file_that_failed_the_gate(monkeypatch, tmp_path):
    _cuda(monkeypatch, (9, 0))
    sam31 = seg.SEGMENTER_VARIANTS["sam3.1"]
    fp8_only = _folder(tmp_path, "model.safetensors", "sam3.1-FP8.safetensors")
    monkeypatch.setattr(sq, "AUTO_ORDER", ("fp8",))
    plan = seg.plan_weights(fp8_only, "cuda:0", variant = sam31, use_cache = False, memory_tight = True)
    assert plan.precision == "bf16"
    monkeypatch.setenv(sq.ENV_PRECISION, "fp8")
    assert seg.plan_weights(fp8_only, "cuda:0", variant = sam31, use_cache = False).precision == "fp8"


def _segment_test_helpers():
    """test_diffusion_segment's tiny config and fake tokenizer, loaded by path (it is not an importable module)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_segment_tests", Path(__file__).with_name("test_diffusion_segment.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._FakeTokenizer, module._tiny_config


# ── a real round trip on a tiny Sam3Model ─────────────────────────────────────────────────────────────────────────
def _tiny_dir(
    tmp_path,
    monkeypatch,
    scheme,
    *,
    convrot = 0,
    keep_dense = True,
):
    """A folder with a tiny random Sam3Model's config, its pre-quantized ``scheme`` file (built with the real
    quantize + write path at min_features 32) and, with ``keep_dense``, its bf16 model.safetensors."""
    transformers = pytest.importorskip("transformers")
    if not hasattr(transformers, "Sam3Model"):
        pytest.skip("transformers without SAM 3")
    pytest.importorskip("torchao")
    import torch

    from core.inference.prequant_safetensors import (
        safetensors_prequant_supported,
        scheme_is_flattenable,
    )
    from core.inference.diffusion_transformer_quant import _make_quant_config

    if (
        not safetensors_prequant_supported()
        or scheme_is_flattenable(_make_quant_config(scheme)) is False
    ):
        pytest.skip(f"this torchao cannot write {scheme} safetensors")
    _FakeTokenizer, _tiny_config = _segment_test_helpers()

    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: _FakeTokenizer()
    )
    monkeypatch.setattr(sq, "SEGMENTER_MIN_FEATURES", 32)
    if scheme == "fp8" and not torch.cuda.is_available():
        # Building is plain tensor math; torchao only asserts a card that could RUN fp8, which the CPU tests do not.
        import torchao.quantization.quant_api as quant_api
        monkeypatch.setattr(
            quant_api, "_check_hardware_support", lambda *a, **k: None, raising = False
        )
    torch.manual_seed(0)
    model = transformers.Sam3Model(_tiny_config()).to(torch.bfloat16)
    if keep_dense:
        model.save_pretrained(tmp_path)
    else:
        model.config.save_pretrained(tmp_path)
    quant = sq.quantize_segmenter_(model, scheme, convrot_group = convrot, min_features = 32)
    assert quant["quantized_linears"] > 0
    artifact = seg.variant_artifact(seg.SEGMENTER_VARIANTS["sam3.1"], scheme)
    name = sq.artifact_filename("sam3.1", scheme, rotated = bool(convrot))
    written = sq.write_prequant_file(
        str(tmp_path / name), model, quant, base_model_id = seg.SAM31_REPO
    )
    assert (
        written["family"] == "sam3"
        and written["fingerprint"]["count"] == quant["quantized_linears"]
    )
    header = json.loads(json.dumps(written, default = str))
    assert header["scheme"] == scheme and (artifact is None or artifact.precision == scheme)
    return str(tmp_path), quant


def test_tiny_fp8_file_dequantizes_and_segments_on_the_cpu(tmp_path, monkeypatch):
    """No bf16 file on disk and no CUDA: the fp8 file is dequantized and Mask from text still runs."""
    from PIL import Image

    path, quant = _tiny_dir(tmp_path, monkeypatch, "fp8", keep_dense = False)
    monkeypatch.setenv(sq.ENV_PRECISION, "fp8")
    segmenter = seg.MaskSegmenter(keep_alive_seconds = 0)
    result = segmenter.segment(
        Image.new("RGB", (64, 40), (120, 80, 40)),
        "sofa, pillow:2",
        threshold = 0.0,
        allow_gpu = False,
        path = path,
    )
    assert result.device == "cpu" and result.precision == "bf16"
    assert [c["found"] for c in result.concepts] == [1, 2]
    assert result.to_json()["precision"] == "bf16"


def test_tiny_dequantized_weights_match_the_quantized_values(tmp_path, monkeypatch):
    import torch

    path, quant = _tiny_dir(tmp_path, monkeypatch, "fp8", keep_dense = False)
    model, info = sq.load_prequantized_segmenter(
        path,
        str(Path(path) / "sam3.1-FP8.safetensors"),
        scheme = "fp8",
        rotated = False,
        device = "cpu",
        dtype = torch.float32,
        base_model_id = seg.SAM31_REPO,
    )
    assert (
        info["kernels"] == "dequantized"
        and info["precision"] == "bf16"
        and info["quantized_linears"] == 0
    )
    from core.inference.prequant_safetensors import load_prequant_safetensors

    stored = load_prequant_safetensors(str(Path(path) / "sam3.1-FP8.safetensors"))["state_dict"]
    name = next(k for k, v in stored.items() if sq._is_quantized_weight(v))
    assert torch.equal(model.state_dict()[name], stored[name].dequantize().float())
    # Refused for another repo's base.
    with pytest.raises(ValueError, match = "built from"):
        sq.load_prequantized_segmenter(
            path,
            str(Path(path) / "sam3.1-FP8.safetensors"),
            scheme = "fp8",
            rotated = False,
            device = "cpu",
            dtype = torch.float32,
            base_model_id = "facebook/sam3",
        )


def test_skeleton_buffers_match_from_pretrained(tmp_path):
    """The quantized load builds the model around the file; its buffers must be a float32 load's. At SAM 3's real
    1008 px input the ViT's global-attention RoPE tables are exact only when built in float32 (a bf16 default moved
    them by up to 0.04 and every mask with them), so the skeleton is built in float32 whatever the caller's default."""
    transformers = pytest.importorskip("transformers")
    if not hasattr(transformers, "Sam3Model"):
        pytest.skip("transformers without SAM 3")
    import torch

    _FakeTokenizer, _tiny_config = _segment_test_helpers()
    config = _tiny_config()
    config.vision_config.backbone_config.image_size = 1008
    transformers.Sam3Model(config).to(torch.bfloat16).save_pretrained(tmp_path)
    # The float32 load is the reference: some 5.x releases (5.18 here) cast the global-attention position_ids to
    # bf16 under from_pretrained(dtype = bfloat16), moving the tables; others keep them in float32.
    dense = dict(
        transformers.Sam3Model.from_pretrained(tmp_path, dtype = torch.float32).named_buffers()
    )
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        built = dict(sq._build_skeleton(str(tmp_path)).named_buffers())
    finally:
        torch.set_default_dtype(previous)
    # The RoPE tables: `rotary_emb.inv_freq` and `position_ids` in transformers 5.x, "rope" in other spellings.
    rope = [
        name for name in dense if any(key in name for key in ("rope", "rotary", "position_ids"))
    ]
    assert dense.keys() == built.keys() and rope, sorted(dense)
    for name, value in dense.items():
        assert built[name].dtype == value.dtype and torch.equal(built[name], value), name


def test_a_broken_file_falls_back_to_bf16(tmp_path, monkeypatch):
    """A pre-quantized file that does not load leaves the bf16 one in charge, never a crash."""
    import torch
    from PIL import Image

    path, _quant = _tiny_dir(tmp_path, monkeypatch, "fp8")
    monkeypatch.setenv(sq.ENV_PRECISION, "fp8")
    _cuda(monkeypatch, (9, 0))
    plan = seg.plan_weights(path, "cuda:0", torch.bfloat16, use_cache = False)
    assert plan.precision == "fp8"
    (Path(path) / "sam3.1-FP8.safetensors").write_bytes(b"not a safetensors file")
    segmenter = seg.MaskSegmenter(keep_alive_seconds = 0)
    # Force the planned device's choice onto the CPU loader path: the quantized load runs (and fails) first.
    monkeypatch.setattr(seg, "choose_device", lambda allow_gpu: ("cpu", torch.float32))
    monkeypatch.setattr(
        seg,
        "plan_weights",
        lambda *a, **k: plan,
    )
    result = segmenter.segment(
        Image.new("RGB", (32, 32)), "sofa", threshold = 0.0, allow_gpu = True, path = path
    )
    assert result.precision == "bf16" and segmenter.last_prequant_failure


def test_a_broken_file_with_no_bf16_beside_it_stops_counting(tmp_path, monkeypatch):
    """Only a pre-quantized file on disk and it does not load, even dequantized: the request fails with a clear
    error once, then the file no longer counts as downloaded and the bf16 download is offered."""
    import torch
    from PIL import Image

    path, _quant = _tiny_dir(tmp_path, monkeypatch, "fp8", keep_dense = False)
    monkeypatch.setenv(sq.ENV_PRECISION, "fp8")
    _cuda(monkeypatch, (9, 0))
    sam31 = _published(monkeypatch, "d" * 40)
    artifact = seg.variant_artifact(sam31, "fp8")
    (Path(path) / artifact.filename).write_bytes(b"not a safetensors file")
    monkeypatch.setattr(seg, "choose_device", lambda allow_gpu: ("cpu", torch.float32))
    segmenter = seg.MaskSegmenter(keep_alive_seconds = 0)
    with pytest.raises(Exception, match = "BF16 weights"):
        segmenter.segment(Image.new("RGB", (32, 32)), "sofa", threshold = 0.0, path = path)
    plan = seg.plan_weights(path, "cpu", variant = sam31, use_cache = False)
    assert (plan.weights, plan.dequantize) == (None, False)
    monkeypatch.setenv(seg.ENV_PATH_OVERRIDE, path)
    assert seg.model_dir() is None
    monkeypatch.delenv(seg.ENV_PATH_OVERRIDE)
    # The same file as the cached copy: the download route offers bf16, not that file again.
    monkeypatch.setattr(
        seg, "_cached_snapshot", lambda model_id, **k: path if "@" in model_id else None
    )
    assert seg._variant_snapshot("sam3.1") is None
    assert seg.download_plan(sam31, "cuda:0")["precision"] == "bf16"


@pytest.mark.skipif(
    not __import__("torch").cuda.is_available(), reason = "quantized kernels need a CUDA card"
)
@pytest.mark.parametrize("scheme, convrot", [("fp8", 0), ("int8", 0), ("int8", 16)])
def test_tiny_quantized_file_runs_quantized_on_cuda(tmp_path, monkeypatch, scheme, convrot):
    import torch

    capability = torch.cuda.get_device_capability()
    if capability < sq._MIN_CAPABILITY[scheme]:
        pytest.skip(f"{scheme} needs a newer card")
    path, quant = _tiny_dir(tmp_path, monkeypatch, scheme, convrot = convrot)
    model, info = sq.load_prequantized_segmenter(
        path,
        str(Path(path) / sq.artifact_filename("sam3.1", scheme, rotated = bool(convrot))),
        scheme = scheme,
        rotated = bool(convrot),
        device = "cuda:0",
        dtype = torch.bfloat16,
        base_model_id = seg.SAM31_REPO,
        convrot_group = convrot or sq.SEGMENTER_CONVROT_GROUP,
    )
    assert (
        info["kernels"] == "quantized" and info["quantized_linears"] == quant["quantized_linears"]
    )
    assert info["rotated_linears"] == len(quant.get("activation_rotation_fqns") or ())
    pixels = torch.zeros(1, 3, 112, 112, device = "cuda:0", dtype = torch.bfloat16)
    with torch.inference_mode():
        vision = model.get_vision_features(pixel_values = pixels)
    assert vision is not None
