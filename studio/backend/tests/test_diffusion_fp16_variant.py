# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""An fp16 pipeline load reads the repo's ``.fp16`` twins (SDXL: 6.5 GB instead of the 12.9 GB fp32 defaults), and the
download plan, prefetch, size estimates and completeness checks agree with what it reads."""

from __future__ import annotations

import json
import types
from pathlib import Path

import pytest

from core.inference import diffusion as diffusion_mod
from core.inference import diffusion_fast_load as fl
from core.inference import diffusion_fp16_variant as fv
from core.inference.diffusion import DiffusionBackend
from core.inference.diffusion_auto_policy import (
    companion_bytes_from_declared,
    resident_bytes_from_declared,
)
from core.inference.diffusion_component_sizes import select_weight_files
from core.inference.diffusion_families import local_pipeline_components_are_complete

MIB = 1024 * 1024
SDXL = "stabilityai/stable-diffusion-xl-base-1.0"
SDXL_INDEX = {
    "_class_name": "StableDiffusionXLPipeline",
    "_diffusers_version": "0.19.0.dev0",
    "force_zeros_for_empty_prompt": True,
    "scheduler": ["diffusers", "EulerDiscreteScheduler"],
    "text_encoder": ["transformers", "CLIPTextModel"],
    "text_encoder_2": ["transformers", "CLIPTextModelWithProjection"],
    "tokenizer": ["transformers", "CLIPTokenizer"],
    "tokenizer_2": ["transformers", "CLIPTokenizer"],
    "unet": ["diffusers", "UNet2DConditionModel"],
    "vae": ["diffusers", "AutoencoderKL"],
}
# stabilityai/stable-diffusion-xl-base-1.0 @ 462165984030d822, sizes in bytes
SDXL_DEFAULTS = {
    "unet/diffusion_pytorch_model.safetensors": 10270077736,
    "vae/diffusion_pytorch_model.safetensors": 334643268,
    "text_encoder/model.safetensors": 492265168,
    "text_encoder_2/model.safetensors": 2778702264,
}
SDXL_TWINS = {
    "unet/diffusion_pytorch_model.fp16.safetensors": 5135149760,
    "vae/diffusion_pytorch_model.fp16.safetensors": 167335342,
    "text_encoder/model.fp16.safetensors": 246144152,
    "text_encoder_2/model.fp16.safetensors": 1389382176,
}
SDXL_CONFIGS = {
    "model_index.json": 609,
    "scheduler/scheduler_config.json": 479,
    "text_encoder/config.json": 565,
    "text_encoder_2/config.json": 575,
    "tokenizer/vocab.json": 1059962,
    "tokenizer/merges.txt": 524619,
    "tokenizer/tokenizer_config.json": 737,
    "tokenizer_2/vocab.json": 1059962,
    "tokenizer_2/merges.txt": 524619,
    "tokenizer_2/tokenizer_config.json": 725,
    "unet/config.json": 1680,
    "vae/config.json": 642,
    "unet/diffusion_pytorch_model.fp16.bin": 5135149760,
    "unet/openvino_model.bin": 5135149760,
    "text_encoder_2/flax_model.msgpack": 2778702264,
}


@pytest.fixture(autouse = True)
def _variant_on(monkeypatch):
    monkeypatch.delenv(fv.FP16_VARIANT_ENV, raising = False)


def _snapshot(
    tmp_path: Path,
    files,
    index = SDXL_INDEX,
) -> Path:
    snap = tmp_path / "snap"
    snap.mkdir(parents = True, exist_ok = True)
    (snap / "model_index.json").write_text(json.dumps(index))
    for rel in files:
        if rel == "model_index.json":
            continue
        path = snap / rel
        path.parent.mkdir(parents = True, exist_ok = True)
        if rel.endswith("config.json"):
            path.write_text(json.dumps({"_class_name": "X"}))
        else:
            path.write_bytes(b"w")
    return snap


# Backend CI has neither diffusers nor transformers, and an unresolvable class keeps the defaults (conservative):
# resolve the classes these tests name from a table, and check the real lookup separately where the libraries exist.
_PINS_FP32 = {
    "UNet2DConditionModel": False,
    "AutoencoderKL": False,
    "AutoencoderKLFlux2": False,
    "CLIPTextModel": False,
    "CLIPTextModelWithProjection": False,
    "UNet2DModel": False,
    "T5EncoderModel": True,
}


@pytest.fixture(autouse = True)
def _class_table(monkeypatch):
    monkeypatch.setattr(
        fv,
        "_class_pins_fp32",
        lambda library, class_name: library not in fv._WEIGHT_LIBRARIES
        or _PINS_FP32.get(class_name, True),
    )


def test_the_real_class_lookup_reads_keep_in_fp32_modules(monkeypatch):
    pytest.importorskip("diffusers")
    pytest.importorskip("transformers")
    monkeypatch.undo()
    assert fv._class_pins_fp32("diffusers", "UNet2DConditionModel") is False
    assert fv._class_pins_fp32("transformers", "CLIPTextModel") is False
    assert fv._class_pins_fp32("transformers", "T5EncoderModel") is True
    assert fv._class_pins_fp32("diffusers", "NoSuchModel") is True
    assert fv._class_pins_fp32("not_a_library", "UNet2DConditionModel") is True


def test_every_sdxl_component_maps_to_its_twin():
    names = [*SDXL_CONFIGS, *SDXL_DEFAULTS, *SDXL_TWINS]
    assert fv.plan_fp16_twins(names, SDXL_INDEX) == {
        "text_encoder": "text_encoder/model.fp16.safetensors",
        "text_encoder_2": "text_encoder_2/model.fp16.safetensors",
        "unet": "unet/diffusion_pytorch_model.fp16.safetensors",
        "vae": "vae/diffusion_pytorch_model.fp16.safetensors",
    }


@pytest.mark.parametrize("missing", sorted(SDXL_TWINS))
def test_one_missing_twin_keeps_the_defaults(missing):
    names = [*SDXL_CONFIGS, *SDXL_DEFAULTS, *(t for t in SDXL_TWINS if t != missing)]
    assert fv.plan_fp16_twins(names, SDXL_INDEX) is None


def test_a_sharded_twin_or_a_bin_twin_is_not_a_twin():
    names = [n for n in (*SDXL_CONFIGS, *SDXL_DEFAULTS, *SDXL_TWINS) if not n.startswith("unet/")]
    sharded = names + [
        "unet/diffusion_pytorch_model.fp16-00001-of-00002.safetensors",
        "unet/diffusion_pytorch_model.fp16-00002-of-00002.safetensors",
        "unet/diffusion_pytorch_model.safetensors.index.fp16.json",
    ]
    assert fv.plan_fp16_twins(sharded, SDXL_INDEX) is None
    assert fv.plan_fp16_twins(names + ["unet/diffusion_pytorch_model.fp16.bin"], SDXL_INDEX) is None


def test_a_component_not_read_from_the_repo_needs_no_twin():
    names = [
        n
        for n in (*SDXL_CONFIGS, *SDXL_DEFAULTS, *SDXL_TWINS)
        if n != "text_encoder_2/model.fp16.safetensors"
    ]
    assert fv.plan_fp16_twins(names, SDXL_INDEX) is None
    plan = fv.plan_fp16_twins(names, SDXL_INDEX, components = ("unet", "vae", "text_encoder"))
    assert set(plan) == {"unet", "vae", "text_encoder"}


def test_a_class_that_pins_fp32_modules_keeps_the_defaults():
    # T5EncoderModel keeps ``wo`` in fp32 at an fp16 load: a twin would upcast fp16-rounded weights.
    index = {**SDXL_INDEX, "text_encoder_2": ["transformers", "T5EncoderModel"]}
    assert fv.plan_fp16_twins([*SDXL_DEFAULTS, *SDXL_TWINS], index) is None
    unknown = {**SDXL_INDEX, "unet": ["diffusers", "NoSuchModel"]}
    assert fv.plan_fp16_twins([*SDXL_DEFAULTS, *SDXL_TWINS], unknown) is None


def test_twin_names():
    assert fv.twin_of("unet/diffusion_pytorch_model.safetensors") == (
        "unet/diffusion_pytorch_model.fp16.safetensors"
    )
    assert fv.twin_of("text_encoder/model.safetensors") == "text_encoder/model.fp16.safetensors"
    assert fv.twin_of("text_encoder/model-00001-of-00002.safetensors") is None
    assert fv.twin_of("unet/diffusion_pytorch_model.fp16.safetensors") is None
    assert fv.is_fp16_twin("vae/diffusion_pytorch_model.fp16.safetensors")
    assert not fv.is_fp16_twin("vae/diffusion_pytorch_model.safetensors")
    assert not fv.is_fp16_twin("vae/diffusion_pytorch_model.fp16.bin")


def test_an_fp16_load_reads_the_twins_and_a_bf16_load_the_defaults(tmp_path):
    snap = _snapshot(tmp_path, [*SDXL_CONFIGS, *SDXL_DEFAULTS, *SDXL_TWINS])
    assert fv.local_fp16_variant(str(snap), load_dtype = "torch.float16") == "fp16"
    import torch

    assert (
        fv.local_fp16_variant(str(snap), load_dtype = torch.float16, torch_dtype = torch.float16)
        == "fp16"
    )
    # fp16 -> bf16 rounds differently from fp32 -> bf16: a bf16 / fp32 load keeps the defaults.
    assert fv.local_fp16_variant(str(snap), load_dtype = torch.bfloat16) is None
    assert fv.local_fp16_variant(str(snap), load_dtype = torch.float32) is None
    mapped = {"default": torch.float16, "unet": torch.float32}
    assert fv.local_fp16_variant(str(snap), load_dtype = torch.float16, torch_dtype = mapped) is None


def test_defaults_only_snapshot_never_asks_for_the_variant(tmp_path):
    snap = _snapshot(tmp_path, [*SDXL_CONFIGS, *SDXL_DEFAULTS])
    assert fv.local_fp16_variant(str(snap), load_dtype = "torch.float16") is None


def test_twins_only_snapshot_loads_the_twins_at_any_dtype(tmp_path):
    # Twins-only snapshot: without the variant nothing loads.
    import torch
    snap = _snapshot(tmp_path, [*SDXL_CONFIGS, *SDXL_TWINS])
    assert fv.local_fp16_variant(str(snap), load_dtype = torch.bfloat16) == "fp16"


@pytest.mark.parametrize(
    "unet_defaults",
    [
        (
            "unet/diffusion_pytorch_model-00001-of-00002.safetensors",
            "unet/diffusion_pytorch_model-00002-of-00002.safetensors",
        ),
        ("unet/diffusion_pytorch_model.safetensors.index.json",),
        ("unet/diffusion_pytorch_model.bin",),
    ],
)
def test_sharded_or_bin_defaults_beside_twins_keep_a_bf16_load_on_the_defaults(
    tmp_path, unet_defaults
):
    import torch

    defaults = [d for d in SDXL_DEFAULTS if not d.startswith("unet/")]
    snap = _snapshot(tmp_path, [*SDXL_CONFIGS, *defaults, *unet_defaults, *SDXL_TWINS])
    assert fv.local_fp16_variant(str(snap), load_dtype = torch.bfloat16) is None
    assert fv.local_fp16_variant(str(snap), load_dtype = torch.float16) == "fp16"


def test_a_supplied_component_needs_no_twin_on_disk(tmp_path):
    files = [
        *SDXL_CONFIGS,
        *SDXL_DEFAULTS,
        *(t for t in SDXL_TWINS if not t.startswith("text_encoder_2/")),
    ]
    snap = _snapshot(tmp_path, files)
    assert fv.local_fp16_variant(str(snap), load_dtype = "torch.float16") is None
    assert (
        fv.local_fp16_variant(
            str(snap), load_dtype = "torch.float16", supplied = ("text_encoder_2", "torch_dtype")
        )
        == "fp16"
    )


def test_kill_switch(tmp_path, monkeypatch):
    snap = _snapshot(tmp_path, [*SDXL_CONFIGS, *SDXL_DEFAULTS, *SDXL_TWINS])
    monkeypatch.setenv(fv.FP16_VARIANT_ENV, "0")
    assert fv.local_fp16_variant(str(snap), load_dtype = "torch.float16") is None


def _stub_hub(
    monkeypatch,
    tmp_path,
    cached = (),
):
    manifest = tmp_path / "model_index.json"
    manifest.write_text(json.dumps(SDXL_INDEX))
    files = {**SDXL_CONFIGS, **SDXL_DEFAULTS, **SDXL_TWINS}

    class _Api:
        def model_info(self, repo_id, **_kwargs):
            siblings = [types.SimpleNamespace(rfilename = n, size = s) for n, s in files.items()]
            return types.SimpleNamespace(siblings = siblings, sha = "4" * 40)

    monkeypatch.setattr("huggingface_hub.HfApi", lambda *a, **k: _Api())
    monkeypatch.setattr("huggingface_hub.hf_hub_download", lambda *a, **k: str(manifest))
    monkeypatch.setattr(
        "huggingface_hub.try_to_load_from_cache",
        lambda repo_id, filename, **k: f"/cache/{filename}" if filename in cached else None,
    )
    monkeypatch.setattr(diffusion_mod, "prefer_ungated_mirror", lambda repo_id, *a, **k: repo_id)


def _plan(fp16_twins):
    resident: list = []
    total, files = DiffusionBackend._estimate_download_bytes(
        SDXL,
        None,
        SDXL,
        None,
        kind = "pipeline",
        resident_file_sizes_out = resident,
        fp16_twins = fp16_twins,
    )
    return total, set(files), dict(resident)


def test_an_fp16_plan_stages_the_twins_instead_of_the_defaults(monkeypatch, tmp_path):
    _stub_hub(monkeypatch, tmp_path)
    total, files, resident = _plan(True)
    assert set(SDXL_TWINS) <= files and not (set(SDXL_DEFAULTS) & files)
    assert resident == SDXL_TWINS
    configs = sum(s for n, s in SDXL_CONFIGS.items() if n.endswith((".json", ".txt")))
    assert total == sum(SDXL_TWINS.values()) + configs
    # Twins price 1:1.
    twins_resident = resident_bytes_from_declared(SDXL, resident.items())
    assert twins_resident == sum(SDXL_TWINS.values())
    assert (
        abs(twins_resident / resident_bytes_from_declared(SDXL, SDXL_DEFAULTS.items()) - 1) < 0.001
    )


def test_a_bf16_plan_is_unchanged(monkeypatch, tmp_path):
    _stub_hub(monkeypatch, tmp_path)
    total, files, resident = _plan(False)
    assert set(SDXL_DEFAULTS) <= files and not (set(SDXL_TWINS) & files)
    assert resident == SDXL_DEFAULTS


def test_cached_defaults_are_not_downloaded_again_as_twins(monkeypatch, tmp_path):
    _stub_hub(monkeypatch, tmp_path, cached = SDXL_DEFAULTS)
    _total, files, _resident = _plan(True)
    assert set(SDXL_DEFAULTS) <= files and not (set(SDXL_TWINS) & files)
    _stub_hub(monkeypatch, tmp_path, cached = (*SDXL_DEFAULTS, *SDXL_TWINS))
    assert set(SDXL_TWINS) <= _plan(True)[1]


def test_kill_switch_keeps_the_default_plan(monkeypatch, tmp_path):
    _stub_hub(monkeypatch, tmp_path)
    monkeypatch.setenv(fv.FP16_VARIANT_ENV, "0")
    assert set(SDXL_DEFAULTS) <= _plan(True)[1]


def test_resident_estimate_ignores_a_twin_beside_its_default():
    both = {**SDXL_DEFAULTS, **SDXL_TWINS}
    assert resident_bytes_from_declared(SDXL, both.items()) == resident_bytes_from_declared(
        SDXL, SDXL_DEFAULTS.items()
    )
    assert companion_bytes_from_declared(SDXL, both.items()) == companion_bytes_from_declared(
        SDXL, SDXL_DEFAULTS.items()
    )
    twins_te = {k: v for k, v in SDXL_TWINS.items() if not k.startswith("unet/")}
    assert companion_bytes_from_declared(SDXL, twins_te.items()) == sum(twins_te.values())


def test_prefetch_reads_the_twins_the_load_reads(tmp_path):
    snap = _snapshot(tmp_path, [*SDXL_CONFIGS, *SDXL_DEFAULTS, *SDXL_TWINS])
    rel = lambda paths: sorted(p[len(str(snap)) + 1 :] for p in paths)  # noqa: E731
    assert rel(fl.pipeline_component_files(str(snap), skip_denoiser = False)) == sorted(SDXL_DEFAULTS)
    assert rel(
        fl.pipeline_component_files(str(snap), skip_denoiser = False, variant = "fp16")
    ) == sorted(SDXL_TWINS)


def test_header_sizes_fall_back_to_a_lone_twin():
    assert select_weight_files(["model.fp16.safetensors"], folder = "text_encoder") == [
        "model.fp16.safetensors"
    ]
    assert select_weight_files(
        ["diffusion_pytorch_model.safetensors", "diffusion_pytorch_model.fp16.safetensors"],
        folder = "unet",
    ) == ["diffusion_pytorch_model.safetensors"]


def test_a_twins_only_snapshot_is_complete(tmp_path):
    snap = _snapshot(tmp_path, [*SDXL_CONFIGS, *SDXL_TWINS])
    assert local_pipeline_components_are_complete(snap, "model_index.json")
    (snap / "unet" / "diffusion_pytorch_model.fp16.safetensors").unlink()
    assert not local_pipeline_components_are_complete(snap, "model_index.json")


def test_a_mix_of_twins_and_defaults_the_loader_cannot_open_is_incomplete(tmp_path, monkeypatch):
    # unet only as a twin, every other component only as a default: no pipeline-wide variant reads both.
    mixed = [t for t in SDXL_TWINS if t.startswith("unet/")]
    mixed += [d for d in SDXL_DEFAULTS if not d.startswith("unet/")]
    snap = _snapshot(tmp_path, [*SDXL_CONFIGS, *mixed])
    assert fv.local_fp16_variant(str(snap), load_dtype = "torch.float16") is None
    assert not local_pipeline_components_are_complete(snap, "model_index.json")
    monkeypatch.setenv(fv.FP16_VARIANT_ENV, "0")
    full = _snapshot(tmp_path / "full", [*SDXL_CONFIGS, *SDXL_TWINS])
    assert not local_pipeline_components_are_complete(full, "model_index.json")


def test_local_pricing_reads_twins_only_where_no_default_sits(tmp_path, monkeypatch):
    import torch

    monkeypatch.setattr(
        DiffusionBackend, "_safetensors_cast_bytes", staticmethod(lambda path, itemsize, pinned: 7)
    )
    configs = [n for n in SDXL_CONFIGS if n.endswith((".json", ".txt"))]
    snap = _snapshot(tmp_path, [*configs, *SDXL_TWINS])
    priced = DiffusionBackend._local_dir_weight_sizes(
        snap, exclude_transformer = False, load_dtype = torch.float16
    )
    assert set(priced) == set(SDXL_TWINS)
    snap_both = _snapshot(tmp_path / "b", [*configs, *SDXL_DEFAULTS, *SDXL_TWINS])
    priced = DiffusionBackend._local_dir_weight_sizes(
        snap_both, exclude_transformer = False, load_dtype = torch.float16
    )
    assert set(priced) == set(SDXL_DEFAULTS)


def test_a_supplied_component_unblocks_the_twins_its_cached_default_blocked(tmp_path):
    # vae cached only as its default file, every other component staged as a twin: the vae is supplied.
    files = [*SDXL_CONFIGS, "vae/diffusion_pytorch_model.safetensors"]
    files += [t for t in SDXL_TWINS if not t.startswith("vae/")]
    snap = _snapshot(tmp_path, files)
    assert fv.local_fp16_variant(str(snap), load_dtype = "torch.float16") is None
    assert fv.local_fp16_variant(str(snap), load_dtype = "torch.float16", supplied = ("vae",)) == "fp16"


def test_the_load_recomputes_the_variant_with_the_final_kwargs_even_after_a_first_none():
    import inspect

    source = inspect.getsource(diffusion_mod)
    assert "if _fp16_variant_eligible:" in source
    assert (
        "if _fp16_variant is not None:\n                                        # Final kwargs"
        not in source
    )


def test_a_fixed_folder_assembler_never_accepts_a_lone_twin(tmp_path):
    # FLUX.3's assembler opens each folder without a variant: an fp16-only vae cannot load there.
    snap = _snapshot(tmp_path, [*SDXL_CONFIGS, *SDXL_DEFAULTS])
    folders = {
        "vae": ("vae", "diffusers", "AutoencoderKL"),
        "unet": ("unet", "diffusers", "UNet2DConditionModel"),
    }
    assert local_pipeline_components_are_complete(
        snap, "model_index.json", fixed_component_folders = folders
    )
    (snap / "vae" / "diffusion_pytorch_model.safetensors").unlink()
    (snap / "vae" / "diffusion_pytorch_model.fp16.safetensors").write_bytes(b"x" * 8)
    assert not local_pipeline_components_are_complete(
        snap, "model_index.json", fixed_component_folders = folders
    )
