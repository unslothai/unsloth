"""Whole-pipeline single files (SDXL) split into U-Net / text-encoder / VAE from the safetensors header, and a
U-Net repo's unet/ folder never counts as a companion."""

import json
import struct
import types

import pytest

from core.inference import diffusion as d
from core.inference.diffusion import DiffusionBackend
from core.inference.diffusion_memory import (
    OFFLOAD_MODEL,
    DeviceMemory,
    calibrated_image_activation,
    plan_diffusion_memory,
)

MIB = 1024 * 1024


def _fake_safetensors(path, parts):
    """A header-only checkpoint: one F16 tensor of ``mib`` MiB per prefix, the file padded to the declared size."""
    header, offset = {}, 0
    for name, mib in parts.items():
        nbytes = mib * MIB
        header[name] = {
            "dtype": "F16",
            "shape": [nbytes // 2],
            "data_offsets": [offset, offset + nbytes],
        }
        offset += nbytes
    blob = json.dumps(header).encode()
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(blob)))
        fh.write(blob)
        fh.truncate(8 + len(blob) + offset)
    return str(path)


SDXL_LIKE = {
    "model.diffusion_model.input_blocks.0.0.weight": 40,
    "conditioner.embedders.0.transformer.text_model.embeddings.weight": 3,
    "conditioner.embedders.1.model.transformer.resblocks.0.weight": 9,
    "first_stage_model.decoder.conv_in.weight": 4,
}


@pytest.fixture
def no_overrides(monkeypatch):
    import core.inference.diffusion_comfy_components as comps
    monkeypatch.setattr(comps, "active_component_overrides", lambda: None)


def test_header_split_prices_encoders_and_vae_as_shares_of_the_file(tmp_path, no_overrides):
    path = _fake_safetensors(tmp_path / "sdxl.safetensors", SDXL_LIKE)
    assert d._whole_pipeline_companion_mib(path, 56) == (16, 12)
    # A widened load (fp8 -> bf16) scales every part alike.
    assert d._whole_pipeline_companion_mib(path, 112) == (32, 24)


def test_header_split_falls_back_for_other_layouts(tmp_path, no_overrides):
    diffusers_keys = _fake_safetensors(
        tmp_path / "diffusers.safetensors",
        {"down_blocks.0.resnets.0.conv1.weight": 40, "text_model.embeddings.weight": 8},
    )
    assert d._whole_pipeline_companion_mib(diffusers_keys, 48) is None
    unet_only = _fake_safetensors(
        tmp_path / "unet.safetensors", {"model.diffusion_model.out.2.weight": 40}
    )
    assert d._whole_pipeline_companion_mib(unet_only, 40) is None
    garbage = tmp_path / "garbage.safetensors"
    garbage.write_bytes(b"\xff" * 64)
    assert d._whole_pipeline_companion_mib(str(garbage), 1) is None
    assert d._whole_pipeline_companion_mib(str(tmp_path / "missing.safetensors"), 10) is None
    assert d._whole_pipeline_companion_mib(None, 10) is None
    assert d._whole_pipeline_companion_mib(diffusers_keys, None) is None


def test_header_split_defers_to_active_component_overrides(tmp_path, monkeypatch):
    import core.inference.diffusion_comfy_components as comps

    path = _fake_safetensors(tmp_path / "sdxl.safetensors", SDXL_LIKE)
    monkeypatch.setattr(
        comps, "active_component_overrides", lambda: types.SimpleNamespace(paths = {"vae": "v"})
    )
    assert d._whole_pipeline_companion_mib(path, 56) is None


def _single_file_plan(
    monkeypatch,
    path,
    fam,
    companion_cache_bytes = 0,
):
    seen = {}

    def plan(**kwargs):
        seen.update(kwargs)
        return plan_diffusion_memory(**kwargs)

    monkeypatch.setattr(d, "plan_diffusion_memory", plan)
    monkeypatch.setattr(
        DiffusionBackend,
        "_companion_cache_bytes",
        staticmethod(lambda *a, **k: companion_cache_bytes),
    )
    monkeypatch.setattr(
        DiffusionBackend, "_text_encoder_cache_bytes", staticmethod(lambda *a, **k: 0)
    )
    monkeypatch.setattr(DiffusionBackend, "_supplied_component_mib", lambda self, *a, **k: None)
    monkeypatch.setattr(
        d,
        "settled_snapshot_device_memory",
        lambda target, **k: DeviceMemory("cuda", "cuda", "discrete_vram", 6_959, 8_188),
    )
    monkeypatch.setattr(d, "_calibrated_activation", lambda *a, **k: None)
    target = types.SimpleNamespace(
        device = "cuda", backend = "cuda", dtype = None, supports_model_cpu_offload = True
    )
    final = DiffusionBackend()._plan_memory(
        target,
        path,
        "stabilityai/stable-diffusion-xl-base-1.0",
        fam,
        None,
        False,
        kind = "single_file",
    )
    return final, seen


def test_plan_memory_splits_a_whole_pipeline_file_and_skips_the_base_cache(
    tmp_path, monkeypatch, no_overrides
):
    from core.inference.diffusion_families import detect_family

    fam = detect_family("stabilityai/stable-diffusion-xl-base-1.0")
    assert fam.single_file_is_pipeline and fam.denoiser_attr == "unet"
    path = _fake_safetensors(tmp_path / "sdxl.safetensors", SDXL_LIKE)
    # A cached base repo holding weights must not be added on top: the file already carries them.
    plan, seen = _single_file_plan(monkeypatch, path, fam, companion_cache_bytes = 500 * MIB)
    assert seen["model_dense_mib"] == 56
    assert seen["companion_dense_mib"] == 16 and seen["text_encoder_dense_mib"] == 12
    assert plan.estimates[d.WHOLE_PIPELINE_SPLIT_KEY] == 1
    # The refusal sizing no longer subtracts a "double-counted" companion from the file.
    sized = DiffusionBackend()._resident_sized_plan(
        plan, fam, fam.base_repo, types.SimpleNamespace(dtype = None), "single_file"
    )
    assert sized.estimates["model_dense_mib"] == plan.estimates["model_dense_mib"]


def test_plan_memory_keeps_todays_sizing_for_a_file_without_the_ldm_prefixes(
    tmp_path, monkeypatch, no_overrides
):
    from core.inference.diffusion_families import detect_family

    fam = detect_family("stabilityai/stable-diffusion-xl-base-1.0")
    path = _fake_safetensors(tmp_path / "odd.safetensors", {"unet.conv_in.weight": 56})
    plan, seen = _single_file_plan(monkeypatch, path, fam, companion_cache_bytes = 20 * MIB)
    assert seen["model_dense_mib"] == 56 + 20 and seen["companion_dense_mib"] == 20
    assert d.WHOLE_PIPELINE_SPLIT_KEY not in plan.estimates


def test_plan_memory_does_not_split_a_denoiser_only_file(tmp_path, monkeypatch, no_overrides):
    """A DiT single file keeps the base-cache companions (it carries no encoders of its own)."""
    fam = types.SimpleNamespace(
        name = "z-image",
        single_file_is_pipeline = False,
        pipeline_only = False,
        denoiser_attr = "transformer",
    )
    monkeypatch.setattr(
        DiffusionBackend, "_comfy_single_file_resident_mib", staticmethod(lambda *a, **k: None)
    )
    path = _fake_safetensors(tmp_path / "dit.safetensors", SDXL_LIKE)
    _plan, seen = _single_file_plan(monkeypatch, path, fam, companion_cache_bytes = 20 * MIB)
    assert seen["model_dense_mib"] == 56 + 20 and seen["companion_dense_mib"] == 20


def test_companion_scan_drops_a_unet_folder_only_when_asked(tmp_path):
    for rel, mib in (
        ("unet/diffusion_pytorch_model.safetensors", 40),
        ("transformer/diffusion_pytorch_model.safetensors", 7),
        ("text_encoder/model.safetensors", 3),
        ("text_encoder_2/model.safetensors", 9),
        ("vae/diffusion_pytorch_model.safetensors", 4),
    ):
        f = tmp_path / rel
        f.parent.mkdir(parents = True, exist_ok = True)
        with open(f, "wb") as fh:
            fh.truncate(mib * MIB)
    default = DiffusionBackend._companion_cache_bytes(str(tmp_path))
    assert default == (40 + 3 + 9 + 4) * MIB, "the default call shape changed"
    unet = DiffusionBackend._companion_cache_bytes(
        str(tmp_path), denoiser_dirs = ("transformer", "unet")
    )
    assert unet == (3 + 9 + 4) * MIB
    assert DiffusionBackend._text_encoder_cache_bytes(str(tmp_path)) == 12 * MIB


def test_pipeline_plan_passes_the_unet_folder_for_a_unet_family(monkeypatch):
    from core.inference.diffusion_families import detect_family

    seen = []

    def companions(
        base,
        staged = None,
        load_dtype = None,
        **kw,
    ):
        seen.append(kw)
        return 0

    monkeypatch.setattr(DiffusionBackend, "_companion_cache_bytes", staticmethod(companions))
    monkeypatch.setattr(
        DiffusionBackend, "_text_encoder_cache_bytes", staticmethod(lambda *a, **k: 0)
    )
    monkeypatch.setattr(DiffusionBackend, "_cache_bytes", staticmethod(lambda *a, **k: 0))
    monkeypatch.setattr(d, "_calibrated_activation", lambda *a, **k: None)
    monkeypatch.setattr(
        d,
        "settled_snapshot_device_memory",
        lambda target, **k: DeviceMemory("cuda", "cuda", "discrete_vram", 6_959, 8_188),
    )
    target = types.SimpleNamespace(
        device = "cuda", backend = "cuda", dtype = None, supports_model_cpu_offload = True
    )
    sdxl = detect_family("stabilityai/stable-diffusion-xl-base-1.0")
    DiffusionBackend()._plan_memory(
        target, None, sdxl.base_repo, sdxl, None, False, kind = "pipeline", repo_id = sdxl.base_repo
    )
    assert seen and seen[-1] == {"denoiser_dirs": ("transformer", "unet")}
    seen.clear()
    flux = detect_family("black-forest-labs/FLUX.1-dev")
    DiffusionBackend()._plan_memory(
        target, None, flux.base_repo, flux, None, False, kind = "pipeline", repo_id = flux.base_repo
    )
    assert seen and seen[-1] == {}, "a DiT family keeps the default call shape"


@pytest.mark.parametrize("gib", [6, 8, 10, 12])
@pytest.mark.parametrize("calibrated", [False, True])
@pytest.mark.parametrize("mode", [None, "fast", "balanced", "low_vram"])
def test_small_cards_place_sdxl_the_same_with_the_split(gib, calibrated, mode):
    """Up to 12 GiB the corrected split leaves SDXL's placement as it was (companion None / whole model)."""
    act = calibrated_image_activation("sdxl") if calibrated else None
    target = types.SimpleNamespace(device = "cuda", backend = "cuda", supports_model_cpu_offload = True)
    memory = DeviceMemory("cuda", "cuda", "discrete_vram", gib * 1024 - 1_229, gib * 1024)

    def place(companion, te):
        p = plan_diffusion_memory(
            target = target,
            device_memory = memory,
            model_dense_mib = 6_775,
            companion_dense_mib = companion,
            text_encoder_dense_mib = te,
            runtime_headroom_mib = 8192,
            requested_mode = mode,
            calibrated_activation = act,
        )
        return p.offload_policy, p.stream_text_encoders, p.vae_tiling, p.vae_slicing

    before_single_file = place(None, None)
    before_pipeline = place(6_775, 1_559)
    after = place(1_719, 1_559)
    assert after == before_single_file == before_pipeline
    if mode is None:
        assert after[0] == OFFLOAD_MODEL
