# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""CPU-only unit tests for the diffusion LoRA trainer's pure helpers.

The training loop needs a GPU + weights, but dataset discovery, config normalisation,
the SDXL add-time-ids, and the dict->config adapter are pure and tested here.
"""

from __future__ import annotations

import json

import pytest

from core.training.diffusion_lora_trainer import (
    DEFAULT_LORA_TARGETS,
    DiffusionLoraConfig,
    _coerce_gradient_checkpointing,
    _config_from_dict,
    compute_sdxl_add_time_ids,
    discover_image_caption_pairs,
    resolve_train_steps,
)


def _touch(p):
    p.write_bytes(b"")


def test_discover_prefers_sidecar_then_metadata_then_instance(tmp_path):
    _touch(tmp_path / "a.png")
    _touch(tmp_path / "b.jpg")
    _touch(tmp_path / "c.webp")
    # a.png captioned via metadata.jsonl only
    (tmp_path / "metadata.jsonl").write_text(
        json.dumps({"file_name": "a.png", "text": "from metadata"}) + "\n", encoding = "utf-8"
    )
    # b.jpg captioned via sidecar only
    (tmp_path / "b.txt").write_text("from sidecar", encoding = "utf-8")
    # c.webp falls back to the instance prompt
    pairs = dict(discover_image_caption_pairs(tmp_path, instance_prompt = "from instance"))
    assert pairs[str(tmp_path / "a.png")] == "from metadata"
    assert pairs[str(tmp_path / "b.jpg")] == "from sidecar"
    assert pairs[str(tmp_path / "c.webp")] == "from instance"


def test_discover_sidecar_overrides_metadata_row(tmp_path):
    # A per-image sidecar is the user's explicit edit and must win over a metadata row
    # for the same image (the labeling grid writes sidecars).
    _touch(tmp_path / "a.png")
    (tmp_path / "metadata.jsonl").write_text(
        json.dumps({"file_name": "a.png", "text": "from metadata"}) + "\n", encoding = "utf-8"
    )
    (tmp_path / "a.txt").write_text("edited sidecar", encoding = "utf-8")
    pairs = dict(discover_image_caption_pairs(tmp_path))
    assert pairs[str(tmp_path / "a.png")] == "edited sidecar"


def test_discover_empty_sidecar_suppresses_metadata_but_uses_instance_prompt(tmp_path):
    # An empty sidecar tombstone must suppress the metadata caption yet leave the image uncaptioned
    # so the dreambooth instance_prompt still applies (not drop the image).
    _touch(tmp_path / "cat.png")
    (tmp_path / "metadata.jsonl").write_text(
        json.dumps({"file_name": "cat.png", "text": "old metadata caption"}) + "\n",
        encoding = "utf-8",
    )
    (tmp_path / "cat.txt").write_text("", encoding = "utf-8")  # empty tombstone
    pairs = discover_image_caption_pairs(tmp_path, instance_prompt = "a photo of sks cat")
    assert pairs == [(str(tmp_path / "cat.png"), "a photo of sks cat")]


def test_discover_empty_sidecar_without_instance_prompt_skips_image(tmp_path):
    # With no instance prompt the tombstoned image is skipped (metadata not resurrected), while a
    # sibling with a real caption is still discovered.
    _touch(tmp_path / "cat.png")
    _touch(tmp_path / "cap.png")
    (tmp_path / "metadata.jsonl").write_text(
        json.dumps({"file_name": "cat.png", "text": "old"})
        + "\n"
        + json.dumps({"file_name": "cap.png", "text": "kept"})
        + "\n",
        encoding = "utf-8",
    )
    (tmp_path / "cat.txt").write_text("", encoding = "utf-8")  # empty tombstone
    pairs = dict(discover_image_caption_pairs(tmp_path))
    assert pairs == {str(tmp_path / "cap.png"): "kept"}


def test_discover_skips_uncaptioned_without_instance_prompt(tmp_path):
    _touch(tmp_path / "cap.png")
    _touch(tmp_path / "nocap.png")
    (tmp_path / "cap.caption").write_text("a caption", encoding = "utf-8")
    pairs = discover_image_caption_pairs(tmp_path)
    assert pairs == [(str(tmp_path / "cap.png"), "a caption")]


def test_discover_captions_jsonl_and_image_key(tmp_path):
    _touch(tmp_path / "x.png")
    (tmp_path / "captions.jsonl").write_text(
        json.dumps({"image": "x.png", "text": "hi"}) + "\n", encoding = "utf-8"
    )
    assert discover_image_caption_pairs(tmp_path) == [(str(tmp_path / "x.png"), "hi")]


def test_discover_custom_caption_column(tmp_path):
    _touch(tmp_path / "x.png")
    (tmp_path / "metadata.jsonl").write_text(
        json.dumps({"file_name": "x.png", "caption": "col"}) + "\n", encoding = "utf-8"
    )
    assert discover_image_caption_pairs(tmp_path, caption_column = "caption")[0][1] == "col"


def test_discover_verify_images_rejects_undecodable(tmp_path):
    # verify_images (opt-in, enabled by the start route) rejects a corrupt / zero-byte image with
    # a clear ValueError -> 400 BEFORE the route frees the resident GPU models, instead of letting
    # the spawned trainer crash in PIL after the teardown. The trainers leave it off (default),
    # since they decode every image anyway.
    from PIL import Image

    good = tmp_path / "good.png"
    Image.new("RGB", (8, 8), "white").save(good)
    (tmp_path / "good.txt").write_text("ok", encoding = "utf-8")
    # A zero-byte file with an image extension + a caption passes filename-only discovery.
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"")
    (tmp_path / "bad.txt").write_text("broken", encoding = "utf-8")

    # Default (verify off): the bad file is accepted (filename-only), matching trainer behavior.
    pairs = dict(discover_image_caption_pairs(tmp_path))
    assert str(bad) in pairs and str(good) in pairs

    # verify_images on: the undecodable file raises a clear ValueError.
    with pytest.raises(ValueError, match = "cannot be decoded"):
        discover_image_caption_pairs(tmp_path, verify_images = True)

    # A dataset of only valid images passes the verify.
    bad.unlink()
    (tmp_path / "bad.txt").unlink()
    assert discover_image_caption_pairs(tmp_path, verify_images = True) == [(str(good), "ok")]


def test_discover_empty_raises(tmp_path):
    _touch(tmp_path / "x.png")  # no captions anywhere, no instance prompt
    with pytest.raises(ValueError, match = "No captioned images"):
        discover_image_caption_pairs(tmp_path)


def test_discover_missing_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        discover_image_caption_pairs(tmp_path / "nope")


def test_config_normalized_defaults():
    cfg = DiffusionLoraConfig(base_model = "b", data_dir = "d", output_dir = "o").normalized()
    assert cfg.lora_alpha == cfg.lora_rank  # alpha defaults to rank
    assert cfg.lora_target_modules == DEFAULT_LORA_TARGETS


@pytest.mark.parametrize(
    "kw",
    [
        {"train_steps": 0},
        {"train_batch_size": 0},
        {"gradient_accumulation_steps": 0},
        {"lora_rank": 0},
        {"resolution": 100},  # not a multiple of 8
        {"resolution": 32},  # too small
        {"mixed_precision": "int4"},
    ],
)
def test_config_normalized_validation(kw):
    with pytest.raises(ValueError):
        DiffusionLoraConfig(base_model = "b", data_dir = "d", output_dir = "o", **kw).normalized()


def test_config_normalized_accepts_mxfp8_dense_base():
    # mxfp8 is a dense speed mode: a dense base + bf16 compute normalises through.
    cfg = DiffusionLoraConfig(
        base_model = "black-forest-labs/FLUX.1-dev",
        data_dir = "d",
        output_dir = "o",
        base_precision = "mxfp8",
    ).normalized()
    assert cfg.base_precision == "mxfp8"


def test_config_normalized_mxfp8_rejects_prequant_base():
    # A prequant (bnb-4bit) base cannot serve the dense mxfp8 base precision.
    with pytest.raises(ValueError, match = "mxfp8"):
        DiffusionLoraConfig(
            base_model = "unsloth/Qwen-Image-2512-unsloth-bnb-4bit",
            data_dir = "d",
            output_dir = "o",
            base_precision = "mxfp8",
        ).normalized()


def test_config_normalized_mxfp8_requires_bf16_compute():
    # Like the other dense modes, mxfp8 trains in bf16 compute; fp16 is refused.
    with pytest.raises(ValueError, match = "mxfp8"):
        DiffusionLoraConfig(
            base_model = "black-forest-labs/FLUX.1-dev",
            data_dir = "d",
            output_dir = "o",
            base_precision = "mxfp8",
            mixed_precision = "fp16",
        ).normalized()


def test_config_normalized_lists_mxfp8_in_invalid_mode_error():
    # The invalid-base_precision message enumerates the allowed modes, including mxfp8.
    with pytest.raises(ValueError, match = "mxfp8"):
        DiffusionLoraConfig(
            base_model = "b", data_dir = "d", output_dir = "o", base_precision = "bogus"
        ).normalized()


def test_config_normalized_krea2_requires_bf16_compute():
    # krea-2 (like qwen-image / z-image) has fp32 RoPE/embedder internals that overflow fp16,
    # so its DiT trains in bf16 only; fp16 must be refused up front by the route preflight,
    # before it reserves training and evicts resident GPU models and the child trainer raises.
    with pytest.raises(ValueError, match = "bf16"):
        DiffusionLoraConfig(
            base_model = "b",
            data_dir = "d",
            output_dir = "o",
            model_family = "krea-2",
            mixed_precision = "fp16",
        ).normalized()


def test_force_bf16_families_matches_trainer_specs():
    # The route-level bf16-only preflight set must list exactly the DiT families whose trainer
    # spec sets force_bf16. If a force_bf16 family is missing from the set (as krea-2 was), an
    # fp16 start passes the route preflight, reserves training + evicts resident models, and
    # only the child trainer raises -- the evict-then-fail the preflight exists to prevent.
    from core.training.diffusion_dit_trainer import _SPECS
    from core.training.diffusion_train_common import _FORCE_BF16_FAMILIES
    assert _FORCE_BF16_FAMILIES == {fam for fam, spec in _SPECS.items() if spec.force_bf16}


def _cfg(**kw):
    return DiffusionLoraConfig(base_model = "b", data_dir = "d", output_dir = "o", **kw)


def test_resolve_train_steps_uses_train_steps_when_epochs_disabled():
    # num_epochs == 0 leaves the explicit train_steps untouched, whatever the image count.
    cfg = _cfg(train_steps = 300, num_epochs = 0)
    assert resolve_train_steps(cfg, 20) == 300
    assert resolve_train_steps(cfg, 1) == 300


def test_resolve_train_steps_epochs_ceil_over_batch_and_grad_accum():
    # One epoch = ceil(N / (batch x grad_accum)) optimizer steps; num_epochs multiplies it.
    # 10 images, batch 4, grad_accum 1 -> ceil(10/4)=3 steps/epoch.
    assert resolve_train_steps(_cfg(num_epochs = 1, train_batch_size = 4), 10) == 3
    assert resolve_train_steps(_cfg(num_epochs = 5, train_batch_size = 4), 10) == 15
    # grad_accum widens the effective batch: 100 images, batch 2, grad_accum 3 -> per_step=6,
    # ceil(100/6)=17 steps/epoch, 2 epochs -> 34.
    cfg = _cfg(num_epochs = 2, train_batch_size = 2, gradient_accumulation_steps = 3)
    assert resolve_train_steps(cfg, 100) == 34
    # An exact multiple does not round up: 8 images / batch 4 -> 2 steps/epoch.
    assert resolve_train_steps(_cfg(num_epochs = 3, train_batch_size = 4), 8) == 6


def test_resolve_train_steps_single_image_dataset():
    # A one-image dataset is one optimizer step per epoch, so num_epochs == steps.
    assert resolve_train_steps(_cfg(num_epochs = 7, train_batch_size = 4), 1) == 7


def test_resolve_train_steps_caps_at_100000():
    # The run length is capped at 100000 even for absurd epoch counts (matches the request
    # model's train_steps ceiling), so a huge epochs x dataset never overflows the loop.
    cfg = _cfg(num_epochs = 1000, train_batch_size = 1)
    assert resolve_train_steps(cfg, 10_000) == 100000


def test_config_normalized_num_epochs_bounds():
    # 0 (disabled) and the 1..1000 range normalise; out-of-range is rejected.
    assert _cfg(num_epochs = 0).normalized().num_epochs == 0
    assert _cfg(num_epochs = 1000).normalized().num_epochs == 1000
    with pytest.raises(ValueError, match = "num_epochs"):
        _cfg(num_epochs = -1).normalized()
    with pytest.raises(ValueError, match = "num_epochs"):
        _cfg(num_epochs = 1001).normalized()


def test_config_from_dict_threads_num_epochs():
    # num_epochs flows through the shared-payload adapter onto the diffusion field.
    cfg = _config_from_dict(
        {"base_model": "b", "data_dir": "d", "output_dir": "o", "num_epochs": 12}
    )
    assert cfg.num_epochs == 12


def test_normalized_rejects_piecewise_constant():
    # piecewise_constant needs a step_rules string the trainers never supply, so get_scheduler()
    # would crash in the trainer subprocess AFTER the resident GPU workloads are freed. It must be
    # rejected up front (a clean ValueError -> 400), not accepted like the other schedulers.
    with pytest.raises(ValueError, match = "lr_scheduler"):
        DiffusionLoraConfig(
            base_model = "b", data_dir = "d", output_dir = "o", lr_scheduler = "piecewise_constant"
        ).normalized()


def test_normalized_accepts_supported_schedulers():
    # Every scheduler in the allow-list runs with only warmup/training steps (no extra required arg).
    for sched in (
        "linear",
        "cosine",
        "cosine_with_restarts",
        "polynomial",
        "constant",
        "constant_with_warmup",
    ):
        cfg = DiffusionLoraConfig(
            base_model = "b", data_dir = "d", output_dir = "o", lr_scheduler = sched
        ).normalized()
        assert cfg.lr_scheduler == sched


def test_api_scheduler_enum_never_advertises_a_rejected_scheduler():
    # The request-model enum must not offer a scheduler that normalized() rejects: a client that
    # picks it straight from the schema would get a 400. Every option the API advertises must be in
    # the validation allow-list (this guards against the enum and allow-list drifting apart again,
    # e.g. piecewise_constant left in one but removed from the other).
    import typing

    from core.training.diffusion_train_common import _LR_SCHEDULERS
    from models.training import DiffusionTrainingStartRequest

    api_options = set(
        typing.get_args(DiffusionTrainingStartRequest.model_fields["lr_scheduler"].annotation)
    )
    assert api_options and api_options <= _LR_SCHEDULERS, api_options - _LR_SCHEDULERS
    assert "piecewise_constant" not in api_options


def test_compute_sdxl_add_time_ids():
    assert compute_sdxl_add_time_ids(1024) == (1024, 1024, 0, 0, 1024, 1024)


def test_config_from_dict_ignores_unknown_and_tuples_targets():
    cfg = _config_from_dict(
        {
            "base_model": "b",
            "data_dir": "d",
            "output_dir": "o",
            "lora_target_modules": ["to_q", "to_v"],
            "unknown_field": 123,  # must be ignored, not crash
        }
    )
    assert cfg.lora_target_modules == ("to_q", "to_v")
    assert not hasattr(cfg, "unknown_field")


def test_config_rejects_zero_lora_alpha():
    # An explicit zero alpha would scale the adapter to nothing; reject it.
    with pytest.raises(ValueError, match = "lora_alpha"):
        DiffusionLoraConfig(base_model = "b", data_dir = "d", output_dir = "o", lora_alpha = 0).normalized()


def test_config_rejects_nonpositive_snr_gamma():
    # gamma <= 0 zeroes/inverts the min-SNR weight; None is the documented disable.
    with pytest.raises(ValueError, match = "snr_gamma"):
        DiffusionLoraConfig(base_model = "b", data_dir = "d", output_dir = "o", snr_gamma = 0).normalized()
    cfg = DiffusionLoraConfig(
        base_model = "b", data_dir = "d", output_dir = "o", snr_gamma = None
    ).normalized()
    assert cfg.snr_gamma is None


def test_config_coerces_string_learning_rate():
    # The Studio config path preserves learning_rate as a string; normalize to float.
    cfg = DiffusionLoraConfig(
        base_model = "b", data_dir = "d", output_dir = "o", learning_rate = "1e-4"
    ).normalized()
    assert cfg.learning_rate == 1e-4
    with pytest.raises(ValueError, match = "learning_rate"):
        DiffusionLoraConfig(
            base_model = "b", data_dir = "d", output_dir = "o", learning_rate = "abc"
        ).normalized()


def test_config_blank_hf_token_is_anonymous():
    cfg = DiffusionLoraConfig(
        base_model = "b", data_dir = "d", output_dir = "o", hf_token = "   "
    ).normalized()
    assert cfg.hf_token is None


def test_config_from_dict_aliases_generic_studio_keys():
    # The generic Studio training payload uses different key names; alias them.
    cfg = _config_from_dict(
        {
            "model_name": "b",
            "data_dir": "d",
            "output_dir": "o",
            "max_steps": 25,
            "batch_size": 3,
            "lora_r": 8,
            "lr_scheduler_type": "cosine",
            "random_seed": 7,
        }
    )
    assert cfg.base_model == "b"
    assert cfg.train_steps == 25
    assert cfg.train_batch_size == 3
    assert cfg.lora_rank == 8
    assert cfg.lr_scheduler == "cosine"
    assert cfg.seed == 7


def test_config_from_dict_canonical_key_beats_alias():
    cfg = _config_from_dict(
        {"base_model": "canon", "model_name": "alias", "data_dir": "d", "output_dir": "o"}
    )
    assert cfg.base_model == "canon"


def test_gradient_checkpointing_string_coercion():
    # Studio sends a string; the disable words are False, everything else truthy True.
    for off in ("none", "None", "false", "0", "no", "off", ""):
        assert _coerce_gradient_checkpointing(off) is False
    for on in ("true", "unsloth", "yes"):
        assert _coerce_gradient_checkpointing(on) is True
    assert _coerce_gradient_checkpointing(True) is True
    assert _coerce_gradient_checkpointing(False) is False
    cfg = _config_from_dict(
        {"base_model": "b", "data_dir": "d", "output_dir": "o", "gradient_checkpointing": "none"}
    )
    assert cfg.gradient_checkpointing is False


def test_config_rejects_nonpositive_learning_rate():
    with pytest.raises(ValueError, match = "learning_rate"):
        DiffusionLoraConfig(
            base_model = "b", data_dir = "d", output_dir = "o", learning_rate = 0
        ).normalized()


def test_config_rejects_untrainable_base_models():
    # GGUF checkpoints and families without a trainer (Kontext editing, SD3) must fail at
    # normalise time (an instant 400 via the API), not minutes later inside from_pretrained.
    for bad in (
        "unsloth/FLUX.1-dev-GGUF",
        "z-image-turbo-Q4_K_M.gguf",
        "stabilityai/stable-diffusion-3-medium",
        "unsloth/FLUX.1-Kontext-dev",
    ):
        with pytest.raises(ValueError):
            DiffusionLoraConfig(base_model = bad, data_dir = "d", output_dir = "o").normalized()


def test_config_resolves_dit_families():
    # FLUX.1 / Qwen-Image / Z-Image bases now resolve to their DiT trainer families.
    for base, fam in (
        ("black-forest-labs/FLUX.1-dev", "flux.1"),
        ("black-forest-labs/FLUX.1-schnell", "flux.1"),
        ("unsloth/Qwen-Image-2512-unsloth-bnb-4bit", "qwen-image"),
        ("Tongyi-MAI/Z-Image-Turbo", "z-image"),
    ):
        cfg = DiffusionLoraConfig(base_model = base, data_dir = "d", output_dir = "o").normalized()
        assert cfg.resolved_family == fam


def test_config_accepts_sdxl_and_unknown_base_models():
    # SDXL names and unclassifiable custom names/paths must pass the guard (a wrong
    # custom pick still fails cleanly in from_pretrained).
    for ok in (
        "stabilityai/stable-diffusion-xl-base-1.0",
        "stabilityai/sdxl-turbo",
        "/data/checkpoints/my-custom-sdxl",
        "my-finetune",
    ):
        cfg = DiffusionLoraConfig(base_model = ok, data_dir = "d", output_dir = "o").normalized()
        assert cfg.base_model == ok


# ── trainer registry + family resolution + metadata sidecar (PR A platform) ──
def test_get_trainer_resolves_sdxl():
    from core.training.diffusion_lora_trainer import get_trainer, run_diffusion_lora_training
    assert get_trainer("sdxl") is run_diffusion_lora_training
    assert get_trainer("SDXL") is run_diffusion_lora_training  # case-insensitive


def test_get_trainer_unknown_family_raises():
    from core.training.diffusion_lora_trainer import get_trainer
    with pytest.raises(ValueError, match = "No trainer"):
        get_trainer("flux.1-kontext")  # a real family with no registered trainer


def test_get_trainer_resolves_dit_families():
    from core.training.diffusion_dit_trainer import run_dit_lora_training
    from core.training.diffusion_lora_trainer import get_trainer
    for fam in ("flux.1", "qwen-image", "z-image", "flux.2-klein", "flux.2-dev"):
        assert get_trainer(fam) is run_dit_lora_training


def test_normalized_sets_resolved_family():
    cfg = DiffusionLoraConfig(
        base_model = "stabilityai/stable-diffusion-xl-base-1.0", data_dir = "d", output_dir = "o"
    ).normalized()
    assert cfg.resolved_family == "sdxl"
    cfg2 = DiffusionLoraConfig(
        base_model = "my-custom-thing", data_dir = "d", output_dir = "o"
    ).normalized()
    assert cfg2.resolved_family == "sdxl"  # unknown -> default SDXL trainer


def test_explicit_model_family_validated():
    from core.training.diffusion_lora_trainer import DiffusionLoraConfig as C

    # A bogus explicit family is rejected up front.
    with pytest.raises(ValueError, match = "Unknown model_family"):
        C(base_model = "b", data_dir = "d", output_dir = "o", model_family = "not-a-family").normalized()
    # A known-but-not-trainable family (Kontext editing) is rejected with a helpful hint.
    with pytest.raises(ValueError):
        C(base_model = "b", data_dir = "d", output_dir = "o", model_family = "flux.1-kontext").normalized()
    # A DiT family that IS trainable resolves to itself.
    assert (
        C(base_model = "b", data_dir = "d", output_dir = "o", model_family = "flux.1")
        .normalized()
        .resolved_family
        == "flux.1"
    )
    # SDXL explicit passes.
    assert (
        C(base_model = "b", data_dir = "d", output_dir = "o", model_family = "sdxl")
        .normalized()
        .resolved_family
        == "sdxl"
    )


def test_publish_writes_metadata_sidecar(tmp_path, monkeypatch):
    import json as _json
    from pathlib import Path

    from core.inference import diffusion_lora
    from core.training.diffusion_lora_trainer import _publish_to_lora_catalog

    loras = tmp_path / "loras"
    loras.mkdir()
    monkeypatch.setattr(diffusion_lora, "loras_dir", lambda: loras)

    src = tmp_path / "run" / "pytorch_lora_weights.safetensors"
    src.parent.mkdir(parents = True)
    src.write_bytes(b"fake-adapter")
    cfg = DiffusionLoraConfig(
        base_model = "stabilityai/sdxl-turbo",
        data_dir = "d",
        output_dir = str(tmp_path / "run"),
        adapter_name = "my.style",
        instance_prompt = "a photo in sks style",
        lora_rank = 8,
    ).normalized()

    dest = _publish_to_lora_catalog(str(src), cfg)
    assert dest is not None
    sidecar = Path(dest).with_suffix(".json")
    assert sidecar.is_file()
    meta = _json.loads(sidecar.read_text())
    assert meta["family"] == "sdxl"
    assert meta["families"] == ["sdxl"]
    assert meta["base_model"] == "stabilityai/sdxl-turbo"
    assert meta["lora_rank"] == 8
    assert meta["trigger_prompt"] == "a photo in sks style"
    assert meta["source"] == "studio-trained"


def test_publish_does_not_clobber_same_name_adapter(tmp_path, monkeypatch):
    # A retrain with the same adapter name must not overwrite a prior mirror: the second
    # publish lands under a numeric suffix (my-style -> my-style-2), sidecar alongside it.
    from pathlib import Path

    from core.inference import diffusion_lora
    from core.training.diffusion_lora_trainer import _publish_to_lora_catalog

    loras = tmp_path / "loras"
    loras.mkdir()
    monkeypatch.setattr(diffusion_lora, "loras_dir", lambda: loras)

    def _publish(payload: bytes) -> str:
        src = tmp_path / "run" / "pytorch_lora_weights.safetensors"
        src.parent.mkdir(parents = True, exist_ok = True)
        src.write_bytes(payload)
        cfg = DiffusionLoraConfig(
            base_model = "stabilityai/sdxl-turbo",
            data_dir = "d",
            output_dir = str(tmp_path / "run"),
            adapter_name = "my-style",
        ).normalized()
        return _publish_to_lora_catalog(str(src), cfg)

    first = _publish(b"adapter-v1")
    second = _publish(b"adapter-v2")
    assert Path(first).name == "my-style.safetensors"
    assert Path(second).name == "my-style-2.safetensors"
    # The first mirror is intact (not clobbered) and the second is the new content.
    assert Path(first).read_bytes() == b"adapter-v1"
    assert Path(second).read_bytes() == b"adapter-v2"
    assert Path(second).with_suffix(".json").is_file()


def test_config_rejects_bad_lr_scheduler():
    # A typo'd scheduler ('constnat') must fail at normalize time, not later in the subprocess.
    with pytest.raises(ValueError, match = "lr_scheduler"):
        DiffusionLoraConfig(
            base_model = "b", data_dir = "d", output_dir = "o", lr_scheduler = "constnat"
        ).normalized()
    # A valid diffusers scheduler passes.
    cfg = DiffusionLoraConfig(
        base_model = "b", data_dir = "d", output_dir = "o", lr_scheduler = "cosine"
    ).normalized()
    assert cfg.lr_scheduler == "cosine"


def test_config_rejects_fp16_on_bf16_only_family():
    # qwen-image / z-image are bf16-only: an fp16 request must be rejected before spawn,
    # in normalized(), not only by the subprocess-side guard.
    for base in ("Tongyi-MAI/Z-Image-Turbo", "unsloth/Qwen-Image-2512-unsloth-bnb-4bit"):
        with pytest.raises(ValueError, match = "bf16"):
            DiffusionLoraConfig(
                base_model = base, data_dir = "d", output_dir = "o", mixed_precision = "fp16"
            ).normalized()
    # FLUX (not force-bf16) still accepts fp16.
    cfg = DiffusionLoraConfig(
        base_model = "black-forest-labs/FLUX.1-dev",
        data_dir = "d",
        output_dir = "o",
        mixed_precision = "fp16",
    ).normalized()
    assert cfg.mixed_precision == "fp16"


def test_gguf_substring_does_not_reject_local_diffusers_dir(tmp_path):
    # A local diffusers directory whose path merely contains 'gguf' is a valid training base
    # (it carries model_index.json, not GGUF weights); the broad substring must not reject it.
    from core.training.diffusion_train_common import resolve_trainable_family

    local = tmp_path / "my-gguf-experiments" / "sdxl-finetune"
    local.mkdir(parents = True)
    (local / "model_index.json").write_text("{}", encoding = "utf-8")
    assert resolve_trainable_family(str(local)) == "sdxl"
    # A real .gguf file still rejects even inside such a dir.
    with pytest.raises(ValueError, match = "GGUF"):
        resolve_trainable_family(str(local / "weights.gguf"))
    # A *-GGUF repo id (not a local dir) still rejects.
    with pytest.raises(ValueError, match = "GGUF"):
        resolve_trainable_family("unsloth/FLUX.1-dev-GGUF")
