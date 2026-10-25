# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""SDXL fine-tunes named for no family (NoobAI-XL V-Pred, Illustrious) are detected from their own
model_index.json / checkpoint header, and load with the scheduler the checkpoint asks for.
Config-only fixtures: safetensors headers with empty tensors, no weights."""

from __future__ import annotations

import json
import struct
import types
from pathlib import Path

import pytest

import core.inference.diffusion_hub_evidence as evidence
from core.inference.diffusion_families import detect_family_for_pick
from core.inference.diffusion_sampler import (
    checkpoint_prediction_overrides,
    pipeline_prediction_overrides,
)

NOOB = "Laxhar/noobai-XL-Vpred-1.0"
NOOB_FILE = "NoobAI-XL-Vpred-v1.0.safetensors"
ILLU = "OnomaAIResearch/Illustrious-xl-early-release-v0"
ILLU_FILE = "Illustrious-XL-v0.1.safetensors"
SHA = "0123456789abcdef0123456789abcdef01234567"

# Laxhar/noobai-XL-Vpred-1.0 model_index.json (trimmed) and scheduler_config.json: the repo's
# diffusers scheduler says epsilon; only the root single file carries v_pred / ztsnr.
SDXL_INDEX = {
    "_class_name": "StableDiffusionXLPipeline",
    "scheduler": ["diffusers", "EulerDiscreteScheduler"],
    "unet": ["diffusers", "UNet2DConditionModel"],
    "text_encoder_2": ["transformers", "CLIPTextModelWithProjection"],
}
NOOB_SCHEDULER = dict(
    beta_end = 0.012,
    beta_schedule = "scaled_linear",
    beta_start = 0.00085,
    num_train_timesteps = 1000,
    prediction_type = "epsilon",
    rescale_betas_zero_snr = False,
    steps_offset = 1,
    timestep_spacing = "leading",
)
# The ComfyUI / original-SD layout of an SDXL checkpoint (label_emb = SDXL's ADM conditioning).
SDXL_UNET_KEYS = [
    "model.diffusion_model.input_blocks.0.0.weight",
    "model.diffusion_model.middle_block.0.in_layers.0.weight",
    "model.diffusion_model.output_blocks.0.0.in_layers.0.weight",
    "model.diffusion_model.label_emb.0.0.weight",
    "model.diffusion_model.time_embed.0.weight",
    "conditioner.embedders.1.model.transformer.resblocks.0.attn.in_proj_weight",
    "first_stage_model.decoder.conv_in.weight",
]
SD15_UNET_KEYS = [
    "model.diffusion_model.input_blocks.0.0.weight",
    "model.diffusion_model.middle_block.0.in_layers.0.weight",
    "model.diffusion_model.output_blocks.0.0.in_layers.0.weight",
    "model.diffusion_model.time_embed.0.weight",
]


def _write_safetensors(path: Path, keys) -> Path:
    header = {k: {"dtype": "F16", "shape": [0], "data_offsets": [0, 0]} for k in keys}
    blob = json.dumps(header).encode()
    path.parent.mkdir(parents = True, exist_ok = True)
    path.write_bytes(struct.pack("<Q", len(blob)) + blob)
    return path


@pytest.fixture(autouse = True)
def _private_studio_cache(tmp_path, monkeypatch):
    """Recorded marker reads land in a per-test Studio cache, never the real one."""
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio_home"))


@pytest.fixture
def hub_cache(tmp_path, monkeypatch):
    """An empty HF hub cache root, pinned as the live one; ``put`` caches a repo file."""
    root = tmp_path / "hub"
    root.mkdir()
    import utils.hf_cache_settings as settings

    monkeypatch.setattr(settings, "active_hf_hub_cache", lambda: str(root))
    # The import-time root is the shared HF cache; these fake ids are never in it, but keep it out anyway.
    import huggingface_hub

    real = huggingface_hub.try_to_load_from_cache

    def only_tmp(
        repo_id,
        filename,
        cache_dir = None,
        **kw,
    ):
        return real(repo_id, filename, cache_dir = str(root), **kw)

    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", only_tmp)
    evidence._HEADER_VERDICTS.clear()

    def put(
        repo_id: str,
        filename: str,
        *,
        payload = None,
        keys = None,
    ) -> Path:
        repo_dir = root / ("models--" + repo_id.replace("/", "--"))
        (repo_dir / "refs").mkdir(parents = True, exist_ok = True)
        (repo_dir / "refs" / "main").write_text(SHA)
        target = repo_dir / "snapshots" / SHA / filename
        target.parent.mkdir(parents = True, exist_ok = True)
        if keys is not None:
            return _write_safetensors(target, keys)
        target.write_text(json.dumps(payload))
        return target

    yield types.SimpleNamespace(root = root, put = put)
    evidence._HEADER_VERDICTS.clear()


def _name(fam):
    return fam.name if fam is not None else None


# ---- detection ---------------------------------------------------------------------------------


def test_a_cached_index_alone_is_not_evidence(hub_cache):
    # The picked file's header is the evidence; a repo's model_index.json does not vouch for anything.
    hub_cache.put(NOOB, "model_index.json", payload = SDXL_INDEX)
    assert detect_family_for_pick(NOOB) is None
    assert detect_family_for_pick(NOOB, NOOB_FILE) is None


def test_vpred_single_file_detected_from_cached_header(hub_cache):
    path = hub_cache.put(NOOB, NOOB_FILE, keys = SDXL_UNET_KEYS + ["v_pred", "ztsnr"])
    assert _name(detect_family_for_pick(NOOB, NOOB_FILE)) == "sdxl"
    # ComfyUI supported_models.SDXL.model_type: v_pred -> V_PREDICTION, ztsnr -> zsnr sampling.
    assert checkpoint_prediction_overrides(str(path)) == {
        "prediction_type": "v_prediction",
        "rescale_betas_zero_snr": True,
        "timestep_spacing": "trailing",
    }


def test_eps_sdxl_single_file_detected_and_left_epsilon(hub_cache):
    path = hub_cache.put(ILLU, ILLU_FILE, keys = SDXL_UNET_KEYS)
    assert _name(detect_family_for_pick(ILLU, ILLU_FILE)) == "sdxl"
    assert checkpoint_prediction_overrides(str(path)) == {}


def test_checkpoint_header_outranks_the_repo_index(hub_cache):
    hub_cache.put(NOOB, "model_index.json", payload = SDXL_INDEX)
    hub_cache.put(NOOB, "sd15.safetensors", keys = SD15_UNET_KEYS)
    assert detect_family_for_pick(NOOB, "sd15.safetensors") is None


def test_index_does_not_vouch_for_a_file_in_a_subfolder(hub_cache):
    hub_cache.put(NOOB, "model_index.json", payload = SDXL_INDEX)
    assert detect_family_for_pick(NOOB, "extras/other.safetensors") is None


def test_fetch_reads_header_then_detection_is_offline(hub_cache, monkeypatch):
    import huggingface_hub

    calls = []

    def parse(
        self,
        repo_id,
        filename,
        token = None,
        **kw,
    ):
        calls.append((repo_id, filename))
        tensors = {k: types.SimpleNamespace(shape = [0]) for k in SDXL_UNET_KEYS + ["v_pred"]}
        return types.SimpleNamespace(tensors = tensors, metadata = {})

    monkeypatch.setattr(huggingface_hub.HfApi, "parse_safetensors_file_metadata", parse)
    monkeypatch.setattr(
        huggingface_hub,
        "hf_hub_download",
        lambda *a, **k: pytest.fail("the header answered; no index fetch"),
    )
    assert detect_family_for_pick(NOOB, NOOB_FILE) is None
    evidence.prefetch_for_pick(NOOB, NOOB_FILE, None, None)
    assert calls == [(NOOB, NOOB_FILE)]
    assert _name(detect_family_for_pick(NOOB, NOOB_FILE)) == "sdxl"
    # Validation after the prefetch answers from the remembered header, with no further read.
    assert _name(detect_family_for_pick(NOOB, NOOB_FILE)) == "sdxl"
    assert calls == [(NOOB, NOOB_FILE)]


def test_a_pipeline_pick_makes_no_hub_call(hub_cache, monkeypatch):
    import huggingface_hub

    def boom(*a, **k):
        raise AssertionError("a pipeline pick reads no evidence")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", boom)
    monkeypatch.setattr(huggingface_hub.HfApi, "parse_safetensors_file_metadata", boom)
    evidence.prefetch_for_pick(NOOB, None, None, "hf_x")
    assert detect_family_for_pick(NOOB) is None


def test_prefetch_is_skipped_for_named_overridden_local_and_gguf_picks(
    hub_cache, monkeypatch, tmp_path
):
    import huggingface_hub

    def boom(*a, **k):
        raise AssertionError("no network for this pick")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", boom)
    monkeypatch.setattr(huggingface_hub.HfApi, "parse_safetensors_file_metadata", boom)
    evidence.prefetch_for_pick("stabilityai/stable-diffusion-xl-base-1.0", None, None, None)
    evidence.prefetch_for_pick(NOOB, NOOB_FILE, "sdxl", None)
    evidence.prefetch_for_pick(NOOB, "noob-Q4_K.gguf", None, None)
    evidence.prefetch_for_pick(str(tmp_path), None, None, None)


def test_name_veto_is_not_overridden_by_the_header(hub_cache):
    # The name matched SDXL and then refused the edit variant; a cached SDXL header must not undo that.
    repo = "someone/neutral"
    hub_cache.put(repo, "sdxl-edit.safetensors", keys = SDXL_UNET_KEYS)
    assert detect_family_for_pick(repo, "sdxl-edit.safetensors") is None


def test_named_sdxl_repos_never_read_evidence(monkeypatch):
    monkeypatch.setattr(
        evidence, "cached_family", lambda *a, **k: pytest.fail("name already answered")
    )
    assert _name(detect_family_for_pick("stabilityai/stable-diffusion-xl-base-1.0")) == "sdxl"
    assert _name(detect_family_for_pick("stabilityai/sdxl-turbo")) == "sdxl"


# ---- scheduler for a diffusers pipeline directory ----------------------------------------------


def _pipeline_dir(tmp_path: Path, scheduler: dict, root_singles: dict) -> Path:
    root = tmp_path / "pipe"
    (root / "scheduler").mkdir(parents = True)
    (root / "model_index.json").write_text(json.dumps(SDXL_INDEX))
    (root / "scheduler" / "scheduler_config.json").write_text(json.dumps(scheduler))
    for name, keys in root_singles.items():
        _write_safetensors(root / name, keys)
    return root


def test_vpred_folder_with_eps_config_takes_the_root_file_markers(tmp_path):
    root = _pipeline_dir(
        tmp_path, NOOB_SCHEDULER, {NOOB_FILE: SDXL_UNET_KEYS + ["v_pred", "ztsnr"]}
    )
    assert pipeline_prediction_overrides(NOOB_SCHEDULER, str(root)) == {
        "prediction_type": "v_prediction",
        "rescale_betas_zero_snr": True,
        "timestep_spacing": "trailing",
    }


def test_eps_sdxl_folder_is_unchanged(tmp_path):
    root = _pipeline_dir(tmp_path, NOOB_SCHEDULER, {ILLU_FILE: SDXL_UNET_KEYS})
    assert pipeline_prediction_overrides(NOOB_SCHEDULER, str(root)) == {}
    assert pipeline_prediction_overrides(NOOB_SCHEDULER, None) == {}
    assert pipeline_prediction_overrides(NOOB_SCHEDULER, "stabilityai/sdxl-turbo") == {}


def test_disagreeing_root_files_keep_the_config(tmp_path):
    root = _pipeline_dir(
        tmp_path,
        NOOB_SCHEDULER,
        {"eps.safetensors": SDXL_UNET_KEYS, "vpred.safetensors": SDXL_UNET_KEYS + ["v_pred"]},
    )
    assert pipeline_prediction_overrides(NOOB_SCHEDULER, str(root)) == {}


def test_declared_vpred_zsnr_config_samples_the_terminal_timestep():
    config = {**NOOB_SCHEDULER, "prediction_type": "v_prediction", "rescale_betas_zero_snr": True}
    assert pipeline_prediction_overrides(config, None) == {"timestep_spacing": "trailing"}
    assert pipeline_prediction_overrides({**config, "timestep_spacing": "trailing"}, None) == {}
    # v-prediction without zero SNR is honoured as declared.
    assert (
        pipeline_prediction_overrides({**NOOB_SCHEDULER, "prediction_type": "v_prediction"}, None)
        == {}
    )


def test_apply_rebuilds_a_real_scheduler(tmp_path):
    diffusers = pytest.importorskip("diffusers")
    from core.inference.diffusion_sampler import apply_pipeline_prediction_type

    root = _pipeline_dir(
        tmp_path, NOOB_SCHEDULER, {NOOB_FILE: SDXL_UNET_KEYS + ["v_pred", "ztsnr"]}
    )
    pipe = types.SimpleNamespace(
        scheduler = diffusers.EulerDiscreteScheduler.from_config(NOOB_SCHEDULER)
    )
    assert apply_pipeline_prediction_type(pipe, str(root)) is True
    cfg = pipe.scheduler.config
    assert (cfg.prediction_type, cfg.rescale_betas_zero_snr, cfg.timestep_spacing) == (
        "v_prediction",
        True,
        "trailing",
    )
    pipe.scheduler.set_timesteps(25)
    assert int(pipe.scheduler.timesteps[0]) == 999  # starts at the zero-SNR terminal step
    assert apply_pipeline_prediction_type(pipe, str(root)) is False  # idempotent

    eps_root = _pipeline_dir(tmp_path / "eps", NOOB_SCHEDULER, {ILLU_FILE: SDXL_UNET_KEYS})
    eps = types.SimpleNamespace(
        scheduler = diffusers.EulerDiscreteScheduler.from_config(NOOB_SCHEDULER)
    )
    assert apply_pipeline_prediction_type(eps, str(eps_root)) is False
    assert eps.scheduler.config.prediction_type == "epsilon"


SDXL_DIFFUSERS_UNET_ONLY_KEYS = [
    "conv_in.weight",
    "time_embedding.linear_1.weight",
    "add_embedding.linear_1.weight",
    "down_blocks.0.resnets.0.conv1.weight",
    "mid_block.resnets.0.conv1.weight",
    "up_blocks.0.resnets.0.conv1.weight",
]


def _stub_header(monkeypatch, keys_by_call):
    import huggingface_hub

    calls = []

    def parse(
        self,
        repo_id,
        filename,
        token = None,
        **kw,
    ):
        keys = keys_by_call[min(len(calls), len(keys_by_call) - 1)]
        calls.append((repo_id, filename))
        return types.SimpleNamespace(
            tensors = {k: types.SimpleNamespace(shape = [0]) for k in keys}, metadata = {}
        )

    monkeypatch.setattr(huggingface_hub.HfApi, "parse_safetensors_file_metadata", parse)
    return calls


def test_a_denoiser_only_unet_is_not_evidence_for_a_whole_pipeline_family(hub_cache, monkeypatch):
    import huggingface_hub

    _stub_header(monkeypatch, [SDXL_DIFFUSERS_UNET_ONLY_KEYS])
    monkeypatch.setattr(
        huggingface_hub, "hf_hub_download", lambda *a, **k: pytest.fail("header read")
    )
    evidence.prefetch_for_pick(NOOB, "unet.safetensors", None, None)
    assert detect_family_for_pick(NOOB, "unet.safetensors") is None
    hub_cache.put(ILLU, "unet.safetensors", keys = SDXL_DIFFUSERS_UNET_ONLY_KEYS)
    assert detect_family_for_pick(ILLU, "unet.safetensors") is None


def test_each_prefetch_rereads_a_replaced_header(hub_cache, monkeypatch):
    calls = _stub_header(monkeypatch, [SD15_UNET_KEYS, SDXL_UNET_KEYS + ["v_pred"]])
    evidence.prefetch_for_pick(NOOB, NOOB_FILE, None, None)
    assert detect_family_for_pick(NOOB, NOOB_FILE) is None
    # The repo replaced the file with an SDXL checkpoint: the next request judges the new header.
    evidence.prefetch_for_pick(NOOB, NOOB_FILE, None, None)
    assert len(calls) == 2
    assert _name(detect_family_for_pick(NOOB, NOOB_FILE)) == "sdxl"


def test_a_remembered_header_expires(hub_cache, monkeypatch):
    _stub_header(monkeypatch, [SDXL_UNET_KEYS])
    evidence.prefetch_for_pick(NOOB, NOOB_FILE, None, None)
    assert _name(detect_family_for_pick(NOOB, NOOB_FILE)) == "sdxl"
    now = evidence.time.monotonic()
    monkeypatch.setattr(evidence.time, "monotonic", lambda: now + evidence._VERDICT_TTL_S + 1)
    assert detect_family_for_pick(NOOB, NOOB_FILE) is None


def test_the_prefetch_writes_nothing_to_the_shared_cache(hub_cache, monkeypatch):
    import huggingface_hub

    monkeypatch.setattr(
        huggingface_hub, "hf_hub_download", lambda *a, **k: pytest.fail("header range read only")
    )
    _stub_header(monkeypatch, [SDXL_UNET_KEYS])
    evidence.prefetch_for_pick(NOOB, NOOB_FILE, None, "hf_private")
    assert _name(detect_family_for_pick(NOOB, NOOB_FILE)) == "sdxl"
    assert list(hub_cache.root.iterdir()) == []


def test_a_cache_only_prefetch_never_touches_the_hub(hub_cache, monkeypatch):
    import huggingface_hub

    def boom(*a, **k):
        raise AssertionError("an automatic load is cache-only")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", boom)
    monkeypatch.setattr(huggingface_hub.HfApi, "parse_safetensors_file_metadata", boom)
    evidence.prefetch_for_pick(NOOB, NOOB_FILE, None, None, allow_network = False)
    evidence.prefetch_for_pick(NOOB, None, None, None, allow_network = False)
    assert detect_family_for_pick(NOOB, NOOB_FILE) is None


def test_the_load_route_passes_user_initiated_to_the_prefetch():
    import ast

    source = (Path(__file__).resolve().parents[1] / "routes" / "inference.py").read_text(
        encoding = "utf-8"
    )
    tree = ast.parse(source)
    route = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.AsyncFunctionDef) and n.name == "load_diffusion_model_gated"
    )
    calls = [
        c
        for c in ast.walk(route)
        if isinstance(c, ast.Call)
        and any(isinstance(a, ast.Name) and a.id == "prefetch_for_pick" for a in c.args)
    ]
    assert len(calls) == 1
    passed = [a.id for a in calls[0].args if isinstance(a, ast.Name)] + [
        k.value.id for k in calls[0].keywords if isinstance(k.value, ast.Name)
    ]
    assert "user_initiated" in passed


FLUX1_DEV_KEYS = [
    "transformer_blocks.0.attn.to_q.weight",
    "single_transformer_blocks.0.attn.to_q.weight",
    "x_embedder.weight",
    "context_embedder.weight",
    "time_text_embed.timestep_embedder.linear_1.weight",
    "time_text_embed.guidance_embedder.linear_1.weight",
]


def test_a_flux_dev_header_is_not_evidence_for_an_unnamed_pick(hub_cache, monkeypatch):
    # FLUX.1 dev and schnell share one layout; resolved by family alone it would get schnell's base and defaults.
    _stub_header(monkeypatch, [FLUX1_DEV_KEYS])
    evidence.prefetch_for_pick("someone/neutral-ft", "model.safetensors", None, None)
    assert detect_family_for_pick("someone/neutral-ft", "model.safetensors") is None
    hub_cache.put("someone/cached-ft", "model.safetensors", keys = FLUX1_DEV_KEYS)
    assert detect_family_for_pick("someone/cached-ft", "model.safetensors") is None


def test_a_transformer_family_index_is_not_evidence_for_an_unnamed_pick(hub_cache):
    hub_cache.put(
        "someone/neutral-pipe", "model_index.json", payload = {"_class_name": "FluxPipeline"}
    )
    assert detect_family_for_pick("someone/neutral-pipe") is None


def test_the_anonymous_token_sentinel_reaches_every_hub_call(hub_cache, monkeypatch):
    import huggingface_hub

    seen = []

    def parse(
        self,
        repo_id,
        filename,
        token = None,
        **kw,
    ):
        seen.append(("header", token))
        raise OSError("not a safetensors")

    def size(
        url,
        token = None,
        **kw,
    ):
        seen.append(("size", token))
        return types.SimpleNamespace(size = 400)

    def download(
        repo_id,
        filename,
        token = None,
        cache_dir = None,
        **kw,
    ):
        seen.append(("download", token))
        return str(hub_cache.put(repo_id, filename, payload = SDXL_INDEX))

    monkeypatch.setattr(huggingface_hub.HfApi, "parse_safetensors_file_metadata", parse)
    monkeypatch.setattr(huggingface_hub, "get_hf_file_metadata", size)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    evidence.prefetch_for_pick(NOOB, NOOB_FILE, None, False)
    assert seen == [("header", False)]


def test_a_positive_verdict_is_reread_when_the_hub_file_changes(hub_cache, monkeypatch):
    calls = _stub_header(monkeypatch, [SDXL_UNET_KEYS, SD15_UNET_KEYS])
    evidence.prefetch_for_pick(NOOB, NOOB_FILE, None, None)
    assert _name(detect_family_for_pick(NOOB, NOOB_FILE)) == "sdxl"
    # The repo replaced the SDXL checkpoint with an SD 1.5 one: the next request must see it.
    evidence.prefetch_for_pick(NOOB, NOOB_FILE, None, None)
    assert len(calls) == 2
    assert detect_family_for_pick(NOOB, NOOB_FILE) is None


def test_a_cached_copy_is_rechecked_against_the_current_hub_header(hub_cache, monkeypatch):
    hub_cache.put(NOOB, NOOB_FILE, keys = SDXL_UNET_KEYS)
    assert _name(detect_family_for_pick(NOOB, NOOB_FILE)) == "sdxl"
    calls = _stub_header(monkeypatch, [SD15_UNET_KEYS])
    evidence.prefetch_for_pick(NOOB, NOOB_FILE, None, None)
    assert len(calls) == 1
    assert detect_family_for_pick(NOOB, NOOB_FILE) is None


def test_a_named_pick_never_reads_hub_evidence(hub_cache, monkeypatch):
    calls = _stub_header(monkeypatch, [SDXL_UNET_KEYS])
    evidence.prefetch_for_pick("someone/my-sdxl-ft", "model.safetensors", None, None)
    assert calls == []


def test_root_single_file_extensions_match_case_insensitively(tmp_path):
    root = _pipeline_dir(
        tmp_path,
        NOOB_SCHEDULER,
        {"NOOBAI-XL-VPRED.SAFETENSORS": SDXL_UNET_KEYS + ["v_pred", "ztsnr"]},
    )
    assert pipeline_prediction_overrides(NOOB_SCHEDULER, str(root)) == {
        "prediction_type": "v_prediction",
        "rescale_betas_zero_snr": True,
        "timestep_spacing": "trailing",
    }


def test_a_staged_pipeline_snapshot_reads_the_root_markers_off_the_hub(tmp_path, monkeypatch):
    import huggingface_hub

    # The staged snapshot keeps only model_index.json at its root; the packaged single file stays on the Hub.
    root = _pipeline_dir(tmp_path, NOOB_SCHEDULER, {})
    seen = []

    def files(
        self,
        repo_id,
        token = None,
        **kw,
    ):
        seen.append(("list", repo_id, token))
        return ["model_index.json", NOOB_FILE, "unet/diffusion_pytorch_model.safetensors"]

    def parse(
        self,
        repo_id,
        filename,
        token = None,
        **kw,
    ):
        seen.append(("header", filename, token))
        keys = SDXL_UNET_KEYS + ["v_pred", "ztsnr"]
        return types.SimpleNamespace(tensors = {k: None for k in keys}, metadata = {})

    monkeypatch.setattr(huggingface_hub.HfApi, "list_repo_files", files)
    monkeypatch.setattr(huggingface_hub.HfApi, "parse_safetensors_file_metadata", parse)
    want = {
        "prediction_type": "v_prediction",
        "rescale_betas_zero_snr": True,
        "timestep_spacing": "trailing",
    }
    assert pipeline_prediction_overrides(NOOB_SCHEDULER, str(root), NOOB, "hf_x", True) == want
    assert seen == [("list", NOOB, "hf_x"), ("header", NOOB_FILE, "hf_x")]
    # A cache-only load never reaches the Hub; the shipped config stands.
    seen.clear()
    assert pipeline_prediction_overrides(NOOB_SCHEDULER, str(root), NOOB, None, False) == {}
    assert seen == []


def _stub_noob_hub(monkeypatch, keys, seen):
    import huggingface_hub

    def files(
        self,
        repo_id,
        token = None,
        **kw,
    ):
        seen.append("list")
        return ["model_index.json", NOOB_FILE]

    def parse(
        self,
        repo_id,
        filename,
        token = None,
        **kw,
    ):
        seen.append("header")
        if keys is None:
            raise OSError("offline")
        return types.SimpleNamespace(tensors = {k: None for k in keys}, metadata = {})

    monkeypatch.setattr(huggingface_hub.HfApi, "list_repo_files", files)
    monkeypatch.setattr(huggingface_hub.HfApi, "parse_safetensors_file_metadata", parse)


def _staged_snapshot(hub_cache, commit = SHA) -> Path:
    # A staged pipeline snapshot: model_index.json + scheduler config, no root single file.
    hub_cache.put(NOOB, "model_index.json", payload = SDXL_INDEX)
    target = hub_cache.put(NOOB, "scheduler/scheduler_config.json", payload = NOOB_SCHEDULER)
    snap = target.parent.parent
    if commit != SHA:
        renamed = snap.parent / commit
        snap.rename(renamed)
        (snap.parent.parent / "refs" / "main").write_text(commit)
        snap = renamed
    return snap


VPRED = {
    "prediction_type": "v_prediction",
    "rescale_betas_zero_snr": True,
    "timestep_spacing": "trailing",
}


def test_a_cache_only_reload_of_a_staged_snapshot_keeps_the_online_marker_read(
    hub_cache, monkeypatch
):
    diffusers = pytest.importorskip("diffusers")
    from core.inference.diffusion_sampler import apply_pipeline_prediction_type

    snap = _staged_snapshot(hub_cache)
    seen = []
    _stub_noob_hub(monkeypatch, SDXL_UNET_KEYS + ["v_pred", "ztsnr"], seen)
    assert pipeline_prediction_overrides(NOOB_SCHEDULER, str(snap), NOOB, None, True) == VPRED
    assert seen == ["list", "header"]
    # Offline loads (by snapshot path or repo id) reuse the recorded markers.
    seen.clear()
    _stub_noob_hub(monkeypatch, None, seen)
    assert pipeline_prediction_overrides(NOOB_SCHEDULER, str(snap), NOOB, None, False) == VPRED
    assert pipeline_prediction_overrides(NOOB_SCHEDULER, None, NOOB, None, False) == VPRED
    assert seen == []
    pipe = types.SimpleNamespace(
        scheduler = diffusers.EulerDiscreteScheduler.from_config(NOOB_SCHEDULER)
    )
    assert apply_pipeline_prediction_type(pipe, str(snap), repo_id = NOOB, allow_network = False)
    cfg = pipe.scheduler.config
    assert (cfg.prediction_type, cfg.rescale_betas_zero_snr, cfg.timestep_spacing) == (
        "v_prediction",
        True,
        "trailing",
    )


def test_a_failed_online_marker_read_falls_back_to_the_recorded_one(hub_cache, monkeypatch):
    snap = _staged_snapshot(hub_cache)
    seen = []
    _stub_noob_hub(monkeypatch, SDXL_UNET_KEYS + ["v_pred", "ztsnr"], seen)
    assert pipeline_prediction_overrides(NOOB_SCHEDULER, str(snap), NOOB, None, True) == VPRED
    _stub_noob_hub(monkeypatch, None, seen)
    assert pipeline_prediction_overrides(NOOB_SCHEDULER, str(snap), NOOB, None, True) == VPRED


def test_a_recorded_marker_read_is_per_snapshot_commit(hub_cache, monkeypatch):
    snap = _staged_snapshot(hub_cache)
    _stub_noob_hub(monkeypatch, SDXL_UNET_KEYS + ["v_pred", "ztsnr"], [])
    assert pipeline_prediction_overrides(NOOB_SCHEDULER, str(snap), NOOB, None, True) == VPRED
    other = snap.parent / ("f" * 40)
    other.mkdir()
    # Another commit of the repo was never read: its shipped config stands offline.
    assert pipeline_prediction_overrides(NOOB_SCHEDULER, str(other), NOOB, None, False) == {}
    # An online read that finds no markers replaces the record, so an eps re-release stays eps offline.
    _stub_noob_hub(monkeypatch, SDXL_UNET_KEYS, [])
    assert pipeline_prediction_overrides(NOOB_SCHEDULER, str(snap), NOOB, None, True) == {}
    assert pipeline_prediction_overrides(NOOB_SCHEDULER, str(snap), NOOB, None, False) == {}


def test_an_unrecorded_cache_only_load_keeps_the_shipped_config(hub_cache, monkeypatch):
    snap = _staged_snapshot(hub_cache)
    seen = []
    _stub_noob_hub(monkeypatch, SDXL_UNET_KEYS + ["v_pred", "ztsnr"], seen)
    assert pipeline_prediction_overrides(NOOB_SCHEDULER, str(snap), NOOB, None, False) == {}
    assert seen == []


def test_the_load_passes_the_repo_to_the_pipeline_marker_check():
    import ast

    source = (
        Path(__file__).resolve().parents[1] / "core" / "inference" / "diffusion.py"
    ).read_text(encoding = "utf-8")
    calls = [
        c
        for c in ast.walk(ast.parse(source))
        if isinstance(c, ast.Call)
        and getattr(c.func, "id", None) == "apply_pipeline_prediction_type"
    ]
    assert len(calls) == 1
    assert {k.arg for k in calls[0].keywords} >= {"repo_id", "hf_token", "allow_network"}


@pytest.mark.parametrize(
    "repo, filename",
    [
        ("someone/neutral-inpaint", "model.safetensors"),
        ("someone/neutral-ft", "model-inpainting-edit.safetensors"),
        ("someone/neutral-ft", "inpaint/model.safetensors"),
    ],
)
def test_a_variant_named_pick_is_not_answered_from_its_header(hub_cache, repo, filename):
    hub_cache.put(repo, filename, keys = SDXL_UNET_KEYS)
    assert detect_family_for_pick(repo, filename) is None


def test_a_refresh_keeps_the_last_verdict_visible(hub_cache, monkeypatch):
    import huggingface_hub

    _stub_header(monkeypatch, [SDXL_UNET_KEYS])
    evidence.prefetch_for_pick(NOOB, NOOB_FILE, None, None)
    during = []

    def parse(
        self,
        repo_id,
        filename,
        token = None,
        **kw,
    ):
        # An overlapping request validating while this refresh reads the header.
        during.append(_name(detect_family_for_pick(NOOB, NOOB_FILE)))
        raise OSError("offline")

    monkeypatch.setattr(huggingface_hub.HfApi, "parse_safetensors_file_metadata", parse)
    evidence.prefetch_for_pick(NOOB, NOOB_FILE, None, None)
    assert during == ["sdxl"]
    # A failed refresh keeps the last good read rather than forgetting it.
    assert _name(detect_family_for_pick(NOOB, NOOB_FILE)) == "sdxl"


@pytest.mark.parametrize("given, sent", [(False, False), ("  hf_x  ", "hf_x"), ("", None)])
def test_the_pipeline_marker_probe_keeps_the_anonymous_sentinel(tmp_path, monkeypatch, given, sent):
    import huggingface_hub

    root = _pipeline_dir(tmp_path, NOOB_SCHEDULER, {})
    seen = []

    def files(
        self,
        repo_id,
        token = None,
        **kw,
    ):
        seen.append(token)
        return [NOOB_FILE]

    def parse(
        self,
        repo_id,
        filename,
        token = None,
        **kw,
    ):
        seen.append(token)
        return types.SimpleNamespace(tensors = {"v_pred": None}, metadata = {})

    monkeypatch.setattr(huggingface_hub.HfApi, "list_repo_files", files)
    monkeypatch.setattr(huggingface_hub.HfApi, "parse_safetensors_file_metadata", parse)
    pipeline_prediction_overrides(NOOB_SCHEDULER, str(root), NOOB, given, True)
    assert seen == [sent, sent]


def test_the_load_hands_the_uncollapsed_token_to_the_marker_probe():
    import ast

    source = (
        Path(__file__).resolve().parents[1] / "core" / "inference" / "diffusion.py"
    ).read_text(encoding = "utf-8")
    tree = ast.parse(source)
    methods = {
        n.name: n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name in ("begin_load", "load_pipeline")
    }
    # begin_load forwards normalize_token(hf_token) to the worker before collapsing hf_token.
    begin = methods["begin_load"]
    assigns = [
        a
        for a in ast.walk(begin)
        if isinstance(a, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "hub_token" for t in a.targets)
    ]
    assert assigns and isinstance(assigns[0].value, ast.Call)
    assert getattr(assigns[0].value.func, "id", None) == "normalize_token"
    assert any(isinstance(k, ast.keyword) and k.arg == "_hub_token" for k in ast.walk(begin))
    # load_pipeline accepts it and the marker probe receives it, never the collapsed hf_token.
    load = methods["load_pipeline"]
    assert "_hub_token" in {a.arg for a in load.args.kwonlyargs}
    probe = [
        c
        for c in ast.walk(load)
        if isinstance(c, ast.Call)
        and getattr(c.func, "id", None) == "apply_pipeline_prediction_type"
    ]
    assert len(probe) == 1
    token_kw = next(k for k in probe[0].keywords if k.arg == "hf_token")
    assert isinstance(token_kw.value, ast.Name) and token_kw.value.id == "hub_token"


def test_the_marker_probe_reads_the_snapshot_it_records(hub_cache, monkeypatch):
    import huggingface_hub

    snap = _staged_snapshot(hub_cache)
    revisions = []

    def files(
        self,
        repo_id,
        token = None,
        **kw,
    ):
        revisions.append(kw.get("revision"))
        return ["model_index.json", NOOB_FILE]

    def parse(
        self,
        repo_id,
        filename,
        token = None,
        **kw,
    ):
        revisions.append(kw.get("revision"))
        keys = SDXL_UNET_KEYS + ["v_pred", "ztsnr"]
        return types.SimpleNamespace(tensors = {k: None for k in keys}, metadata = {})

    monkeypatch.setattr(huggingface_hub.HfApi, "list_repo_files", files)
    monkeypatch.setattr(huggingface_hub.HfApi, "parse_safetensors_file_metadata", parse)
    # No revision pin: both Hub reads still target the staged commit, not the moving HEAD.
    assert pipeline_prediction_overrides(NOOB_SCHEDULER, str(snap), NOOB, None, True) == VPRED
    assert revisions == [SHA, SHA]


@pytest.mark.parametrize("markers", [5, [["v_pred"]], "v_pred", [1]])
def test_a_malformed_recorded_entry_is_a_cache_miss(hub_cache, monkeypatch, markers):
    from core.inference.diffusion_sampler import _marks_key, _marks_record_path

    snap = _staged_snapshot(hub_cache)
    path = _marks_record_path()
    path.parent.mkdir(parents = True, exist_ok = True)
    path.write_text(json.dumps({_marks_key(NOOB, SHA): [{"markers": markers}]}))
    assert pipeline_prediction_overrides(NOOB_SCHEDULER, str(snap), NOOB, None, False) == {}
    _stub_noob_hub(monkeypatch, None, [])
    assert pipeline_prediction_overrides(NOOB_SCHEDULER, str(snap), NOOB, None, True) == {}


def test_concurrent_marker_records_keep_every_entry(hub_cache, monkeypatch):
    import os
    import threading

    import core.inference.diffusion_sampler as sampler

    real_replace = os.replace
    first_read = threading.Event()
    release = threading.Event()
    calls = []

    def slow_replace(src, dst):
        # The first writer has read the record; hold it until the second writer has had its turn.
        if not calls:
            calls.append(1)
            first_read.set()
            release.wait(5)
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", slow_replace)
    a = threading.Thread(target = sampler._record_marks, args = (NOOB, "a" * 40, [["v_pred"]]))
    a.start()
    assert first_read.wait(5)
    b = threading.Thread(target = sampler._record_marks, args = (NOOB, "b" * 40, [["v_pred"]]))
    b.start()
    b.join(1)
    release.set()
    a.join(5)
    b.join(5)
    record = json.loads(sampler._marks_record_path().read_text(encoding = "utf-8"))
    assert {sampler._marks_key(NOOB, "a" * 40), sampler._marks_key(NOOB, "b" * 40)} <= set(record)
