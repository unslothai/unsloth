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
    # The size HEAD the index fetch makes first: a small index unless a test says otherwise.
    monkeypatch.setattr(
        huggingface_hub, "get_hf_file_metadata", lambda *a, **k: types.SimpleNamespace(size = 400)
    )
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


def test_vpred_diffusers_repo_detected_from_cached_model_index(hub_cache):
    assert detect_family_for_pick(NOOB) is None  # nothing cached: the name says nothing
    hub_cache.put(NOOB, "model_index.json", payload = SDXL_INDEX)
    assert _name(detect_family_for_pick(NOOB)) == "sdxl"
    # The packaged root single file of the same repo is vouched for by that index too.
    assert _name(detect_family_for_pick(NOOB, NOOB_FILE)) == "sdxl"


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


def test_fetch_falls_back_to_the_index_for_a_pipeline_pick(hub_cache, monkeypatch):
    import huggingface_hub

    fetched = []

    def download(
        repo_id,
        filename,
        token = None,
        cache_dir = None,
        **kw,
    ):
        fetched.append((repo_id, filename, cache_dir))
        return str(hub_cache.put(repo_id, filename, payload = SDXL_INDEX))

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    evidence.prefetch_for_pick(NOOB, None, None, "hf_x")
    assert fetched == [(NOOB, "model_index.json", str(hub_cache.root))]
    assert _name(detect_family_for_pick(NOOB)) == "sdxl"


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


def test_name_veto_is_not_overridden_by_the_index(hub_cache):
    # The name matched SDXL and then refused the edit variant; a cached index must not undo that.
    repo = "someone/sdxl-edit-ft"
    hub_cache.put(repo, "model_index.json", payload = SDXL_INDEX)
    assert detect_family_for_pick(repo) is None


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


# ---- review follow-ups -------------------------------------------------------------------------

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


def test_the_index_fetch_records_the_credential_it_used(hub_cache, monkeypatch):
    import contextlib

    import huggingface_hub

    import hub.utils.hf_tokens as hf_tokens

    events = []

    @contextlib.contextmanager
    def recording(
        token,
        repo_id,
        repo_type = "model",
    ):
        events.append(("enter", token, repo_id, repo_type))
        yield
        events.append(("exit",))

    def download(
        repo_id,
        filename,
        token = None,
        cache_dir = None,
        **kw,
    ):
        events.append(("download", repo_id, filename))
        return str(hub_cache.put(repo_id, filename, payload = SDXL_INDEX))

    monkeypatch.setattr(hf_tokens, "recording_a_request_token_fetch", recording)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    evidence.prefetch_for_pick(NOOB, None, None, "hf_private")
    assert events == [
        ("enter", "hf_private", NOOB, "model"),
        ("download", NOOB, "model_index.json"),
        ("exit",),
    ]
    assert _name(detect_family_for_pick(NOOB)) == "sdxl"


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

    source = (Path(__file__).resolve().parents[1] / "routes" / "inference.py").read_text()
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


# ---- review round 2 ----------------------------------------------------------------------------

FLUX1_DEV_KEYS = [
    "transformer_blocks.0.attn.to_q.weight",
    "single_transformer_blocks.0.attn.to_q.weight",
    "x_embedder.weight",
    "context_embedder.weight",
    "time_text_embed.timestep_embedder.linear_1.weight",
    "time_text_embed.guidance_embedder.linear_1.weight",
]


@pytest.mark.parametrize("size", [50_000_000, None])
def test_an_oversized_or_unsized_index_is_never_downloaded(hub_cache, monkeypatch, size):
    import huggingface_hub

    monkeypatch.setattr(
        huggingface_hub, "get_hf_file_metadata", lambda *a, **k: types.SimpleNamespace(size = size)
    )
    monkeypatch.setattr(
        huggingface_hub, "hf_hub_download", lambda *a, **k: pytest.fail("sized past the cap")
    )
    evidence.prefetch_for_pick(NOOB, None, None, None)
    assert detect_family_for_pick(NOOB) is None


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
