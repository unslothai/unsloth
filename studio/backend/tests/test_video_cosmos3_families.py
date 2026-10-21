# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Cosmos3-Nano and Cosmos3-Edge video families: detection by repo id, HF-cache snapshot path and folder name, and of a
renamed local folder by its own configs; the named refusals for the Cosmos3 builds the family names also match
(Reasoner, Policy, Sim, Transfer, FP8); the model cards' defaults, frame windows and canvases; the catalog, training and
diffusers-version gates; and a table proving every repo the other video families already claimed keeps its family.

Hermetic: no torch, no network, no weights. Hub listings and configs come from fixtures/cosmos3_checkpoints.json (public
Hub metadata at pinned revisions); folders are written under tmp_path. "Nano card" and "Edge card" line numbers refer to
README.md of nvidia/Cosmos3-Nano@e59a53c2 and nvidia/Cosmos3-Edge@344d602b.

Cosmos3-Edge on the Video page: detection by repo, folder and file name, the model card's sampling recipe (UniPC at
flow shift 12 without Karras sigmas, the prompt templates off, 832x480, 121 frames at 24 fps, 20 steps, CFG 6), the
omni pipeline's shape (no text encoder, the first frame through its own image=), the checkpoint's guardrail choice, and
load validation for the hub id, the cached snapshot and a renamed local folder. A table proves every repo the other
video families already claimed keeps its family. The Cosmos3 repos that are not generators (the Reasoners, the Policy /
Sim / Transfer and FP8 builds) detect no family, so neither the HF-cache nor the local listing offers them as video, and
the load still names the reason. Cosmos3 loads only as its pipeline, so no listing offers a Cosmos3-named checkpoint
without one (a reasoner fine-tune, a quant, a GGUF, a loose file) as video either.

The recipe is nvidia/Cosmos3-Edge's README (revision 344d602b), diffusers example and serving examples."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Optional
import pytest
from core.inference import diffusion_content as dc
from core.inference import video_families as vf

_FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures" / "cosmos3_checkpoints.json").read_text(encoding = "utf-8")
)

_NANO_SNAPSHOT = (
    "/home/me/.cache/huggingface/hub/models--nvidia--Cosmos3-Nano/snapshots/"
    "e59a53c25979a090fa8706c9acc0c254a6e89b92"
)
_EDGE_SNAPSHOT = (
    "/home/me/.cache/huggingface/hub/models--nvidia--Cosmos3-Edge/snapshots/"
    "344d602b128d1bbdacb43b08d0a3626f46343e29"
)
_REASONER = "Cosmos3 Reasoner checkpoint"
_CONDITIONED = "Policy, Sim, Transfer or FP8 build"


def _checkpoint(repo: str) -> dict:
    """The fixture entry of ``repo`` (keys are ``<repo>@<revision>``)."""
    return next(entry for key, entry in _FIXTURE.items() if key.split("@")[0] == repo)


def _revision(repo: str) -> str:
    return next(key.split("@")[1] for key in _FIXTURE if key.split("@")[0] == repo)


def _family(name: str) -> vf.VideoFamily:
    fam = vf.detect_video_family("", override = name)
    assert fam is not None, f"no video family named {name}"
    return fam


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents = True, exist_ok = True)
    path.write_text(json.dumps(payload), encoding = "utf-8")


def _write_configs(folder: Path, repo: str) -> Path:
    """The three files a local Cosmos3 folder is recognised by, as ``repo`` ships them."""
    entry = _checkpoint(repo)
    _write_json(folder / "model_index.json", entry["model_index"])
    _write_json(folder / "config.json", entry["config"])
    _write_json(folder / "transformer" / "config.json", entry["transformer_config"])
    return folder


def _folder_gb(listing: list, prefix: str) -> float:
    return sum(size for path, size in listing if path.startswith(prefix)) / 1e9


# ── detection ────────────────────────────────────────────────────────────────

# (identifier, family on this branch). Rows above the marker are what main already detected and must not move; the
# Cosmos3 rows below it all detected None on main.
_TABLE = [
    ("Wan-AI/Wan2.2-TI2V-5B-Diffusers", "wan2.2-ti2v-5b"),
    ("unsloth/Wan2.2-TI2V-5B-GGUF", "wan2.2-ti2v-5b"),
    ("Wan-AI/Wan2.2-T2V-A14B-Diffusers", "wan2.2-t2v-a14b"),
    ("Wan-AI/Wan2.1-T2V-1.3B-Diffusers", "wan2.1-t2v-1.3b"),
    ("Wan-AI/Wan2.1-T2V-14B-Diffusers", "wan2.1-t2v-14b"),
    ("Lightricks/LTX-2", "ltx-2"),
    ("unsloth/LTX-2.3-GGUF", "ltx-2"),
    ("Lightricks/LTX-2.5-Diffusers", "ltx-2.5"),
    ("hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v", "hunyuanvideo-1.5"),
    ("hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_t2v", "hunyuanvideo-1.5-720p"),
    ("MiniMaxAI/MiniMax-H3", "minimax-h3"),
    ("unsloth/MiniMax-H3-GGUF", "minimax-h3"),
    ("tencent/HunyuanVideo", None),
    ("Wan-AI/Wan2.2-I2V-A14B-Diffusers", None),
    # ---- Cosmos3 (None on main for every row) ----
    ("nvidia/Cosmos3-Nano", "cosmos3-nano"),
    ("nvidia/Cosmos3-Edge", "cosmos3-edge"),
    ("nvidia/cosmos3-nano", "cosmos3-nano"),
    ("NVIDIA/COSMOS3-EDGE", "cosmos3-edge"),
    (_NANO_SNAPSHOT, "cosmos3-nano"),
    (_EDGE_SNAPSHOT, "cosmos3-edge"),
    (
        "C:\\Users\\me\\.cache\\huggingface\\hub\\models--nvidia--Cosmos3-Edge\\snapshots\\"
        "344d602b128d1bbdacb43b08d0a3626f46343e29",
        "cosmos3-edge",
    ),
    ("/data/models/Cosmos3-Nano", "cosmos3-nano"),
    ("/data/models/Cosmos3-Edge/", "cosmos3-edge"),
    ("~/models/cosmos3_nano", "cosmos3-nano"),
    # Super (hidden_size 5120) and its distilled builds have no family; neither do the other Cosmos generations, the
    # Cosmos-Predict2 GGUFs, or a folder that names no size.
    ("nvidia/Cosmos3-Super", None),
    ("nvidia/Cosmos3-Super-Image2Video", None),
    ("nvidia/Cosmos3-Super-Image2Video-4Step", None),
    ("nvidia/Cosmos3-Super-Text2Image", None),
    ("nvidia/Cosmos3-Super-Text2Image-4Step", None),
    ("nvidia/Cosmos-Reason1-7B", None),
    ("nvidia/Cosmos-Predict2-2B-Video2World", None),
    ("calcuis/cosmos-predict2-gguf", None),
    ("calcuis/cosmos-predict2-gguf/cosmos-predict2-q4_0.gguf", None),
    ("/data/models/Cosmos3", None),
]


@pytest.mark.parametrize("identifier, expected", _TABLE)
def test_cosmos3_families_detection_table(identifier, expected):
    fam = vf.detect_video_family(identifier)
    assert (fam.name if fam else None) == expected


@pytest.mark.parametrize("identifier, expected", [row for row in _TABLE if row[1] is not None])
def test_cosmos3_detected_names_carry_no_refusal(identifier, expected):
    fam = vf.detect_video_family(identifier)
    assert vf.unsupported_video_checkpoint_reason(identifier) is None
    assert vf.video_checkpoint_refusal(fam, identifier) is None


@pytest.mark.parametrize(
    "override, expected",
    [
        ("cosmos3-nano", "cosmos3-nano"),
        ("cosmos3_edge", "cosmos3-edge"),
        ("Cosmos3 Nano", "cosmos3-nano"),
        ("COSMOS3.EDGE", "cosmos3-edge"),
        (" cosmos3-edge ", "cosmos3-edge"),
        # A bare token would claim Super, the Reasoners and the Cosmos-Predict2 GGUF arch alike.
        ("cosmos", None),
        ("cosmos3", None),
        ("cosmos3-super", None),
        # A transformers model_type is not a family name.
        ("cosmos3_omni", None),
        ("cosmos3-nano-reasoner", None),
    ],
)
def test_cosmos3_family_override_matches_names_separator_insensitively(override, expected):
    fam = vf.detect_video_family("x", override = override)
    assert (fam.name if fam else None) == expected


def test_cosmos3_invariant_no_family_answers_to_a_bare_cosmos_token():
    from core.inference.family_name_match import normalize_family_name
    for fam in vf._FAMILIES:
        for token in (fam.name, *fam.aliases):
            assert normalize_family_name(token).strip("-") not in ("cosmos", "cosmos3"), (
                fam.name,
                token,
            )


def test_cosmos3_rows_leave_the_cosmos_gguf_arch_unrunnable():
    """llama.cpp's ``cosmos`` GGUF architecture is Cosmos-Predict2's, which no page runs: the catalog tags it
    image-diffusion-unsupported, and llama_cpp.py's _UNRUNNABLE_MEDIA_ARCHES refusal names no page because no video
    family answers to that arch. A Cosmos-named repo or a Cosmos3 name among the hints must change neither."""
    from hub.services.models import catalog_classification as classification

    # The loader resolves a renamed GGUF by its general.architecture through exactly this override
    # (video.py _resolve_load_family_with_source).
    assert vf.detect_video_family("calcuis/cosmos-predict2-gguf", override = "cosmos") is None
    for hints in (
        ("calcuis/cosmos-predict2-gguf", "cosmos-predict2-q4_0.gguf"),
        ("nvidia/Cosmos3-Nano",),
    ):
        assert (
            classification._arch_to_task("cosmos", hints)
            == classification._UNSUPPORTED_DIFFUSION_TASK
        )


@pytest.mark.parametrize(
    "identifier",
    [
        "nvidia/Cosmos3-Super",
        "nvidia/Cosmos3-Super-Image2Video",
        "nvidia/Cosmos3-Super-Image2Video-4Step",
        "nvidia/Cosmos3-Super-Text2Image",
        "nvidia/Cosmos3-Super-Text2Image-4Step",
        "nvidia/Cosmos-Reason1-7B",
        "nvidia/Cosmos-Predict2-2B-Video2World",
    ],
)
def test_cosmos3_siblings_without_a_family_get_the_unsupported_model_answer(identifier):
    from core.inference.video import VideoBackend

    with pytest.raises(ValueError) as excinfo:
        VideoBackend().validate_load_request(identifier)
    message = str(excinfo.value)
    assert "is not a supported text-to-video model" in message
    # The answer lists what does load, the two Cosmos3 rows included.
    assert "cosmos3-nano, cosmos3-edge" in message


# ── named refusals ───────────────────────────────────────────────────────────

# The family names also match sibling builds that are not text/image-to-video generators. The Reasoners answer in text;
# Policy, Sim and Transfer need action or control inputs and their own pipelines; both cards say "Only BF16 precision
# is tested" (Nano card :168, Edge card :156). Of these builds only the Policy-DROID repos are public; the other rows
# use neutral ids.
_REFUSED = [
    ("nvidia/Cosmos3-Nano-Policy-DROID", _CONDITIONED),
    ("nvidia/Cosmos3-Edge-Policy-DROID", _CONDITIONED),
    ("/models/Cosmos3-Nano-Reasoner/", _REASONER),
    ("someone/Cosmos3-Edge-Reasoner", _REASONER),
    ("someone/cosmos3_nano_reasoner", _REASONER),
    ("/data/cosmos3-nano-sim", _CONDITIONED),
    ("someone/Cosmos3-Edge-Sim", _CONDITIONED),
    ("someone/Cosmos3-Nano-Transfer", _CONDITIONED),
    ("someone/Cosmos3-Edge-FP8", _CONDITIONED),
    ("/data/COSMOS3.NANO.FP8", _CONDITIONED),
]


@pytest.mark.parametrize("identifier, reason", _REFUSED)
def test_cosmos3_non_generator_builds_are_refused_by_name(identifier, reason):
    from core.inference.video import VideoBackend

    # Whichever size the name resolves to, or is forced to, the refusal holds. Today both rows carry it
    # (unsupported_checkpoints); the shared table detect_video_family consults (unsupported_video_checkpoint_reason,
    # where the Wan2.1 refusals live) would serve as well, so either place counts.
    for name in ("cosmos3-nano", "cosmos3-edge"):
        message = vf.video_checkpoint_refusal(_family(name), identifier)
        message = message or vf.unsupported_video_checkpoint_reason(identifier)
        assert message is not None and reason in message
        # Not a dead end: the refusal names the two checkpoints that do load.
        assert "nvidia/Cosmos3-Nano" in message and "nvidia/Cosmos3-Edge" in message
    with pytest.raises(ValueError, match = reason):
        VideoBackend().validate_load_request(identifier)


@pytest.mark.parametrize(
    "filename, kind, reason",
    [
        ("Cosmos3-Edge-FP8.safetensors", "single_file", _CONDITIONED),
        ("cosmos3-nano-reasoner-Q4_K_M.gguf", "gguf", _REASONER),
    ],
)
def test_cosmos3_refused_build_picked_as_a_file_is_refused(filename, kind, reason):
    # The repo id names no family; the picked file does, and it is judged on its own name.
    from core.inference.video import VideoBackend
    with pytest.raises(ValueError, match = reason):
        VideoBackend().validate_load_request(
            "someone/cosmos3-files", gguf_filename = filename, model_kind = kind
        )


def test_cosmos3_refusal_reads_the_display_id_of_an_opaque_snapshot():
    # A local snapshot's last segment is a commit hash; the logical Hub id travels as display_repo_id
    # (models/inference.py), and the refusal must read it.
    from core.inference.video import VideoBackend
    snapshot = (
        "/cache/hub/models--nvidia--Cosmos3-Nano-Policy-DROID/snapshots/"
        "805c0d6d46196ecdc789213a6de72303c383895b"
    )
    with pytest.raises(ValueError, match = _CONDITIONED):
        VideoBackend().validate_load_request(
            snapshot, display_repo_id = "nvidia/Cosmos3-Nano-Policy-DROID"
        )


@pytest.mark.parametrize(
    "identifier, expected",
    [
        ("/home/me/cosmos3-nano-reasoner/models/Cosmos3-Nano", "cosmos3-nano"),
        ("cosmos3-edge-fp8-lab/Cosmos3-Edge", "cosmos3-edge"),
        ("/data/cosmos3-nano-sim/Cosmos3-Nano/", "cosmos3-nano"),
        ("/srv/cosmos3-edge-transfer/exports/Cosmos3-Edge", "cosmos3-edge"),
        ("/data/policy/Cosmos3-Edge", "cosmos3-edge"),
        # A marker is a whole word of the model name.
        ("/data/Cosmos3-Nano-simple", "cosmos3-nano"),
        ("/data/Cosmos3-Edge-transferred", "cosmos3-edge"),
    ],
)
def test_cosmos3_markers_outside_the_model_name_do_not_refuse(identifier, expected):
    fam = vf.detect_video_family(identifier)
    assert fam is not None and fam.name == expected
    assert vf.video_checkpoint_refusal(fam, identifier) is None


@pytest.mark.parametrize("override", ["cosmos3-nano", "cosmos3-edge"])
@pytest.mark.parametrize(
    "identifier, reason",
    [
        ("nvidia/Cosmos3-Nano-Policy-DROID", _CONDITIONED),
        ("/models/Cosmos3-Nano-Reasoner/", _REASONER),
        ("someone/Cosmos3-Edge-FP8", _CONDITIONED),
    ],
)
def test_cosmos3_family_override_does_not_bypass_the_refusal(identifier, reason, override):
    from core.inference.video import VideoBackend
    with pytest.raises(ValueError, match = reason):
        VideoBackend().validate_load_request(identifier, family_override = override)


# ── a local folder answers by its configs ───────────────────────────────────


@pytest.mark.parametrize(
    "repo, expected",
    [
        ("nvidia/Cosmos3-Nano", "cosmos3-nano"),
        ("nvidia/Cosmos3-Edge", "cosmos3-edge"),
        # hidden_size 5120: no family serves Super.
        ("nvidia/Cosmos3-Super", None),
        # Edge-sized (2048) under Nano's model_type cosmos3_omni: neither family's transformer.
        ("nvidia/Cosmos3-Edge-Policy-DROID", None),
    ],
)
def test_cosmos3_local_folder_resolves_by_its_shipped_configs(tmp_path, repo, expected):
    folder = _write_configs(tmp_path / "renamed", repo)
    assert vf.local_cosmos3_pipeline_family(str(folder)) == expected


# Edge's index names the released Cosmos3OmniPipeline; Nano's, saved by diffusers 0.37.1, the pre-release
# Cosmos3OmniDiffusersPipeline (fixture model_index). The root config's model_type and the transformer's hidden_size
# then pick the size: Nano is cosmos3_omni at 4096, Edge cosmos3_edge at 2048 (fixture config / transformer_config).
_SERVED = {("cosmos3_omni", 4096): "cosmos3-nano", ("cosmos3_edge", 2048): "cosmos3-edge"}


@pytest.mark.parametrize("hidden", [4096, 2048, 5120])
@pytest.mark.parametrize("model_type", ["cosmos3_omni", "cosmos3_edge", "qwen3_vl"])
@pytest.mark.parametrize(
    "index_class", ["Cosmos3OmniPipeline", "Cosmos3OmniDiffusersPipeline", "WanPipeline"]
)
def test_cosmos3_local_folder_matrix_of_index_class_model_type_and_size(
    tmp_path, index_class, model_type, hidden
):
    # Nano's shipped configs with the three deciding values swapped in.
    entry, folder = _checkpoint("nvidia/Cosmos3-Nano"), tmp_path / "renamed"
    _write_json(folder / "model_index.json", {**entry["model_index"], "_class_name": index_class})
    _write_json(folder / "config.json", {**entry["config"], "model_type": model_type})
    _write_json(
        folder / "transformer" / "config.json",
        {**entry["transformer_config"], "hidden_size": hidden},
    )
    expected = _SERVED.get((model_type, hidden)) if index_class != "WanPipeline" else None
    assert vf.local_cosmos3_pipeline_family(str(folder)) == expected


@pytest.mark.parametrize(
    "broken, text",
    [
        ("model_index.json", None),
        ("transformer/config.json", None),
        ("model_index.json", "{not json"),
        ("config.json", "{not json"),
        ("transformer/config.json", "{not json"),
        ("model_index.json", "[]"),
        ("config.json", '"cosmos3_omni"'),
        ("transformer/config.json", "4096"),
    ],
)
def test_cosmos3_local_folder_that_cannot_be_read_is_not_claimed(tmp_path, broken, text):
    folder = _write_configs(tmp_path / "renamed", "nvidia/Cosmos3-Nano")
    assert vf.local_cosmos3_pipeline_family(str(folder)) == "cosmos3-nano"
    if text is None:
        (folder / broken).unlink()
    else:
        (folder / broken).write_text(text, encoding = "utf-8")
    assert vf.local_cosmos3_pipeline_family(str(folder)) is None


def test_cosmos3_local_folder_without_a_root_config_resolves_by_its_transformer(tmp_path):
    """A diffusers save_pretrained export has no transformers config.json at the root; its transformer's class and
    hidden_size still name the family."""
    folder = _write_configs(tmp_path / "renamed", "nvidia/Cosmos3-Nano")
    (folder / "config.json").unlink()
    assert vf.local_cosmos3_pipeline_family(str(folder)) == "cosmos3-nano"


def test_cosmos3_local_folder_lookup_reads_only_a_real_folder(tmp_path):
    assert vf.local_cosmos3_pipeline_family(None) is None
    assert vf.local_cosmos3_pipeline_family("") is None
    assert vf.local_cosmos3_pipeline_family(str(tmp_path / "missing")) is None
    folder = _write_configs(tmp_path / "renamed", "nvidia/Cosmos3-Edge")
    assert vf.local_cosmos3_pipeline_family(str(folder / "model_index.json")) is None
    # A Hub id is not a folder: nothing is fetched to answer it.
    assert vf.local_cosmos3_pipeline_family("nvidia/Cosmos3-Edge") is None


@pytest.mark.parametrize(
    "repo, expected",
    [
        ("nvidia/Cosmos3-Nano", "cosmos3-nano"),
        ("nvidia/Cosmos3-Edge", "cosmos3-edge"),
        ("nvidia/Cosmos3-Super", None),
    ],
)
def test_cosmos3_renamed_local_folder_loads_as_its_family(tmp_path, repo, expected):
    from core.inference.video import _detect_load_family

    folder = _write_configs(tmp_path / "my-render-model", repo)
    # The name alone claims nothing, so the configs decide.
    assert vf.detect_video_family(str(folder)) is None
    fam = _detect_load_family(str(folder), None, None)
    assert (fam.name if fam else None) == expected


# ── catalog: which picker lists a checkpoint ─────────────────────────────────


@pytest.mark.parametrize(
    "repo, hf_cache, task",
    [
        ("nvidia/Cosmos3-Nano", False, "text-to-video"),
        ("nvidia/Cosmos3-Edge", False, "text-to-video"),
        ("nvidia/Cosmos3-Edge", True, "text-to-video"),
        ("nvidia/Cosmos3-Super", False, None),
    ],
)
def test_cosmos3_local_checkout_lands_on_the_video_picker(tmp_path, repo, hf_cache, task):
    from hub.services.models import catalog_classification as classification
    from models.models import LocalModelInfo

    if hf_cache:
        snapshots = tmp_path / "hub" / f"models--{repo.replace('/', '--')}" / "snapshots"
        folder = snapshots / _revision(repo)
    else:
        folder = tmp_path / "my-render-model"
    _write_configs(folder, repo)
    model = LocalModelInfo(
        id = str(folder),
        display_name = folder.name,
        path = str(folder),
        source = "hf_cache" if hf_cache else "models_dir",
    )
    assert classification._local_model_task(model) == task


def _cached_pipeline_repo(repo_id: str) -> SimpleNamespace:
    """A cached diffusers repo (root model_index.json beside transformer shards), as the HF cache scan reports one."""
    files = [
        SimpleNamespace(file_name = "model_index.json"),
        SimpleNamespace(file_name = "diffusion_pytorch_model-00001-of-00002.safetensors"),
    ]
    return SimpleNamespace(
        repo_id = repo_id, repo_path = "/nonexistent", revisions = [SimpleNamespace(files = files)]
    )


@pytest.mark.parametrize(
    "repo_id, task",
    [
        ("nvidia/Cosmos3-Nano", "text-to-video"),
        ("nvidia/Cosmos3-Edge", "text-to-video"),
        # A full pipeline loads only from the official base repos (the load path's trust rule), so a cached
        # Policy-DROID snapshot or a community repack is never offered: its pick would be refused.
        ("nvidia/Cosmos3-Nano-Policy-DROID", None),
        ("someone/cosmos3-nano-repack", None),
        # No family serves Super.
        ("nvidia/Cosmos3-Super", None),
    ],
)
def test_cosmos3_cached_pipeline_reaches_the_video_picker_only_from_a_base_repo(repo_id, task):
    from hub.services.models import catalog_classification as classification
    assert classification._cached_repo_task(_cached_pipeline_repo(repo_id)) == task


def _cached_repo(repo_id: str) -> SimpleNamespace:
    """A cached transformers-style repo (weight shards, no model_index.json), as the HF cache scan reports one."""
    files = [SimpleNamespace(file_name = "model-00001-of-00002.safetensors")]
    return SimpleNamespace(
        repo_id = repo_id, repo_path = "/nonexistent", revisions = [SimpleNamespace(files = files)]
    )


def _local_vlm_folder(folder: Path) -> Path:
    # A Cosmos3 Reasoner's own config is not public; a generic VLM config stands in. The catalog decides by name.
    _write_json(
        folder / "config.json",
        {"model_type": "qwen3_vl", "architectures": ["Qwen3VLForConditionalGeneration"]},
    )
    (folder / "model.safetensors").write_bytes(b"\0")
    return folder


def test_cosmos3_reason1_stays_a_chat_model_in_the_catalog(tmp_path):
    from hub.services.models import catalog_classification as classification
    from models.models import LocalModelInfo

    info = _cached_repo("nvidia/Cosmos-Reason1-7B")
    assert classification._repo_is_diffusers(info) is False
    assert classification._cached_repo_task(info) is None
    folder = _local_vlm_folder(tmp_path / "Cosmos-Reason1-7B")
    model = LocalModelInfo(
        id = str(folder), display_name = folder.name, path = str(folder), source = "models_dir"
    )
    assert classification._local_is_diffusers(model) is False
    assert classification._local_model_task(model) is None


# The Reasoner refusal says "Load it from Chat" (video_families.py _COSMOS3_REASONER_REFUSAL). The catalog asks
# detect_video_family, which claims a Reasoner for its size's family, and never the refusal: a cached or local Reasoner
# is then flagged diffusers and tagged text-to-video, and the Chat picker routes a text-to-video row to the Video page
# (frontend pickers.tsx mediaPageForTask), whose load refuses it and points back to Chat.
def test_cosmos3_cached_reasoner_repo_is_not_flagged_as_diffusers():
    from hub.services.models import catalog_classification as classification
    info = SimpleNamespace(repo_id = "someone/Cosmos3-Nano-Reasoner", repo_path = "/nonexistent")
    assert classification._repo_is_diffusers(info) is False


def test_cosmos3_cached_reasoner_snapshot_is_not_tagged_text_to_video():
    from hub.services.models import catalog_classification as classification
    assert classification._cached_repo_task(_cached_repo("someone/Cosmos3-Edge-Reasoner")) is None


def test_cosmos3_local_reasoner_folder_is_not_tagged_text_to_video(tmp_path):
    from hub.services.models import catalog_classification as classification
    from models.models import LocalModelInfo

    folder = _local_vlm_folder(tmp_path / "Cosmos3-Nano-Reasoner")
    model = LocalModelInfo(
        id = str(folder), display_name = folder.name, path = str(folder), source = "models_dir"
    )
    assert classification._local_is_diffusers(model) is False
    assert classification._local_model_task(model) is None


# ── fields and defaults, against the model cards ─────────────────────────────

# Nano card: :417-423 (vLLM-Omni text-to-video: 1280x720, 189 frames, fps 24, 35 steps, guidance 6.0, flow_shift 10.0),
# :887-896 (Diffusers: UniPCMultistepScheduler.from_config(..., flow_shift=10.0), the same settings) and :113 ("durations
# from 5 to 400 frames"). Edge card: :494-500 (vLLM-Omni image-to-video: 832x480, 121 frames, fps 24, 20 steps, guidance
# 6.0, flow_shift 12.0), :759-772 (Diffusers: flow_shift=12.0, use_karras_sigmas=False) and :102 ("50-150 frames").
# The windows are the 4k+1 counts inside those ranges. Edge's Diffusers call also turns off the resolution and duration
# sentences Cosmos3OmniPipeline appends to the prompt (Edge card :774-775: add_resolution_template=False,
# add_duration_template=False); Nano's (Nano card :889-897) passes neither, so both keep the pipeline's default, True
# (pipeline_cosmos3_omni.py:1346-1347).
_CARD = {
    "cosmos3-nano": {
        "base_repo": "nvidia/Cosmos3-Nano",
        "size": (1280, 720),
        "frames": 189,
        "fps": 24,
        "steps": 35,
        "guidance": 6.0,
        "scheduler": (("flow_shift", 10.0),),
        "window": (5, 397),
        "call_kwargs": (),
    },
    "cosmos3-edge": {
        "base_repo": "nvidia/Cosmos3-Edge",
        "size": (832, 480),
        "frames": 121,
        "fps": 24,
        "steps": 20,
        "guidance": 6.0,
        "scheduler": (("flow_shift", 12.0), ("use_karras_sigmas", False)),
        "window": (53, 149),
        "call_kwargs": (("add_resolution_template", False), ("add_duration_template", False)),
    },
}


@pytest.mark.parametrize("name", ["cosmos3-nano", "cosmos3-edge"])
def test_cosmos3_rows_carry_the_model_card_sampling(name):
    fam, card = _family(name), _CARD[name]
    assert (fam.pipeline_class, fam.transformer_class) == (
        "Cosmos3OmniPipeline",
        "Cosmos3OmniTransformer",
    )
    assert fam.base_repo == card["base_repo"]
    # The family selector names the rows from the frontend LABELS (family-override.ts). VideoFamily.label is the
    # Train UI's, which the other generation-only rows leave unset, so these rows may drop it, but never carry
    # another model's name.
    assert fam.label in (None, card["base_repo"].split("/")[1])
    assert (fam.default_steps, fam.default_guidance) == (card["steps"], card["guidance"])
    assert (fam.default_num_frames, fam.default_fps) == (card["frames"], card["fps"])
    assert fam.resolution_presets[0] == card["size"]
    assert fam.scheduler_overrides == card["scheduler"]
    assert fam.call_kwargs == card["call_kwargs"]
    assert (fam.min_num_frames, fam.max_num_frames) == card["window"]
    assert fam.cfg_kwarg == "guidance_scale" and fam.supports_cfg
    # The repos also ship a modular_model_index.json; Studio loads the conventional pipeline.
    assert fam.modular_workflow is None
    # Video only: the load passes sound_tokenizer=None (video_cosmos3.pipeline_kwargs).
    assert fam.has_audio is False
    assert fam.supports_references is False
    assert fam.trainable is False
    # Regional compile stays off until measured: the DiT is a language model over a prompt-length sequence.
    assert fam.supports_torch_compile is False


@pytest.mark.parametrize(
    "name, presets, card_clip",
    [
        # 7.875 s at 24 fps is the card's 189 frames.
        ("cosmos3-nano", (2.0, 5.0, 7.875), 7.875),
        # 5 s at 24 fps lands on the card's 121 frames.
        ("cosmos3-edge", (3.0, 5.0, 6.0), 5.0),
    ],
)
def test_cosmos3_duration_presets_reach_the_card_default_clip(name, presets, card_clip):
    from routes.video import _frames_for_seconds

    fam = _family(name)
    assert fam.duration_presets == presets
    defaults = {
        "fps": fam.default_fps,
        "frame_step": fam.frame_step,
        "frame_offset": fam.frame_offset,
    }
    assert _frames_for_seconds(card_clip, defaults) == fam.default_num_frames


@pytest.mark.parametrize(
    "name, repo", [("cosmos3-nano", "nvidia/Cosmos3-Nano"), ("cosmos3-edge", "nvidia/Cosmos3-Edge")]
)
def test_cosmos3_lattice_and_grid_follow_the_shipped_vae_config(name, repo):
    # Both repos ship the Wan2.2-TI2V-5B VAE (AutoencoderKLWan, z_dim 48). Cosmos3OmniPipeline only requires sizes on
    # its spatial factor (pipeline_cosmos3_omni.py:1066-1068): 16, so a true 1280x720 runs. Time compresses by 4, so
    # frame counts are 4k+1.
    vae = _checkpoint(repo)["vae_config"]
    fam = _family(name)
    assert (vae["_class_name"], vae["z_dim"]) == ("AutoencoderKLWan", 48)
    assert (fam.frame_step, fam.frame_offset) == (vae["scale_factor_temporal"], 1)
    assert fam.resolution_multiple == vae["scale_factor_spatial"]
    assert all(w % 16 == 0 and h % 16 == 0 for w, h in fam.resolution_presets)


def test_cosmos3_vae_untiled_tables_follow_the_wan22_ti2v_vae():
    from core.inference import video_vae_untiled as vu
    for name, repo in (
        ("cosmos3-nano", "nvidia/Cosmos3-Nano"),
        ("cosmos3-edge", "nvidia/Cosmos3-Edge"),
    ):
        vae = _checkpoint(repo)["vae_config"]
        assert vu._BYTES_PER_LATENT_PIXEL[name] == vu._BYTES_PER_LATENT_PIXEL["wan2.2-ti2v-5b"]
        assert vu._OUTPUT_SCALE[name] == (vae["scale_factor_spatial"], vae["scale_factor_temporal"])


@pytest.mark.parametrize(
    "name, repo", [("cosmos3-nano", "nvidia/Cosmos3-Nano"), ("cosmos3-edge", "nvidia/Cosmos3-Edge")]
)
def test_cosmos3_scheduler_overrides_name_keys_the_shipped_scheduler_has(name, repo):
    # Both repos ship UniPC at flow_shift 1.0 with Karras sigmas on. Each card rebuilds it from that config with its
    # own values (Nano card :887 flow_shift=10.0; Edge card :759-760 flow_shift=12.0, use_karras_sigmas=False), so
    # every override must name a key the shipped config has and change it; a key it lacks is a typo from_config
    # ignores.
    shipped = _checkpoint(repo)["scheduler_config"]
    assert shipped["_class_name"] == "UniPCMultistepScheduler"
    for key, value in _family(name).scheduler_overrides:
        assert key in shipped and shipped[key] != value, (key, shipped.get(key), value)


# The canvases Cosmos3OmniPipeline itself snaps action-conditioned inputs to (diffusers d42ebf9,
# pipelines/cosmos/pipeline_cosmos3_omni.py:187-215, _ACTION_RESOLUTION_BINS; text- and image-to-video only require
# /16 sizes): tier "720" and tier "480" at 1:1, 3:4, 4:3, 9:16 and 16:9, the aspect ratios both cards list (Nano card
# :93-94, Edge card :86). Edge renders 256p and 480p only (Edge card :102).
_TIER_720 = ((1280, 720), (720, 1280), (960, 960), (1104, 832), (832, 1104))
_TIER_480 = ((832, 480), (480, 832), (640, 640), (736, 544), (544, 736))
_CARD_ASPECTS = (16 / 9, 4 / 3, 1.0, 3 / 4, 9 / 16)


@pytest.mark.parametrize(
    "name, presets",
    [("cosmos3-nano", _TIER_720 + ((832, 480), (480, 832))), ("cosmos3-edge", _TIER_480)],
)
def test_cosmos3_presets_are_the_pipeline_canvases_at_card_aspects(name, presets):
    assert _family(name).resolution_presets == presets
    for w, h in presets:
        assert min(abs(math.log(w / h / aspect)) for aspect in _CARD_ASPECTS) < 0.03, (w, h)


@pytest.mark.parametrize(
    "name, ids",
    [
        ("cosmos3-nano", ("nvidia/Cosmos3-Nano",)),
        ("cosmos3-nano", (None, None, None, _NANO_SNAPSHOT)),
        ("cosmos3-edge", (None, None, "nvidia/Cosmos3-Edge", "nvidia/Cosmos3-Edge")),
        ("cosmos3-edge", ("/data/models/Cosmos3-Edge",)),
    ],
)
def test_cosmos3_generation_defaults_fall_back_to_the_card(name, ids):
    # No other model's _VIDEO_GENERATION_DEFAULTS key names a Cosmos3 checkpoint, so the backend's own call (the
    # family's defaults as the fallback, video.py status and generate) lands on the card's steps and CFG, not the
    # generic LTX (40, 4.0). The table's own Cosmos3 keys (name-only, first) give the same numbers.
    fam, card = _family(name), _CARD[name]
    assert vf.video_generation_variant(*ids) in (None, name)
    assert vf.default_video_generation_params(
        *ids, fallback = (fam.default_steps, fam.default_guidance), family = fam.name
    ) == (card["steps"], card["guidance"])


def _studio_generation_defaults(
    fam: vf.VideoFamily,
    repo_id: str,
    display_repo_id: Optional[str] = None,
) -> set:
    """(steps, guidance) as video.py asks for them on a pipeline load of ``repo_id``: the load's step-cache plan passes
    (variant_id, gguf_filename, repo_id, base), status and generate pass (variant_id, gguf_filename, display_repo_id,
    repo_id, base_repo), all with the family's own defaults as the fallback. A Cosmos3 load is a pipeline load: no
    variant id, no GGUF file, and its base is the selected model itself (load_pipeline: base = repo_id). A downloaded
    snapshot carries its Hub id as display_repo_id; a plain local folder has none."""
    own = {"fallback": (fam.default_steps, fam.default_guidance), "family": fam.name}
    return {
        vf.default_video_generation_params(None, None, repo_id, repo_id, **own),
        vf.default_video_generation_params(None, None, display_repo_id, repo_id, repo_id, **own),
    }


# The ltx, distilled, hunyuanvideo and a14b keys match anywhere in the path (only the Wan, FusionX and Cosmos3 keys are
# read from the model name), so the Cosmos3 rows sit first: a parent folder spelling one of those keys must not give a
# Cosmos3 load that model's steps and CFG. A snapshot path's Cosmos3 name is read from its models--<owner>--<name> folder.


@pytest.mark.parametrize(
    "name, folder, display_repo_id",
    [
        # Read from the model name only, so these already fall back to the card.
        ("cosmos3-edge", "/data/fusionx/Cosmos3-Edge", None),
        ("cosmos3-nano", "/models/wan/Cosmos3-Nano", None),
        ("cosmos3-nano", "/models/ltx/Cosmos3-Nano", None),
        ("cosmos3-nano", "/home/distilled/Cosmos3-Nano", None),
        # Nano: hunyuanvideo's (20, 6.0) happens to be Edge's card.
        ("cosmos3-nano", "/data/hunyuanvideo/Cosmos3-Nano", None),
        ("cosmos3-edge", "/srv/a14b/Cosmos3-Edge", None),
        # A downloaded snapshot loads by its path with the Hub id as display_repo_id (the picker's on-device pick).
        # The load's step-cache plan reads only the path, whose last segment is a commit hash, so there the cache root
        # decides: 8 steps, under auto FBCache's 20-step floor (diffusion_cache.FBCACHE_MIN_STEPS). Cosmos3 rows
        # read from the name alone leave that call as it is; a fix that hands the plan the display id must update
        # _studio_generation_defaults with it.
        pytest.param(
            "cosmos3-nano",
            _NANO_SNAPSHOT.replace("/home/me/.cache/huggingface", "/data/distilled"),
            "nvidia/Cosmos3-Nano",
        ),
    ],
)
def test_cosmos3_generation_defaults_ignore_a_folder_named_after_another_model(
    name, folder, display_repo_id
):
    fam, card = _family(name), _CARD[name]
    # The load resolves the folder to the family by its own name...
    assert vf.detect_video_family(folder) is fam
    # ...so every caller must land on that family's card.
    assert _studio_generation_defaults(fam, folder, display_repo_id) == {
        (card["steps"], card["guidance"])
    }


@pytest.mark.parametrize(
    "ids, expected",
    [
        (("/models/Cosmos3-Nano/wan2.1-t2v-14b-Q4_K_M.gguf",), (20, 6.0)),
        (("/home/cosmos3-nano/models/Wan2.1-T2V-1.3B-Diffusers",), (30, 6.0)),
        (("/srv/cosmos3_nano/HunyuanVideo-1.5-Diffusers-480p_t2v",), (20, 6.0)),
        (("/models/Cosmos3-Edge/wan2.2-ti2v-5b-Q4_K_M.gguf",), (20, 5.0)),
        (("/data/cosmos3-edge/LTX-2",), (40, 4.0)),
    ],
)
def test_cosmos3_folder_names_leave_other_models_defaults_alone(ids, expected):
    # The other side of the folder rule: a Cosmos3 key, once the table has one, counts only in the model name, so
    # another model kept under a Cosmos3-named folder keeps its own recipe.
    assert vf.default_video_generation_params(*ids) == expected


@pytest.mark.parametrize(
    "name, repo", [("cosmos3-nano", "nvidia/Cosmos3-Nano"), ("cosmos3-edge", "nvidia/Cosmos3-Edge")]
)
def test_cosmos3_size_table_matches_the_hub_folder_sizes(name, repo):
    listing = _checkpoint(repo)["listing"]
    transformer_gb, text_encoder_gb, vae_gb = _family(name).bf16_components_gb
    # The weights ship in bf16 (Nano's transformer/config.json: "dtype": "bfloat16"; both cards test BF16 only, Nano
    # card :168, Edge card :156), so the resident size is the download size: within one 0.1 GB rounding step of the Hub
    # folder sums.
    assert abs(transformer_gb - _folder_gb(listing, "transformer/")) < 0.1
    assert abs(vae_gb - _folder_gb(listing, "vae/")) < 0.1
    # The omni transformer is its own text model: no text_encoder/, and text_tokenizer/ holds no weights.
    assert text_encoder_gb == 0.0
    assert not any(path.startswith("text_encoder/") for path, _size in listing)
    assert not any(
        path.startswith("text_tokenizer/") and path.endswith((".safetensors", ".bin"))
        for path, _size in listing
    )


# The check above only asks for the nearest 0.1 GB step, which a mis-rounded term passes. test_diffusion_component_sizes
# holds every family's table to TABLE_ROUNDING_GB of its tensor headers, but only where the checkpoint sits in the local
# HF cache; the fixture listings give the same sums (the folders add a few kB of safetensors headers and configs), so
# this holds the Cosmos3 terms to that bound on every host.
@pytest.mark.parametrize(
    "name, repo, folder, term",
    [
        ("cosmos3-nano", "nvidia/Cosmos3-Nano", "transformer/", 0),
        ("cosmos3-nano", "nvidia/Cosmos3-Nano", "vae/", 2),
        ("cosmos3-edge", "nvidia/Cosmos3-Edge", "transformer/", 0),
        ("cosmos3-edge", "nvidia/Cosmos3-Edge", "vae/", 2),
    ],
)
def test_cosmos3_size_table_terms_round_the_hub_folder_sums(name, repo, folder, term):
    from core.inference.diffusion_component_sizes import TABLE_ROUNDING_GB

    listing = _checkpoint(repo)["listing"]
    table_gb, folder_gb = _family(name).bf16_components_gb[term], _folder_gb(listing, folder)
    assert abs(table_gb - folder_gb) <= TABLE_ROUNDING_GB, (table_gb, folder_gb)


def test_cosmos3_rows_follow_the_existing_families_on_the_video_page():
    existing = (
        "minimax-h3",
        "ltx-2.5",
        "ltx-2",
        "wan2.2-ti2v-5b",
        "wan2.2-t2v-a14b",
        "wan2.1-t2v-1.3b",
        "wan2.1-t2v-14b",
        "hunyuanvideo-1.5",
        "hunyuanvideo-1.5-720p",
    )
    names = vf.supported_video_family_names()
    # Appended, so every existing family keeps its place in the 400 text, /video/status and the override selector.
    assert names[: len(existing) + 2] == existing + ("cosmos3-nano", "cosmos3-edge")
    # video.py branches on this tuple (guardrail, from_pretrained kwargs, fps kwarg, .video output, first frame).
    assert vf.COSMOS3_FAMILIES == ("cosmos3-nano", "cosmos3-edge")
    for name in vf.COSMOS3_FAMILIES:
        assert dc.family_page(name) == dc.PAGE_VIDEO


# ── request shapes ───────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "name, width, height, frames",
    [
        ("cosmos3-nano", 1280, 720, 189),
        ("cosmos3-nano", 720, 1280, 189),
        ("cosmos3-nano", 1280, 720, 5),
        ("cosmos3-nano", 1280, 720, 397),
        ("cosmos3-edge", 832, 480, 121),
        ("cosmos3-edge", 480, 832, 121),
        ("cosmos3-edge", 832, 480, 53),
        ("cosmos3-edge", 832, 480, 149),
    ],
)
def test_cosmos3_accepts_the_model_card_shapes(name, width, height, frames):
    vf.validate_video_request_shape(_family(name), width, height, frames)


@pytest.mark.parametrize(
    "name, shape, fragments",
    [
        ("cosmos3-nano", {"num_frames": 188}, ("k * 4 + 1", "185 and 189")),
        # The card's own upper bound sits between lattice points; only the lower one is loadable.
        ("cosmos3-nano", {"num_frames": 400}, ("k * 4 + 1", "nearest supported count is 397")),
        ("cosmos3-nano", {"num_frames": 401}, ("5 to 397",)),
        # num_frames=1 switches Cosmos3OmniPipeline to text-to-image (pipeline_cosmos3_omni.py:1358).
        ("cosmos3-nano", {"num_frames": 1}, ("5 to 397",)),
        ("cosmos3-edge", {"num_frames": 50}, ("nearest supported count is 53",)),
        ("cosmos3-edge", {"num_frames": 150}, ("nearest supported count is 149",)),
        # Nano's default clip is past Edge's window.
        ("cosmos3-edge", {"num_frames": 189}, ("53 to 149",)),
        ("cosmos3-nano", {"width": 1000, "height": 700}, ("multiples of 16",)),
        ("cosmos3-edge", {"width": 840, "height": 480}, ("multiples of 16",)),
        # Nano's 720p canvas is past Edge's trained area (its card offers 256p and 480p only).
        ("cosmos3-edge", {"width": 1280, "height": 720}, ("409,600 pixels",)),
        ("cosmos3-edge", {"width": 1280, "height": 304}, ("aspect ratio",)),
    ],
)
def test_cosmos3_refuses_shapes_off_its_lattice_window_or_grid(name, shape, fragments):
    with pytest.raises(vf.VideoShapeError) as excinfo:
        vf.validate_video_request_shape(_family(name), **shape)
    message = str(excinfo.value)
    assert name in message
    for fragment in fragments:
        assert fragment in message, message


@pytest.mark.parametrize(
    "name, requested, snapped",
    [
        ("cosmos3-nano", 189, 189),
        ("cosmos3-nano", 188, 185),
        ("cosmos3-nano", 1, 5),
        ("cosmos3-nano", 400, 397),
        ("cosmos3-nano", 1000, 397),
        ("cosmos3-edge", 121, 121),
        ("cosmos3-edge", 10, 53),
        ("cosmos3-edge", 50, 53),
        ("cosmos3-edge", 150, 149),
        ("cosmos3-edge", 189, 149),
    ],
)
def test_cosmos3_frame_snapping_stays_inside_the_card_window(name, requested, snapped):
    fam = _family(name)
    assert vf.snap_num_frames(fam, requested) == snapped
    vf.validate_video_request_shape(fam, num_frames = snapped)


@pytest.mark.parametrize("fam", vf._FAMILIES, ids = lambda f: f.name)
def test_cosmos3_invariant_every_family_window_ends_on_its_lattice(fam):
    """Added with Cosmos3, whose cards give 5-400 and 50-150 frames: 400, 50 and 150 are off the 4k+1 lattice. A window
    end declared off the lattice makes snap_num_frames return a count the gate refuses, and the refusal name a range
    whose ends it refuses."""
    step, offset = max(1, fam.frame_step), max(1, fam.frame_offset)
    floor = max(offset, fam.min_num_frames)
    ends = (floor,) if fam.max_num_frames is None else (floor, fam.max_num_frames)
    for end in ends:
        assert (end - offset) % step == 0, (fam.name, end)
        vf.validate_video_request_shape(fam, num_frames = end)
    assert vf.snap_num_frames(fam, 1) == floor


@pytest.mark.parametrize("fam", vf._FAMILIES, ids = lambda f: f.name)
def test_cosmos3_invariant_every_duration_preset_is_a_valid_count(fam):
    """The Video page builds its duration select from duration_presets at the family fps without clamping to the frame
    window (frontend video-page.tsx durationOptions), and /v1/videos maps seconds the same way (routes/video.py
    _frames_for_seconds), so a preset outside the window offers a clip the shape gate answers with a 422. A narrow
    window such as Cosmos3-Edge's 53..149 cannot keep the generic (1, 2, 3, 5) s default."""
    from routes.video import _frames_for_seconds

    step, offset = max(1, fam.frame_step), max(1, fam.frame_offset)
    defaults = {
        "fps": fam.default_fps,
        "frame_step": fam.frame_step,
        "frame_offset": fam.frame_offset,
    }
    for seconds in fam.duration_presets:
        route = _frames_for_seconds(seconds, defaults)
        # The page rounds half up (Math.round); Python's round() goes to even. Both must land in the window.
        page = max(1, math.floor((seconds * fam.default_fps - offset) / step + 0.5)) * step + offset
        for frames in {route, page}:
            vf.validate_video_request_shape(fam, num_frames = frames)


# ── first frame ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize("name", ["cosmos3-nano", "cosmos3-edge"])
def test_cosmos3_takes_a_first_frame_through_its_own_pipeline(name):
    from core.inference.video import _keyframe_canvas

    fam = _family(name)
    # Cosmos3OmniPipeline.__call__ takes the first frame itself (image=, pipeline_cosmos3_omni.py:1322), so its "i2v
    # class" is the t2v class; there is no last-frame or reference input.
    assert fam.i2v_pipeline_class == fam.pipeline_class == "Cosmos3OmniPipeline"
    assert fam.i2v_pipeline_class in vf.I2V_IN_T2V_PIPELINE_CLASSES
    assert vf.video_family_accepts_first_frame(fam)
    assert not vf.video_family_accepts_last_frame(fam)
    assert vf.video_family_uses_i2v_pipeline(fam)
    vf.validate_video_keyframe_conditioning(fam, None, has_keyframes = True)
    with pytest.raises(ValueError, match = "first frame only"):
        vf.validate_video_keyframe_conditioning(fam, None, has_keyframes = True, has_last_frame = True)
    with pytest.raises(ValueError, match = "takes no reference"):
        vf.validate_video_reference_conditioning(fam, None, has_references = True)
    # The canvas comes from the presets (status canvas_from_presets), not MiniMax-H3's "match source" rule.
    assert _keyframe_canvas(fam) == (None, None)


@pytest.mark.parametrize(
    "name, source, canvas",
    [
        ("cosmos3-nano", (1920, 1080), (1280, 720)),
        ("cosmos3-nano", (1080, 1920), (720, 1280)),
        ("cosmos3-nano", (1024, 1024), (960, 960)),
        ("cosmos3-nano", (1600, 1200), (1104, 832)),
        ("cosmos3-nano", (1200, 1600), (832, 1104)),
        ("cosmos3-edge", (1920, 1080), (832, 480)),
        ("cosmos3-edge", (1080, 1920), (480, 832)),
        ("cosmos3-edge", (1024, 1024), (640, 640)),
        ("cosmos3-edge", (1600, 1200), (736, 544)),
        ("cosmos3-edge", (1200, 1600), (544, 736)),
    ],
)
def test_cosmos3_first_frame_canvas_is_the_nearest_preset(name, source, canvas):
    assert vf.first_frame_canvas(_family(name), *source) == canvas


# ── training and the diffusers gate ──────────────────────────────────────────


@pytest.mark.parametrize(
    "base, model_family, family",
    [
        ("nvidia/Cosmos3-Nano", None, "cosmos3-nano"),
        ("nvidia/Cosmos3-Edge", None, "cosmos3-edge"),
        ("/data/models/Cosmos3-Nano", None, "cosmos3-nano"),
        (_EDGE_SNAPSHOT, None, "cosmos3-edge"),
        ("nvidia/Cosmos3-Nano", "cosmos3-nano", "cosmos3-nano"),
        ("/data/checkouts/opaque", "Cosmos3 Edge", "cosmos3-edge"),
    ],
)
def test_cosmos3_is_generation_only_and_training_refuses_it_by_name(base, model_family, family):
    from core.training.diffusion_train_common import (
        TRAINABLE_VIDEO_FAMILIES,
        resolve_trainable_family,
    )

    assert _family(family).trainable is False and family not in TRAINABLE_VIDEO_FAMILIES
    # Before the by-name gate a video base fell through to the SDXL trainer, which failed inside from_pretrained.
    with pytest.raises(ValueError) as excinfo:
        resolve_trainable_family(base, model_family)
    message = str(excinfo.value)
    assert family in message and "video" in message.lower()
    # Not a dead end: the refusal names the video families that do train.
    assert "ltx-2" in message


@pytest.mark.parametrize(
    "repo, family",
    [("nvidia/Cosmos3-Nano", "cosmos3-nano"), ("nvidia/Cosmos3-Edge", "cosmos3-edge")],
)
def test_cosmos3_renamed_checkout_is_refused_for_training(tmp_path, repo, family):
    from core.training.diffusion_train_common import resolve_trainable_family

    folder = _write_configs(tmp_path / "my-video-base", repo)
    # The load path and the catalog resolve this folder by its configs (local_cosmos3_pipeline_family); training must
    # agree, as it refuses a local LTX-2.5 folder by its model_index.json (assert_local_pipeline_not_ltx25).
    assert vf.local_cosmos3_pipeline_family(str(folder)) == family
    with pytest.raises(ValueError, match = family):
        resolve_trainable_family(str(folder))


def test_cosmos3_pipeline_gate_is_the_release_reading_edge_index_flags():
    from core.inference.diffusion_families import pipeline_class_requirement

    # Cosmos3OmniPipeline is exported from 0.39.0, but Edge's index (saved by a 0.40 dev build) sets
    # use_native_flow_schedule, a pipeline config 0.40.0 added; 0.39.0 drops it and samples Edge on the wrong
    # schedule. 0.40.0 is past diffusers' Python 3.9 cutoff (0.37.0), hence the second field.
    assert pipeline_class_requirement("Cosmos3OmniPipeline") == ("0.40.0", True)
    edge_index = _checkpoint("nvidia/Cosmos3-Edge")["model_index"]
    assert edge_index["_class_name"] == "Cosmos3OmniPipeline"
    assert edge_index["use_native_flow_schedule"] is True
    assert edge_index["_diffusers_version"].startswith("0.40.")


@pytest.mark.parametrize(
    "installed, available", [("0.39.0", False), ("0.40.0", True), ("0.41.0.dev0", True)]
)
def test_cosmos3_rows_hide_on_a_diffusers_older_than_the_gate(monkeypatch, installed, available):
    from core.inference import diffusion_families as df

    # The listing judges installed metadata until diffusers is imported; pretend it is not.
    monkeypatch.delitem(sys.modules, "diffusers", raising = False)
    monkeypatch.setattr(df, "_installed_diffusers_version", lambda: installed)
    for name in ("cosmos3-nano", "cosmos3-edge"):
        assert df.family_pipeline_available(_family(name)) is available


# ── Cosmos3-Edge path ──

_EDGE = "cosmos3-edge"
_NANO = "cosmos3-nano"
# The registered refusals, word for word (absent on main, where this file fails only its Cosmos3 rows).
_EDGE_REASONER = getattr(vf, "_COSMOS3_REASONER_REFUSAL", None)
_EDGE_CONDITIONED = getattr(vf, "_COSMOS3_CONDITIONED_REFUSAL", None)

# (identifier, family on this branch). Rows above the marker are what main already detected and must not move.
_EDGE_TABLE = [
    ("Wan-AI/Wan2.2-TI2V-5B-Diffusers", "wan2.2-ti2v-5b"),
    ("Wan-AI/Wan2.2-T2V-A14B-Diffusers", "wan2.2-t2v-a14b"),
    ("Wan-AI/Wan2.1-T2V-1.3B-Diffusers", "wan2.1-t2v-1.3b"),
    ("Lightricks/LTX-2", "ltx-2"),
    ("unsloth/LTX-2.3-GGUF", "ltx-2"),
    ("hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v", "hunyuanvideo-1.5"),
    ("MiniMaxAI/MiniMax-H3", "minimax-h3"),
    ("nvidia/Cosmos-Predict2-2B-Video2World", None),
    ("nvidia/Cosmos3-Super", None),
    # --- Cosmos3-Edge: no family on main
    ("nvidia/Cosmos3-Edge", _EDGE),
    ("Cosmos3-Edge", _EDGE),
    ("cosmos3_edge", _EDGE),
    ("/models/cosmos3-edge/", _EDGE),
    ("edge-lab/Cosmos3-Edge", _EDGE),
    # A variant marker in an owner or folder name decides nothing: only the model's own name is judged.
    ("reasoner-lab/Cosmos3-Edge", _EDGE),
    ("/models/cosmos3-edge-policy/Cosmos3-Edge/", _EDGE),
    ("nvidia/Cosmos3-Nano", _NANO),
    # --- Cosmos3 repos that are not generators: no family, as on main, so no listing offers them as video
    ("nvidia/Cosmos3-Edge-Reasoner", None),
    ("nvidia/Cosmos3-Nano-Reasoner", None),
    ("unsloth/Cosmos3-Nano-Reasoner", None),
    ("unsloth/Cosmos3-Nano-Reasoner-GGUF", None),
    ("cosmos3-nano-reasoner-Q4_K_M.gguf", None),
    ("nvidia/Cosmos3-Edge-Policy-DROID", None),
    ("nvidia/Cosmos3-Nano-Policy-DROID", None),
    ("nvidia/Cosmos3-Nano-Sim-Transfer", None),
    ("nvidia/Cosmos3-Nano-FP8", None),
    ("/models/cosmos3_edge_fp8/", None),
]


def _fam(name = _EDGE):
    return vf.detect_video_family("", override = name)


def _skip_without_the_pipeline():
    """Validation refuses a family whose pipeline the installed diffusers lacks (it stays silent when diffusers is not
    importable at all, as on Backend CI), so a host with a diffusers older than Cosmos3OmniPipeline skips the accepts."""
    try:
        import diffusers
    except Exception:
        return
    if not hasattr(diffusers, "Cosmos3OmniPipeline"):
        pytest.skip("the installed diffusers has no Cosmos3OmniPipeline")


@pytest.mark.parametrize("identifier, expected", _EDGE_TABLE)
def test_detection_table(identifier, expected):
    fam = vf.detect_video_family(identifier)
    assert (fam.name if fam else None) == expected


@pytest.mark.parametrize(
    "identifier, reason",
    [
        ("nvidia/Cosmos3-Edge-Reasoner", _EDGE_REASONER),
        ("nvidia/Cosmos3-Nano-Reasoner", _EDGE_REASONER),
        ("unsloth/Cosmos3-Nano-Reasoner", _EDGE_REASONER),
        ("unsloth/Cosmos3-Nano-Reasoner-GGUF", _EDGE_REASONER),
        ("cosmos3-nano-reasoner-Q4_K_M.gguf", _EDGE_REASONER),
        ("nvidia/Cosmos3-Edge-Policy-DROID", _EDGE_CONDITIONED),
        ("nvidia/Cosmos3-Nano-Policy-DROID", _EDGE_CONDITIONED),
        ("nvidia/Cosmos3-Nano-Sim-Transfer", _EDGE_CONDITIONED),
        ("nvidia/Cosmos3-Nano-FP8", _EDGE_CONDITIONED),
        ("/models/cosmos3_edge_fp8/", _EDGE_CONDITIONED),
    ],
)
def test_non_generator_siblings_name_their_reason(identifier, reason):
    """The family tokens match these names too. Refused by name, they resolve to no family, so a cached Reasoner is a
    Chat row again instead of a video row the Chat picker sends to the Video page, which refuses it."""
    assert vf.unsupported_video_checkpoint_reason(identifier) == reason
    assert vf.detect_video_family(identifier) is None


@pytest.mark.parametrize("identifier, expected", [row for row in _EDGE_TABLE if row[1] is not None])
def test_supported_names_carry_no_refusal(identifier, expected):
    assert vf.unsupported_video_checkpoint_reason(identifier) is None


def test_edge_samples_with_its_model_cards_recipe():
    fam = _fam()
    assert (fam.pipeline_class, fam.transformer_class) == (
        "Cosmos3OmniPipeline",
        "Cosmos3OmniTransformer",
    )
    assert fam.base_repo == "nvidia/Cosmos3-Edge"
    assert not fam.is_moe and fam.transformer2_class is None
    assert (fam.default_steps, fam.default_guidance) == (20, 6.0)
    assert (fam.default_num_frames, fam.default_fps, fam.frame_step) == (121, 24, 4)
    assert (fam.min_num_frames, fam.max_num_frames) == (53, 149)
    assert fam.resolution_presets == ((832, 480), (480, 832), (640, 640), (736, 544), (544, 736))
    assert fam.resolution_multiple == 16 and fam.duration_presets == (3.0, 5.0, 6.0)
    assert dict(fam.scheduler_overrides) == {"flow_shift": 12.0, "use_karras_sigmas": False}
    assert dict(fam.call_kwargs) == {
        "add_resolution_template": False,
        "add_duration_template": False,
    }
    assert fam.bf16_components_gb == (6.7, 0.0, 1.4)
    assert not fam.has_audio and not fam.supports_torch_compile
    # Same Wan2.2 VAE as TI2V-5B, so the same reproducibility policy: no per-process cuDNN conv pick.
    assert fam.cudnn_benchmark is False
    assert fam.cudnn_benchmark == _fam("wan2.2-ti2v-5b").cudnn_benchmark
    assert vf.detect_video_family("", override = "cosmos3_edge").name == _EDGE
    vf.validate_video_request_shape(fam, 832, 480, 121)
    with pytest.raises(vf.VideoShapeError):
        vf.validate_video_request_shape(fam, 832, 480, 120)


def test_the_omni_transformer_is_the_text_model():
    """No text encoder to price, fetch or quantize: the pipeline only tokenizes, and the denoiser reads the prompt."""
    from core.inference import diffusion_component_sizes as dcs

    fam = _fam()
    assert dcs.video_family_has_no_text_encoder(fam)
    assert "text_encoder" not in dcs.video_required_roles(fam)
    assert not fam.te_quant_auto


def test_the_first_frame_goes_through_the_same_pipeline():
    fam = _fam()
    assert fam.i2v_pipeline_class == fam.pipeline_class
    assert fam.i2v_pipeline_class in vf.I2V_IN_T2V_PIPELINE_CLASSES


@pytest.mark.parametrize(
    "ids",
    [
        ("nvidia/Cosmos3-Edge",),
        (None, "nvidia/Cosmos3-Edge"),
        ("/models/cosmos3-edge/",),
    ],
)
def test_generation_defaults(ids):
    # As video.py asks: the family's own recipe is the fallback no variant key overrides.
    fam = _fam()
    assert vf.default_video_generation_params(
        *ids, fallback = (fam.default_steps, fam.default_guidance), family = fam.name
    ) == (20, 6.0)


def test_registry_lists_the_cosmos3_families_last():
    names = vf.supported_video_family_names()
    assert names[-2:] == ("cosmos3-nano", _EDGE)
    assert names[-4:-2] == ("hunyuanvideo-1.5", "hunyuanvideo-1.5-720p")
    for name in names[-2:]:
        assert dc.family_page(name) == dc.PAGE_VIDEO


@pytest.mark.parametrize(
    "index, opted_out, installed, refused",
    [
        ({"enable_safety_checker": False}, False, False, False),  # Edge ships it off
        ({"enable_safety_checker": True}, False, False, True),
        ({"enable_safety_checker": True}, True, False, False),
        ({"enable_safety_checker": True}, False, True, False),
        (None, False, False, False),  # not on disk yet: the base repo's shipped value
    ],
)
def test_the_checkpoint_decides_the_guardrail(monkeypatch, index, opted_out, installed, refused):
    from core.inference import video_cosmos3 as vc

    monkeypatch.setenv(vc.SAFETY_CHECKER_ENV, "0" if opted_out else "")
    monkeypatch.setattr(vc, "guardrail_installed", lambda: installed)
    monkeypatch.setattr(vc, "_cached_model_index", lambda repo_id: None)
    assert bool(vc.guardrail_refusal("nvidia/Cosmos3-Edge", index)) is refused


def _snapshot(root: Path, *, root_config: bool = True) -> Path:
    """A local Cosmos3-Edge pipeline folder: the configs that name it, every component of its model_index. Without
    ``root_config``, a diffusers ``save_pretrained`` export, which writes no root config.json."""
    files = {
        "model_index.json": {
            "_class_name": "Cosmos3OmniPipeline",
            "scheduler": ["diffusers", "UniPCMultistepScheduler"],
            "text_tokenizer": ["transformers", "PreTrainedTokenizerFast"],
            "transformer": ["diffusers", "Cosmos3OmniTransformer"],
            "vae": ["diffusers", "AutoencoderKLWan"],
            "safety_checker": [None, None],
            "sound_tokenizer": [None, None],
            "enable_safety_checker": False,
        },
        "config.json": {"model_type": "cosmos3_edge"},
        "transformer/config.json": {"_class_name": "Cosmos3OmniTransformer", "hidden_size": 2048},
        "vae/config.json": {"_class_name": "AutoencoderKLWan"},
        "scheduler/scheduler_config.json": {"_class_name": "UniPCMultistepScheduler"},
        "text_tokenizer/tokenizer_config.json": {},
        "text_tokenizer/tokenizer.json": {},
    }
    if not root_config:
        del files["config.json"]
    for name, payload in files.items():
        (root / name).parent.mkdir(parents = True, exist_ok = True)
        (root / name).write_text(json.dumps(payload), encoding = "utf-8")
    for name in (
        "transformer/diffusion_pytorch_model.safetensors",
        "vae/diffusion_pytorch_model.safetensors",
    ):
        (root / name).write_bytes(b"weights")
    return root


@pytest.mark.parametrize(
    "folder, display_repo_id, root_config",
    [
        ("Cosmos3-Edge", None, True),
        (
            "344d602b128d1bbdacb43b08d0a3626f46343e29",
            "nvidia/Cosmos3-Edge",
            True,
        ),  # the HF-cache snapshot, by display id
        ("my-video-model", None, True),  # renamed: its configs name it
        ("edge-export", None, False),  # renamed save_pretrained export: its transformer names it
    ],
)
def test_load_validation_accepts_a_local_edge_pipeline(
    tmp_path, folder, display_repo_id, root_config
):
    from core.inference.video import VideoBackend

    _skip_without_the_pipeline()
    path = _snapshot(tmp_path / folder, root_config = root_config)
    fam = VideoBackend().validate_load_request(
        str(path), model_kind = "pipeline", display_repo_id = display_repo_id
    )
    assert fam.name == _EDGE


def test_load_validation_accepts_the_hub_id():
    from core.inference.video import VideoBackend
    _skip_without_the_pipeline()
    assert (
        VideoBackend().validate_load_request("nvidia/Cosmos3-Edge", model_kind = "pipeline").name
        == _EDGE
    )


def test_an_unknown_model_keeps_the_generic_refusal():
    from core.inference.video import VideoBackend
    with pytest.raises(ValueError, match = "not a supported text-to-video model"):
        VideoBackend().validate_load_request("someorg/some-image-model")


@pytest.mark.parametrize(
    "repo, kwargs, refusal",
    [
        ("nvidia/Cosmos3-Edge-Reasoner", {}, _EDGE_REASONER),
        ("unsloth/Cosmos3-Nano-Reasoner", {"model_kind": "pipeline"}, _EDGE_REASONER),
        (
            "unsloth/Cosmos3-Nano-Reasoner-GGUF",
            {"gguf_filename": "cosmos3-nano-reasoner-Q4_K_M.gguf", "model_kind": "gguf"},
            _EDGE_REASONER,
        ),
        # A family override cannot make refused weights runnable, in their own lineage or another one.
        ("nvidia/Cosmos3-Nano-Reasoner", {"family_override": _NANO}, _EDGE_REASONER),
        (
            "unsloth/Cosmos3-Edge-Policy-DROID",
            {"family_override": "wan2.2-ti2v-5b", "model_kind": "pipeline"},
            _EDGE_CONDITIONED,
        ),
        ("nvidia/Cosmos3-Nano-Reasoner", {"family_override": "hunyuanvideo-1.5"}, _EDGE_REASONER),
        # The display id and the picked file are judged too, not only the repo that names a family.
        (
            "nvidia/Cosmos3-Edge",
            {"display_repo_id": "nvidia/Cosmos3-Edge-Reasoner", "model_kind": "pipeline"},
            _EDGE_REASONER,
        ),
        (
            "nvidia/Cosmos3-Edge",
            {"gguf_filename": "cosmos3-edge-fp8.safetensors", "model_kind": "single_file"},
            _EDGE_CONDITIONED,
        ),
        ("nvidia/Cosmos3-Edge-Policy-DROID", {}, _EDGE_CONDITIONED),
        ("nvidia/Cosmos3-Nano-Sim-Transfer", {}, _EDGE_CONDITIONED),
        ("nvidia/Cosmos3-Nano-FP8", {"model_kind": "pipeline"}, _EDGE_CONDITIONED),
    ],
)
def test_load_validation_still_names_the_reason(repo, kwargs, refusal):
    """The load raises the registered refusal word for word, before any eviction and ahead of the generic "not a
    supported text-to-video model" and the untrusted-repo refusals."""
    from core.inference.video import VideoBackend

    with pytest.raises(ValueError) as excinfo:
        VideoBackend().validate_load_request(repo, **kwargs)
    assert str(excinfo.value) == refusal


def _edge_cached_repo(repo_id: str, files: tuple[str, ...]) -> SimpleNamespace:
    """An HF-cache scan row as the inventory hands it to the listing classifier."""
    return SimpleNamespace(
        repo_id = repo_id,
        repo_type = "model",
        repo_path = Path(f"/hf-cache/models--{repo_id.replace('/', '--')}"),
        revisions = [
            SimpleNamespace(
                files = [
                    SimpleNamespace(file_name = name, size_on_disk = 1, blob_path = None)
                    for name in files
                ]
            )
        ],
    )


_PIPELINE_FILES = (
    "model_index.json",
    "transformer/diffusion_pytorch_model.safetensors",
    "vae/diffusion_pytorch_model.safetensors",
)
# The Reasoner / Policy / FP8 layout: a root config.json and root shards, no model_index.json.
_ROOT_SHARD_FILES = (
    "config.json",
    "model.safetensors.index.json",
    "model-00001-of-00002.safetensors",
    "model-00002-of-00002.safetensors",
)


@pytest.mark.parametrize(
    "repo_id, files, video",
    [
        ("nvidia/Cosmos3-Edge", _PIPELINE_FILES, True),
        ("nvidia/Cosmos3-Nano", _PIPELINE_FILES, True),
        ("nvidia/Cosmos3-Edge-Reasoner", _ROOT_SHARD_FILES, False),
        ("nvidia/Cosmos3-Nano-Reasoner", _ROOT_SHARD_FILES, False),
        ("unsloth/Cosmos3-Nano-Reasoner", _ROOT_SHARD_FILES, False),
        ("nvidia/Cosmos3-Edge-Policy-DROID", _ROOT_SHARD_FILES, False),
        ("nvidia/Cosmos3-Nano-Sim-Transfer", _ROOT_SHARD_FILES, False),
        ("nvidia/Cosmos3-Nano-FP8", _ROOT_SHARD_FILES, False),
        # Cosmos3 loads only as its pipeline, so no Cosmos3-named repo without a model_index.json is the generator,
        # whoever owns it: a reasoner fine-tune, a merge, a quant, a re-upload, the base repo's reasoner files alone.
        ("someone/Cosmos3-Nano-finetuned-robotics", _ROOT_SHARD_FILES, False),
        ("someone/Cosmos3-Nano-lora-merged", _ROOT_SHARD_FILES, False),
        ("someone/Cosmos3-Edge", _ROOT_SHARD_FILES, False),
        ("unsloth/Cosmos3-Nano-unsloth-bnb-4bit", _ROOT_SHARD_FILES, False),
        ("unsloth/Cosmos3-Edge-bnb-4bit", _ROOT_SHARD_FILES, False),
        ("nvidia/Cosmos3-Edge", _ROOT_SHARD_FILES, False),
        # A pipeline lists from a repo the loader trusts only.
        ("unsloth/Cosmos3-Edge", _PIPELINE_FILES, True),
        ("someone/Cosmos3-Edge", _PIPELINE_FILES, False),
    ],
)
def test_cached_repo_listing_offers_only_the_generators_as_video(repo_id, files, video):
    """The task the On Device pickers filter on. A sibling's root shards pass the single-file trust rule (unsloth/ is
    trusted by owner), so a family name alone listed it as text-to-video, and the Chat picker sent it to the Video
    page."""
    from hub.services.models import catalog_classification as cc

    repo = _edge_cached_repo(repo_id, files)
    assert cc._cached_repo_task(repo) == (cc._VIDEO_GEN_TASK if video else None)
    # Diffusers only as a pipeline: root shards keep the row a transformers one.
    assert cc._repo_is_diffusers(repo) is (files == _PIPELINE_FILES)


def test_local_listing_offers_only_the_generators_as_video(tmp_path):
    from hub.services.models import catalog_classification as cc
    from models.models import LocalModelInfo

    def row(path: Path, model_id = None) -> LocalModelInfo:
        return LocalModelInfo(
            id = str(path),
            display_name = path.name,
            path = str(path),
            source = "models_dir",
            model_id = model_id,
        )

    def root_shards(path: Path) -> Path:
        path.mkdir(parents = True)
        for name in _ROOT_SHARD_FILES:
            (path / name).write_text("{}", encoding = "utf-8")
        return path

    video = cc._VIDEO_GEN_TASK
    assert cc._local_model_task(row(_snapshot(tmp_path / "Cosmos3-Edge"))) == video
    # Renamed, its configs name it: the Hub layout by its root config.json, a save_pretrained export by its transformer.
    assert cc._local_model_task(row(_snapshot(tmp_path / "my-video-model"))) == video
    export = _snapshot(tmp_path / "edge-export", root_config = False)
    assert cc._local_model_task(row(export)) == video
    # Root shards are no pipeline: a Reasoner, or any other Cosmos3-named folder (Cosmos3 has no single-file load).
    reasoner = root_shards(tmp_path / "Cosmos3-Nano-Reasoner")
    assert cc._local_model_task(row(reasoner, "nvidia/Cosmos3-Nano-Reasoner")) is None
    assert cc._local_model_task(row(root_shards(tmp_path / "Cosmos3-Nano-finetune"))) is None
    # A name the loader refuses lists nothing, even over a generator's pipeline: the loader judges names first.
    assert cc._local_model_task(row(_snapshot(tmp_path / "Cosmos3-Edge-FP8"))) is None
    assert cc._local_model_task(row(_snapshot(tmp_path / "Cosmos3-Edge-Reasoner"))) is None
    fp8_snapshot = _snapshot(
        tmp_path / "models--nvidia--Cosmos3-Edge-FP8" / "snapshots" / "0123abcd"
    )
    assert cc._local_model_task(row(fp8_snapshot, "nvidia/Cosmos3-Edge-FP8")) is None


@pytest.mark.parametrize(
    "root_config, transformer, family",
    [
        ({"model_type": "cosmos3_edge"}, {"hidden_size": 2048}, _EDGE),
        ({"model_type": "cosmos3_omni"}, {"hidden_size": 4096}, _NANO),
        # No root config.json (a save_pretrained export): the transformer's class and size decide.
        (None, {"_class_name": "Cosmos3OmniTransformer", "hidden_size": 2048}, _EDGE),
        (None, {"_class_name": "Cosmos3OmniTransformer", "hidden_size": 4096}, _NANO),
        (None, {"_class_name": "Cosmos3OmniTransformer", "hidden_size": 5120}, None),  # Super
        (None, {"_class_name": "WanTransformer3DModel", "hidden_size": 2048}, None),
        # A root config.json that is not Cosmos3's still decides: no family.
        (
            {"model_type": "qwen3_vl"},
            {"_class_name": "Cosmos3OmniTransformer", "hidden_size": 2048},
            None,
        ),
    ],
)
def test_a_local_pipeline_is_named_by_its_configs(tmp_path, root_config, transformer, family):
    files = {
        "model_index.json": {"_class_name": "Cosmos3OmniPipeline"},
        "transformer/config.json": transformer,
    }
    if root_config is not None:
        files["config.json"] = root_config
    for name, payload in files.items():
        (tmp_path / name).parent.mkdir(parents = True, exist_ok = True)
        (tmp_path / name).write_text(json.dumps(payload), encoding = "utf-8")
    assert vf.local_cosmos3_pipeline_family(str(tmp_path)) == family


def test_a_cosmos3_named_gguf_is_no_video_row():
    """No GGUF runs Cosmos3's generator: a Cosmos3-named GGUF known only by name (its reasoner) stays a Chat row, and
    one declaring a placeholder diffusion arch reads as a diffusion file Studio cannot run."""
    from hub.services.models import catalog_classification as cc

    assert (
        cc._unhydrated_gguf_task(("unsloth/Cosmos3-Nano-GGUF", "Cosmos3-Nano-Q4_K_M.gguf")) is None
    )
    unsupported = cc._UNSUPPORTED_DIFFUSION_TASK
    assert cc._name_hint_media_task(("Cosmos3-Edge-Q8_0.gguf",), unsupported) == unsupported


def test_a_loose_cosmos3_file_is_not_offered(tmp_path):
    """A ComfyUI loose checkpoint whose header names no family falls back to its name; Cosmos3 has no single-file load,
    while a Wan2.1 T2V name keeps its row."""
    from hub.utils import comfy_models
    for name, offered in (
        ("cosmos3_edge_bf16.safetensors", False),
        ("Cosmos3-Nano.safetensors", False),
        ("wan2.1_t2v_1.3B_bf16.safetensors", True),
    ):
        path = tmp_path / name
        path.write_bytes(b"weights")
        assert comfy_models._offer_loose_checkpoint(path) is offered, name


# nvidia/Cosmos3-Edge @ 344d602b scheduler/scheduler_config.json, as shipped (flow_shift 1.0, Karras sigmas on).
_EDGE_SCHEDULER_CONFIG = {
    "_class_name": "UniPCMultistepScheduler",
    "beta_end": 0.02,
    "beta_schedule": "linear",
    "beta_start": 0.0001,
    "disable_corrector": [],
    "dynamic_thresholding_ratio": 0.995,
    "final_sigmas_type": "zero",
    "flow_shift": 1.0,
    "lower_order_final": True,
    "num_train_timesteps": 1000,
    "predict_x0": True,
    "prediction_type": "flow_prediction",
    "rescale_betas_zero_snr": False,
    "sample_max_value": 1.0,
    "shift_terminal": None,
    "sigma_max": 200.0,
    "sigma_min": 0.147,
    "solver_order": 2,
    "solver_p": None,
    "solver_type": "bh2",
    "steps_offset": 0,
    "thresholding": False,
    "time_shift_type": "exponential",
    "timestep_spacing": "linspace",
    "trained_betas": None,
    "use_beta_sigmas": False,
    "use_dynamic_shifting": False,
    "use_exponential_sigmas": False,
    "use_flow_sigmas": True,
    "use_karras_sigmas": True,
}


def test_the_overrides_give_the_cards_unipc_schedule():
    """The card rebuilds UniPC from the shipped config with flow_shift=12.0, use_karras_sigmas=False; the family's
    overrides must give the same 20-step timesteps and sigmas, and the shipped schedule must differ (Karras sigmas
    ignore flow_shift, so a shift alone would not do)."""
    diffusers = pytest.importorskip("diffusers")
    import types

    from core.inference.diffusion_flow_shift import apply_scheduler_overrides

    unipc = getattr(diffusers, "UniPCMultistepScheduler")

    def schedule(scheduler):
        scheduler.set_timesteps(20)
        return scheduler.timesteps.tolist(), scheduler.sigmas.tolist()

    pipe = types.SimpleNamespace(scheduler = unipc.from_config(_EDGE_SCHEDULER_CONFIG))
    shipped = schedule(unipc.from_config(_EDGE_SCHEDULER_CONFIG))
    assert apply_scheduler_overrides(pipe, _fam().scheduler_overrides) is True
    card = unipc.from_config(_EDGE_SCHEDULER_CONFIG, flow_shift = 12.0, use_karras_sigmas = False)
    assert schedule(pipe.scheduler) == schedule(card)
    assert schedule(pipe.scheduler) != shipped


def test_the_pipeline_takes_the_call_the_family_makes():
    """The keywords the Edge path relies on exist on the installed Cosmos3OmniPipeline."""
    import inspect

    diffusers = pytest.importorskip("diffusers")
    pipeline = getattr(diffusers, "Cosmos3OmniPipeline", None)
    if pipeline is None:
        pytest.skip("the installed diffusers has no Cosmos3OmniPipeline")
    params = inspect.signature(pipeline.__call__).parameters
    assert {"image", "fps", "num_frames", "negative_prompt", "generator"} <= set(params)
    assert set(dict(_fam().call_kwargs)) <= set(params)


# nvidia/Cosmos3-Edge @ 344d602b, the files a pipeline load must stage and the ones around them (sizes as on the Hub).
_EDGE_LISTING = [
    ("model_index.json", 555),
    ("modular_model_index.json", 1357),
    ("config.json", 1679),
    ("model.safetensors.index.json", 68564),
    ("tokenizer.json", 17077500),
    ("tokenizer_config.json", 177253),
    ("chat_template.jinja", 12158),
    ("generation_config.json", 154),
    ("preprocessor_config.json", 334),
    ("processor_config.json", 875),
    ("video_preprocessor_config.json", 367),
    ("README.md", 53043),
    ("assets/negative_prompt.json", 17578),
    ("assets/example_i2v_input.jpg", 860765),
    ("scheduler/scheduler_config.json", 889),
    ("text_tokenizer/chat_template.jinja", 12158),
    ("text_tokenizer/special_tokens_map.json", 563),
    ("text_tokenizer/tokenizer.json", 17077500),
    ("text_tokenizer/tokenizer_config.json", 393),
    ("transformer/config.json", 1075),
    ("transformer/diffusion_pytorch_model-00001-of-00002.safetensors", 5000040072),
    ("transformer/diffusion_pytorch_model-00002-of-00002.safetensors", 1739335512),
    ("transformer/diffusion_pytorch_model.safetensors.index.json", 53623),
    ("vae/config.json", 1755),
    ("vae/diffusion_pytorch_model.safetensors", 1409400600),
    ("vision_encoder/model.safetensors", 978739880),
]


def test_the_download_plan_stages_every_pipeline_component():
    """Cosmos3 repos ship a modular_model_index.json beside model_index.json; read as MiniMax-H3's layout it dropped
    text_tokenizer/. A pipeline pick stages the index, the scheduler, the tokenizer, the transformer and the VAE."""
    import types

    from core.inference.video import VideoBackend

    info = types.SimpleNamespace(
        siblings = [types.SimpleNamespace(rfilename = n, size = s) for n, s in _EDGE_LISTING]
    )
    names = {n for n, _ in VideoBackend._base_download_files(info, "pipeline")}
    assert {
        "model_index.json",
        "scheduler/scheduler_config.json",
        "text_tokenizer/tokenizer.json",
        "text_tokenizer/tokenizer_config.json",
        "transformer/config.json",
        "transformer/diffusion_pytorch_model.safetensors.index.json",
        "transformer/diffusion_pytorch_model-00001-of-00002.safetensors",
        "transformer/diffusion_pytorch_model-00002-of-00002.safetensors",
        "vae/config.json",
        "vae/diffusion_pytorch_model.safetensors",
    } <= names
