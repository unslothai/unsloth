# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Proving a media model is already downloaded, before the switch evicts anything for it.

Auto-switch never downloads. That promise is easy to state and hard to keep, because the index
only sees CHECKPOINTS: a GGUF or single-file pick loads its text encoders and VAE from a
companion base repo, HiDream-I1 fetches a 16 GB Llama encoder no amount of pipeline on disk
accounts for, and an LTX-2.3 checkpoint pulls VAE, audio and connector artifacts the planner
only recognises by name. Any of those would let one API request spend tens of gigabytes.

So locality is verified through the same download planner ``/images/download-plan`` serves, and
the answer is tri-state: complete, incomplete by some number of bytes, or unverifiable. Zero
bytes from a planner that failed is not evidence of a complete cache, and the switch refuses on
anything short of proof.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Optional

# The auto-switch pin lives in diffusion_device, so every resolve_auto_cuda_ordinal caller sees it.
from core.inference.diffusion_device import _PLANNED_ORDINAL, _UNPLANNED
from core.inference.gpu_arbiter import DIFFUSION, VIDEO
from core.inference.media_model_index import MediaModelPick
from core.inference.media_switch_backends import backend_for
from core.inference.media_switch_errors import UNSIZED_MISSING
from loggers import get_logger

logger = get_logger(__name__)

# the image family whose pipeline loads a separate encoder repo its own directory cannot hold
_EXTERNAL_ENCODER_FAMILIES = frozenset({"hidream-i1"})

# encoder repos that always ship sharded, where a missing index means an interrupted download
_SHARDED_ENCODER_REPOS = frozenset({"unsloth/Meta-Llama-3.1-8B-Instruct"})

# what from_pretrained reads besides the weights, tiny next to them but still a download
_ENCODER_METADATA_FILES = ("config.json", "tokenizer.json", "tokenizer_config.json")

# suffixes a weight-bearing pipeline component can satisfy from_pretrained with
_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".msgpack", ".onnx")

# any one of these is the vocabulary a tokenizer class builds itself from
_TOKENIZER_ASSETS = (
    "tokenizer.json",
    "vocab.json",
    "vocab.txt",
    "merges.txt",
    "spiece.model",
    "tokenizer.model",
    "sentencepiece.bpe.model",
)


def detected_image_family(pick: MediaModelPick) -> Any:
    """Path first, since only it can carry a local model_index.json; the id is a fallback."""
    from core.inference.diffusion_families import detect_family_for_pick

    for needle in (pick.model_path, pick.model_id):
        if not needle:
            continue
        try:
            fam = detect_family_for_pick(needle, pick.gguf_filename, None)
        except Exception:  # noqa: BLE001 -- a probe failure must not refuse a loadable pick
            continue
        if fam is not None:
            return fam
    return None


def normalized_pick(pick: MediaModelPick) -> MediaModelPick:
    """Pick as the load routes read it, so companions of a bare single-file directory are planned."""
    from core.inference.diffusion import resolve_local_single_file, split_local_checkpoint_path

    if pick.model_kind or pick.gguf_filename:
        return pick
    split = split_local_checkpoint_path(pick.model_path)
    if split is not None:
        return replace(pick, model_path = split[0], gguf_filename = split[1], model_kind = "single_file")
    sole = resolve_local_single_file(pick.model_path)
    if sole is None:
        return pick
    return replace(pick, gguf_filename = sole, model_kind = "single_file")


def is_edit_only(pick: MediaModelPick) -> bool:
    """Instruction-edit families lack txt2img, though the local catalog tags them text-to-image."""
    from core.inference.diffusion import _family_workflows

    fam = detected_image_family(normalized_pick(pick))
    if fam is None:
        return False
    return "txt2img" not in _family_workflows(fam)


def _needs_external_encoder(pick: MediaModelPick) -> bool:
    """Whether this pick's pipeline fetches an encoder that its own directory cannot hold."""
    fam = detected_image_family(pick)
    return fam is not None and getattr(fam, "name", "") in _EXTERNAL_ENCODER_FAMILIES


def _cached_snapshot_file(repo_id: str, filename: str) -> Optional[str]:
    """The cached path of ``filename`` in ``repo_id``, or None when it is not downloaded."""
    from huggingface_hub import try_to_load_from_cache

    from core.inference.diffusion import hub_cache_dir

    hit = try_to_load_from_cache(repo_id, filename, cache_dir = hub_cache_dir())
    return hit if isinstance(hit, str) else None


def encoder_repo_complete(repo_id: str) -> bool:
    """All shards plus config and tokenizer must be cached, or from_pretrained fetches the rest."""
    import json

    from core.inference.diffusion_families import _upstream_is_cached, cache_holds_files

    if not _upstream_is_cached(repo_id):
        return False
    if not cache_holds_files(repo_id, list(_ENCODER_METADATA_FILES)):
        return False
    index = _cached_snapshot_file(repo_id, "model.safetensors.index.json")
    if index is None:
        # a repo known to be sharded has no unsharded reading, so a missing index means a partial
        return repo_id not in _SHARDED_ENCODER_REPOS
    with open(index, encoding = "utf-8") as handle:
        shards = sorted(set((json.load(handle).get("weight_map") or {}).values()))
    return bool(shards) and cache_holds_files(repo_id, shards)


def _missing_external_encoder(pick: MediaModelPick) -> Optional[int]:
    """HiDream loads an external Llama encoder (~16 GB) that the local pipeline does not contain."""
    if not _needs_external_encoder(pick):
        return 0
    from core.inference.diffusion_hidream import HIDREAM_LLAMA_REPO

    try:
        if encoder_repo_complete(HIDREAM_LLAMA_REPO):
            return 0
    except Exception as exc:  # noqa: BLE001 -- an unreadable cache is not proof of locality
        logger.debug("media auto-switch: hidream encoder probe failed: %s", exc)
        return None
    return UNSIZED_MISSING


def hidden_ltx23_extras(owner: str, pick: MediaModelPick) -> bool:
    """A renamed LTX-2.3 checkpoint: the planner judges by name, the loader by its header."""
    if owner != VIDEO or not pick.gguf_filename:
        return False
    try:
        from core.inference.diffusion_families import resolve_local_gguf_child
        from core.inference.video import _detect_load_family
        from core.inference.video_ltx2 import LTX23_EXTRAS_REPO, is_ltx23_checkpoint
    except Exception:  # noqa: BLE001 -- no ltx support here means nothing to hide
        return False
    fam = _detect_load_family(pick.model_path, pick.gguf_filename, None) or (
        _detect_load_family(pick.model_id, pick.gguf_filename, None) if pick.model_id else None
    )
    if fam is None or getattr(fam, "name", None) != "ltx-2":
        return False
    root = Path(pick.model_path).expanduser()
    try:
        if root.exists():
            checkpoint = resolve_local_gguf_child(root, pick.gguf_filename)
        else:
            # a cached repo id: the checkpoint is on disk all the same, and its header decides
            cached = _cached_snapshot_file(pick.model_path, pick.gguf_filename)
            if cached is None:
                return False
            checkpoint = Path(cached)
    except Exception:  # noqa: BLE001 -- an unreadable pick is refused by the load itself
        return False
    if not is_ltx23_checkpoint(checkpoint):
        return False
    from core.inference.diffusion_families import cache_holds_files
    from core.inference.video_ltx2 import ltx23_extras_files

    extras = ltx23_extras_files(checkpoint)
    # the exact three artifacts, since the repo also holds checkpoints that prove nothing here
    return bool(extras) and not cache_holds_files(LTX23_EXTRAS_REPO, list(extras))


def planners_for(owner: str, pick: MediaModelPick) -> list:
    """Every engine the pick may load through: sd.cpp can fall back to diffusers, with other companions."""
    if owner != DIFFUSION:
        return [backend_for(owner)]
    from core.inference.diffusion import resolve_model_kind
    from core.inference.diffusion_engine_router import (
        engine_for,
        native_binary_installed,
        predict_engine,
    )
    from core.inference.sd_cpp_engine import ENGINE_DIFFUSERS, ENGINE_SD_CPP

    fam = detected_image_family(pick)
    if fam is None:
        return [backend_for(owner)]
    kind = resolve_model_kind(pick.gguf_filename, pick.model_kind)
    predicted = predict_engine(fam, model_kind = kind, gguf_filename = pick.gguf_filename)
    names = [predicted]
    if predicted == ENGINE_SD_CPP and not native_binary_installed():
        names.append(ENGINE_DIFFUSERS)
    return [engine_for(name) for name in names]


def pin_planned_gpu_ordinal(ordinal: Optional[int]) -> Any:
    """Carry ``ordinal`` into this context's plan and load; returns the reset token."""
    return _PLANNED_ORDINAL.set(ordinal)


def planned_gpu_ordinal() -> tuple[bool, Optional[int]]:
    """(True, ordinal) when this context carries a pinned automatic ordinal, else (False, None)."""
    ordinal = _PLANNED_ORDINAL.get()
    return (False, None) if ordinal is _UNPLANNED else (True, ordinal)


def plan_gpu_ordinal() -> Optional[int]:
    """The card the load route will rank for itself, so the plan sizes the same file set.

    Automatic precision is chosen per card, and a different card can select a different hosted
    pre-quantized artifact, which a plan plotted against the default device would omit.
    """
    pinned, ordinal = planned_gpu_ordinal()
    if pinned:
        return ordinal
    from core.inference.diffusion_device import auto_cuda_ordinal_for_load
    from core.inference.diffusion_memory import upcoming_load_budget

    with upcoming_load_budget():
        return auto_cuda_ordinal_for_load()


def catalog_gpu_ordinal() -> Optional[int]:
    """The one ordinal a catalog refresh plans every pick against, ranked once unless the cards are identical.

    Identical visible cards pick the same precision and so the same artifacts on any of them, so the
    refresh plans against the current device as it always did instead of re-ranking per pick.
    """
    pinned, ordinal = planned_gpu_ordinal()
    if pinned:
        return ordinal
    from core.inference.diffusion_device import auto_cuda_ranking_possible

    if not auto_cuda_ranking_possible():
        return None
    try:
        from utils.hardware import hardware as hw

        # Under a CUDA order nvidia-smi does not share, mask ids are not its rows: compare every card.
        ids = hw.get_parent_visible_gpu_ids() if hw._cuda_order_matches_smi() else None
        if hw._cuda_cards_identical(ids) is True:
            return None
    except Exception:  # noqa: BLE001 -- unknown identity ranks once, like mixed cards
        pass
    return plan_gpu_ordinal()


def _pipeline_components_present(root: Path) -> bool:
    """The components a pipeline index names must exist, or the loader drops the resident model first."""
    import json

    for name in ("model_index.json", "modular_model_index.json"):
        index_file = root / name
        if not index_file.is_file():
            continue
        try:
            with open(index_file, encoding = "utf-8-sig") as handle:
                index = json.load(handle)
        except Exception as exc:  # noqa: BLE001 -- an index the loader cannot read is not complete
            logger.debug("media auto-switch: unreadable pipeline index under %s: %s", root, exc)
            return False
        if not isinstance(index, dict):
            return False
        if name == "modular_model_index.json" and _is_flux3_manifest(index):
            # FLUX.3 loads a fixed set of components from fixed folders of this root and never follows a spec's
            # source, so judge it on exactly those: the optional prompt upsampler the manifests list since
            # 2026-10-16 lives in another (~19 GB) repo an image load never reads.
            from core.inference.diffusion_families import local_pipeline_components_are_complete
            from core.inference.diffusion_flux3 import FLUX3_COMPONENT_FOLDERS

            if not local_pipeline_components_are_complete(
                root, name, fixed_component_folders = FLUX3_COMPONENT_FOLDERS
            ):
                return False
            continue
        for component, entry in index.items():
            if component.startswith("_") or not isinstance(entry, (list, tuple)):
                continue
            # [null, null] marks a component this pipeline deliberately ships without.
            if len(entry) not in (2, 3) or not entry[1]:
                continue
            # A modular spec can name another repo this directory never holds, yet the load still pulls it.
            hosted = _hosted_source(entry[2]) if len(entry) == 3 else None
            if hosted is not None:
                if not _hosted_component_cached(*hosted):
                    return False
                continue
            if not _component_present(root / component):
                return False
    return True


def flux3_seeded_cache_repo(path: Any) -> Optional[str]:
    """The Hub repo id of a FLUX.3 snapshot in the HF cache (the one refs/main names) whose only gaps are the dense
    transformer and text-encoder weights (what a download planned around a hosted pre-quantized seed leaves: the int8
    ConvRot encoder is the default, so a normally staged snapshot holds no dense encoder shards either), or None.
    Everything else the loader opens, both configs and the tokenizer included, must be there; whether the seeds or
    the dense shards the load will open are cached is the download planner's question, asked of the repo id."""
    import json
    try:
        snapshot = Path(path)
        repo_dir = snapshot.parent.parent
        if snapshot.parent.name != "snapshots" or not repo_dir.name.startswith("models--"):
            return None
        # A load by repo id resolves refs/main, so only the snapshot main names may be rewritten: another (or a
        # commit-pinned) snapshot would be swapped for a different one.
        if (repo_dir / "refs" / "main").read_text(encoding = "utf-8").strip() != snapshot.name:
            return None
        index = json.loads((snapshot / "modular_model_index.json").read_text(encoding = "utf-8-sig"))
        if not isinstance(index, dict) or not _is_flux3_manifest(index):
            return None
        from core.inference.diffusion_families import local_pipeline_components_are_complete
        from core.inference.diffusion_flux3 import FLUX3_COMPONENT_FOLDERS

        if not local_pipeline_components_are_complete(
            snapshot,
            "modular_model_index.json",
            excluded_components = ("transformer", "text_encoder"),
            fixed_component_folders = FLUX3_COMPONENT_FOLDERS,
        ):
            return None
        owner, _, name = repo_dir.name[len("models--") :].partition("--")
        return f"{owner}/{name}" if owner and name else None
    except (OSError, ValueError):
        return None


def seeded_family_override(local_path: Any, repo_id: str) -> Optional[str]:
    """The family a seeded snapshot's modular_model_index.json declares, when its repo id alone would not name it
    (an opaquely named repo), else None. Planning and loading by the repo id both need it."""
    from core.inference.diffusion_families import detect_family_for_pick

    local = detect_family_for_pick(str(local_path))
    by_repo = detect_family_for_pick(repo_id)
    if local is not None and (by_repo is None or by_repo.name != local.name):
        return local.name
    return None


def _is_flux3_manifest(index: dict) -> bool:
    from core.inference.diffusion_flux3 import flux3_index_family
    return flux3_index_family(index) is not None


def _hosted_source(spec: Any) -> Optional[tuple[str, str, str, str]]:
    """A modular spec may pin a commit or variant; checking the default snapshot would miss it."""
    if not isinstance(spec, dict):
        return None
    source = spec.get("pretrained_model_name_or_path") or spec.get("repo")
    if not isinstance(source, str) or not source.strip():
        return None

    def _text(key: str) -> str:
        value = spec.get(key)
        return value.strip() if isinstance(value, str) else ""

    return source.strip(), _text("subfolder"), _text("revision"), _text("variant")


def _cached_snapshot_root(repo_id: str, revision: str = "") -> Optional[Path]:
    """The snapshot of the revision the loader requests, not whichever snapshot sorts first."""
    from core.inference.diffusion import hub_cache_dir

    repo_dir = Path(hub_cache_dir()) / f"models--{repo_id.replace('/', '--')}"
    snapshots = repo_dir / "snapshots"
    # a pinned revision is a commit sha, or a branch or tag the cache records under refs/, and it is the only
    # candidate: falling back to main is how the default snapshot approves a pin
    for candidate in [revision] if revision else ["main"]:
        pinned = snapshots / candidate
        if pinned.is_dir():
            return pinned
        try:
            ref = (repo_dir / "refs" / candidate).read_text(encoding = "utf-8").strip()
        except OSError:
            continue
        resolved = snapshots / ref if ref else None
        if resolved is not None and resolved.is_dir():
            return resolved
    if revision:
        return None
    try:
        # no ref file means a commit-pinned download, where any cached revision is the one
        return next((child for child in sorted(snapshots.iterdir()) if child.is_dir()), None)
    except OSError:
        return None


def _hosted_component_cached(source: str, subfolder: str, revision: str, variant: str) -> bool:
    """The loader pulls each repo a local modular index names, so those must be cached too."""
    local = Path(source).expanduser()
    try:
        if local.is_dir():
            return _component_present(local / subfolder if subfolder else local, variant)
    except OSError:
        return False
    snapshot = _cached_snapshot_root(source, revision)
    if snapshot is None:
        return False
    # An interrupted sharded pull leaves a single weight file, so use the component rules
    return _component_present(snapshot / subfolder if subfolder else snapshot, variant)


def _component_present(component: Path, variant: str = "") -> bool:
    """Judged on real files, not names: a symlink whose blob was deleted would pass a name-only check."""
    try:
        if not component.is_dir():
            return False
        entries = list(component.iterdir())
        files = [entry for entry in entries if entry.is_file()]
    except OSError:
        return False
    if not entries:
        return False
    if not _shards_present(component):
        return False
    if variant and not any(f".{variant}." in entry.name for entry in files):
        return False
    # Kept on the full listing so an index whose blob is gone refuses, not passes on a sibling
    if any(entry.name.endswith(".index.json") for entry in entries):
        # an index is proof only once it declares something; an empty weight_map declares nothing
        return _shards_declared(component)
    # Weight-bearing components declare config.json; metadata-only ones use *_config.json instead.
    if (component / "config.json").is_file():
        return any(entry.suffix.lower() in _WEIGHT_SUFFIXES for entry in files)
    # A tokenizer needs its vocabulary file, which name varies by class, so any known spelling answers.
    if (component / "tokenizer_config.json").is_file():
        return any((component / name).is_file() for name in _TOKENIZER_ASSETS)
    # a metadata-only component is its config: a scheduler or processor directory holding anything else at all (a stray
    # README) builds nothing and is fetched at load time
    return any(entry.name.endswith("config.json") for entry in files)


def _shards_declared(component: Path) -> bool:
    """Whether any shard index in *component* names at least one weight file."""
    import json

    for index_file in component.glob("*.index.json"):
        try:
            with open(index_file, encoding = "utf-8-sig") as handle:
                if (json.load(handle) or {}).get("weight_map"):
                    return True
        except Exception:  # noqa: BLE001 -- an unreadable index declares nothing
            return False
    return False


def _shards_present(component: Path) -> bool:
    """Whether a sharded component holds every file its own weight index names."""
    import json

    for index_file in component.glob("*.index.json"):
        try:
            with open(index_file, encoding = "utf-8-sig") as handle:
                weight_map = (json.load(handle) or {}).get("weight_map") or {}
        except Exception:  # noqa: BLE001 -- an unreadable shard index is not evidence of presence
            return False
        if any(not (component / shard).is_file() for shard in set(weight_map.values())):
            return False
    return True


def missing_download_bytes(
    owner: str,
    pick: MediaModelPick,
    hf_token: Optional[str] = None,
) -> Optional[int]:
    """None when locality is unproven, since zero bytes is not evidence of a complete cache."""
    target = normalized_pick(pick)
    family_override = None
    local_pipeline = not target.gguf_filename and Path(target.model_path).is_dir()
    if local_pipeline and not _pipeline_components_present(Path(target.model_path)):
        repo = flux3_seeded_cache_repo(target.model_path) if owner == DIFFUSION else None
        if repo is None:
            return UNSIZED_MISSING
        # A FLUX.3 snapshot Studio staged for a pre-quantized seed holds no dense transformer shards: whether the seed
        # is cached and is what the load picks is the planner's question, so ask it of the repo, which is also what
        # the switch loads (cache only), instead of requiring dense weights the load never opens.
        family_override = seeded_family_override(target.model_path, repo)
        target = replace(target, model_path = repo)
        local_pipeline = False
    if owner == DIFFUSION:
        # Applies to all image picks; single-file HiDream plans clean but assembly loads the encoder repo.
        external = _missing_external_encoder(target)
        if external is None or external:
            return external
        if local_pipeline:
            return 0
    try:
        ordinal = plan_gpu_ordinal()
        plans = [
            planner.download_plan(
                target.model_path,
                gguf_filename = target.gguf_filename,
                model_kind = target.model_kind,
                gpu_ordinal = ordinal,
                hf_token = hf_token,
                # Verdict only, not the probe: must count the same files the load will fetch
                memory_verdict = False,
                **({"family_override": family_override} if family_override else {}),
            )
            or {}
            for planner in planners_for(owner, target)
        ]
    except Exception as exc:  # noqa: BLE001 -- see the docstring
        logger.debug("media auto-switch: download plan for %s failed: %s", pick.model_id, exc)
        return None
    if any(plan.get("plan_failed") for plan in plans):
        return None
    # cached in full and still unloadable (a flux.2 gguf on a different-size base) shows up here
    if any(plan.get("incompatible_reason") for plan in plans):
        return None
    if hidden_ltx23_extras(owner, target):
        return UNSIZED_MISSING
    missing = max((max(0, int(plan.get("total_bytes") or 0)) for plan in plans), default = 0)
    # both planners coerce an unknown size to zero, so entries decide and bytes only describe
    if not missing and any(plan.get("entries") for plan in plans):
        return UNSIZED_MISSING
    return missing


__all__ = [
    "detected_image_family",
    "encoder_repo_complete",
    "hidden_ltx23_extras",
    "is_edit_only",
    "flux3_seeded_cache_repo",
    "seeded_family_override",
    "missing_download_bytes",
    "normalized_pick",
    "pin_planned_gpu_ordinal",
    "plan_gpu_ordinal",
    "planned_gpu_ordinal",
    "planners_for",
    "catalog_gpu_ordinal",
]
