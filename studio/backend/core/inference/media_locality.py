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
    """The diffusion family for *pick*, tried against its path and then its id.

    The path comes first because it is the only needle the load route is ever handed, and the
    only one that can carry a local ``model_index.json``: ``detect_family_for_pick`` reads that
    index ahead of any guess made from a name, so asking about the id first answers FLUX for a
    HiDream pipeline in a directory called ``flux.1`` while the loader answers HiDream and
    fetches its 16 GB encoder. The id is kept as a fallback for a pick whose path says nothing.
    """
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
    """The pick as the LOAD route will read it, with a bare single-file directory reinterpreted.

    Both load routes turn a kindless directory holding exactly one checkpoint into a
    ``single_file`` load and then resolve that family's companions. Planning the un-normalized
    pick describes a local pipeline with nothing to fetch, and misses those companions.
    """
    from core.inference.diffusion import resolve_local_single_file

    if pick.model_kind or pick.gguf_filename:
        return pick
    sole = resolve_local_single_file(pick.model_path)
    if sole is None:
        return pick
    return replace(pick, gguf_filename = sole, model_kind = "single_file")


def is_edit_only(pick: MediaModelPick) -> bool:
    """Whether *pick* is an instruction-editing family, which has no text-to-image mode.

    The local catalog tags these text-to-image, so without this the switch would evict a working
    model for a multi-GB pipeline that /v1/images/generations then refuses for lacking txt2img.
    """
    from core.inference.diffusion import _family_workflows

    fam = detected_image_family(normalized_pick(pick))
    if fam is None:
        return False
    return "txt2img" not in _family_workflows(fam)


def _needs_external_encoder(pick: MediaModelPick) -> bool:
    """Whether this pick's pipeline fetches an encoder that its own directory cannot hold."""
    # an unrecognised family keeps the shortcut, since refusing every on-device model is worse
    fam = detected_image_family(pick)
    return fam is not None and getattr(fam, "name", "") in _EXTERNAL_ENCODER_FAMILIES


def _cached_snapshot_file(repo_id: str, filename: str) -> Optional[str]:
    """The cached path of ``filename`` in ``repo_id``, or None when it is not downloaded."""
    from huggingface_hub import try_to_load_from_cache

    from core.inference.diffusion import hub_cache_dir

    hit = try_to_load_from_cache(repo_id, filename, cache_dir = hub_cache_dir())
    return hit if isinstance(hit, str) else None


def encoder_repo_complete(repo_id: str) -> bool:
    """Whether every shard of a cached encoder repo is present, not merely one of them.

    ``_upstream_is_cached`` counts any single weight file, while the pipeline calls
    from_pretrained on the whole repository, so an interrupted sharded pull would otherwise
    read as local and the load would fetch the rest.

    The config and tokenizer files count as much as the shards. They are kilobytes rather than
    gigabytes, but the encoder is built with ``AutoTokenizer.from_pretrained`` and
    ``LlamaForCausalLM.from_pretrained`` on the whole repository, so a cache holding every shard
    and none of those still reaches the Hub during an accepted switch.
    """
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
    """0 when this local pipeline needs nothing more, else what its outside dependency costs.

    HiDream-I1 loads unsloth/Meta-Llama-3.1-8B-Instruct unconditionally, around 16 GB, which no
    amount of the pipeline being on disk accounts for. Checked against the cache directly rather
    than through the planner, which cannot be handed an absolute pipeline path.
    """
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
    """Whether this local video pick is an LTX-2.3 checkpoint the plan did not treat as one.

    The planner judges 2.3 by name, while the loader reads the checkpoint header and then pulls
    the 2.3 VAE, audio and connector artifacts. A renamed checkpoint therefore plans as 2.0,
    reports nothing missing, and downloads those extras during assembly.

    The family is resolved the way the loader resolves it, which falls back to the checkpoint's
    ``general.architecture`` where neither the repo nor the filename carries a family token.
    Deciding by name alone left a generically named LTX checkpoint exempt from the very check
    its header would have triggered.
    """
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
    """Every engine whose plan this pick could end up loading through.

    Usually one. ``predict_engine`` treats an absent sd.cpp binary as available whenever its
    installation is allowed, while ``select_and_activate_engine`` falls back to diffusers when
    that install produces nothing runnable, and the two engines read different companion sets.
    Both are verified only in that case: with a runnable binary already on disk the load stays
    native, and demanding the diffusers shards too would refuse a model sd.cpp can serve.
    """
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
    predicted = predict_engine(fam, model_kind = kind)
    names = [predicted]
    if predicted == ENGINE_SD_CPP and not native_binary_installed():
        names.append(ENGINE_DIFFUSERS)
    return [engine_for(name) for name in names]


def plan_gpu_ordinal() -> Optional[int]:
    """The card the load route will rank for itself, so the plan sizes the same file set.

    Automatic precision is chosen per card, and a different card can select a different hosted
    pre-quantized artifact, which a plan plotted against the default device would omit.
    """
    from core.inference.diffusion_device import (
        resolve_diffusion_device_target,
        resolve_selected_cuda_ordinal,
    )

    if resolve_diffusion_device_target().device != "cuda":
        return None
    return resolve_selected_cuda_ordinal(None)


def _pipeline_components_present(root: Path) -> bool:
    """Whether every component a local pipeline's own index names is on disk.

    A directory carrying a pipeline index is treated as complete by definition, because
    from_pretrained reads it off disk and the planner cannot be asked about an absolute path.
    That holds only if the components are actually there: a hand-copied or interrupted pipeline
    passes the index check, and the loader then tears the resident pipeline down before
    from_pretrained discovers the gap, leaving the API with no model at all.

    Judged on what can be seen without reading weights. A directory carrying neither index is
    not this function's business. A modular entry whose spec names another repository is checked
    against the cache instead, since the load pulls that repository itself.
    """
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
        for component, entry in index.items():
            if component.startswith("_") or not isinstance(entry, (list, tuple)):
                continue
            # [null, null] marks a component this pipeline deliberately ships without.
            if len(entry) not in (2, 3) or not entry[1]:
                continue
            # a modular entry is [library, class, spec], and its spec can name another repo,
            # which this directory is never expected to hold but the load still pulls
            hosted = _hosted_source(entry[2]) if len(entry) == 3 else None
            if hosted is not None:
                if not _hosted_component_cached(*hosted):
                    return False
                continue
            if not _component_present(root / component):
                return False
    return True


def _hosted_source(spec: Any) -> Optional[tuple[str, str, str, str]]:
    """What a modular index entry asks the loader for: repo, subfolder, revision and variant.

    ``ComponentSpec.load`` is handed the whole spec, so a component can pin a commit or a named
    weight variant. Checking the default snapshot for those would approve a switch and then
    download the pinned files after the resident pipeline is gone.
    """
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
    """The cached snapshot a load of *repo_id* would read, or None when it is not downloaded.

    The revision the loader asks for: the pinned one where a spec names it, else the one
    ``refs/main`` resolves to, rather than whichever snapshot sorts first. A superseded revision
    can hold a complete component while the active one is partial, and approving the old copy is
    how the load ends up fetching the new one.
    """
    from core.inference.diffusion import hub_cache_dir

    repo_dir = Path(hub_cache_dir()) / f"models--{repo_id.replace('/', '--')}"
    snapshots = repo_dir / "snapshots"
    # a pinned revision is a commit sha, or a branch or tag the cache records under refs/, and
    # it is the only candidate: falling back to main is how the default snapshot approves a pin
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
        # pinned and not cached under that name: the default snapshot is not what will load
        return None
    try:
        # no ref file means a commit-pinned download, where any cached revision is the one
        return next((child for child in sorted(snapshots.iterdir()) if child.is_dir()), None)
    except OSError:
        return None


def _hosted_component_cached(source: str, subfolder: str, revision: str, variant: str) -> bool:
    """Whether a modular component the index sources elsewhere is already on disk.

    ``load_components`` pulls each repository the index names, and the video planner omits its
    base manifest whenever the selected path exists, so a local modular directory with a missing
    hosted component would otherwise verify clean and download it after the eviction.
    """
    local = Path(source).expanduser()
    try:
        if local.is_dir():
            return _component_present(local / subfolder if subfolder else local, variant)
    except OSError:
        return False
    snapshot = _cached_snapshot_root(source, revision)
    if snapshot is None:
        return False
    # the same component rules either way: _upstream_is_cached's no-manifest branch is satisfied
    # by a single weight file, which an interrupted sharded pull leaves behind
    return _component_present(snapshot / subfolder if subfolder else snapshot, variant)


def _component_present(component: Path, variant: str = "") -> bool:
    """Whether one named pipeline component holds what from_pretrained will ask it for.

    ``variant`` is the named weight set a modular spec can pin (``fp16`` and the like), which
    from_pretrained requires by name rather than falling back to the default files.

    Judged on entries that are real FILES, not merely names in the directory listing. An HF cache
    snapshot holds symlinks into ``blobs/``, and a deleted blob leaves the link behind: matching
    on the name alone reads such a component as complete, evicts the resident pipeline, and then
    fails in from_pretrained with nothing loaded. The same listing on Windows without developer
    mode holds copies rather than links and cannot express that state at all, so the two hosts
    disagreed about the very same repository. A directory that merely ends in ``.safetensors``
    is excluded by the same test.
    """
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
    # kept on the full listing: a shard index whose blob is gone must still route here, where
    # _shards_declared reads the unreadable index and refuses, rather than fall through to the
    # weight test below and pass on whichever sibling shard did survive
    if any(entry.name.endswith(".index.json") for entry in entries):
        # an index is proof only once it declares something; an empty weight_map declares nothing
        return _shards_declared(component)
    # a weight-bearing component declares config.json; schedulers, tokenizers and processors
    # carry their own *_config.json instead and ship no weights at all
    if (component / "config.json").is_file():
        return any(entry.suffix.lower() in _WEIGHT_SUFFIXES for entry in files)
    # a tokenizer ships no weights but is still useless without its vocabulary, and which file
    # that is varies by class, so any one of the known spellings answers for all of them
    if (component / "tokenizer_config.json").is_file():
        return any((component / name).is_file() for name in _TOKENIZER_ASSETS)
    # a metadata-only component is its config: a scheduler or processor directory holding
    # anything else at all (a stray README) builds nothing and is fetched at load time
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
    """Bytes this pick would still have to fetch, or 0 when nothing is missing.

    Planned against the engine that will LOAD this pick, the way /images/download-plan does:
    the resident engine can be native sd.cpp while the target loads through diffusers, and its
    planner refuses the pick, which the catch below would read as nothing missing.

    A local full IMAGE pipeline is complete by definition, since from_pretrained reads it off
    disk and the planner would ask the Hub about an absolute path and fail, which reads as
    unverifiable and would refuse every on-device model. Video is excluded: a local MiniMax-H3
    modular pipeline still substitutes a hosted quantized conditioner, tens of GB the loader
    fetches during assembly, so it has to be planned like any other pick.

    Returns None when locality could not be established: the image planner raises, and the
    video one returns zero bytes with ``plan_failed`` because its own caller falls back to an
    inline pull. Either way zero is not evidence of a complete cache, and treating it as such
    would allow exactly the download this exists to prevent, so the switch refuses instead.
    """
    target = normalized_pick(pick)
    local_pipeline = not target.gguf_filename and Path(target.model_path).is_dir()
    if local_pipeline and not _pipeline_components_present(Path(target.model_path)):
        return UNSIZED_MISSING
    if owner == DIFFUSION:
        # asked of every image pick, not only local pipelines: a single-file HiDream checkpoint
        # plans clean and its assembly still loads the encoder repo unconditionally
        external = _missing_external_encoder(target)
        if external is None or external:
            return external
        if local_pipeline:
            # the pipeline is present, so only a dependency outside it could still be fetched
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
                # Only the verdict, not the probe: this asks whether the pick is already on
                # disk, so it must count the SAME files the load will fetch. Clearing the probe
                # drops the pre-cast encoder and the GGUF dense-transformer widening, which is
                # how a "fully downloaded" answer goes wrong.
                memory_verdict = False,
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
    "missing_download_bytes",
    "normalized_pick",
    "plan_gpu_ordinal",
    "planners_for",
]
