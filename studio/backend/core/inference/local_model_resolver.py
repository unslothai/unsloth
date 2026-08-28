# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Resolve an OpenAI-request ``model`` string to a downloaded local model.

Used by the opt-in auto-switch path. Two kinds of local model qualify: a GGUF
(served by llama.cpp, with a quant that is actually on disk) and a non-GGUF
checkpoint such as safetensors or MLX weights (served by the inference
orchestrator). The match is conservative either way: only names that map to
something already downloaded are eligible, so an arbitrary OpenAI model string
still falls through to the loaded model (drop-in compat) and no surprise
multi-GB download is ever triggered. The local-model scan is cached for a few
seconds since auto-switch consults it per request.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Optional

from core.inference.model_ids import public_model_id
from loggers import get_logger

logger = get_logger(__name__)


@dataclass(frozen = True)
class _LocalGgufEntry:
    loader_id: str  # advertised id (repo id / folder name), also the override key
    load_path: str  # concrete on-disk dir/file passed to /load so it never downloads
    variants: tuple[str, ...]  # local quant labels; () for a standalone .gguf
    is_gguf: bool = True  # False routes the load to the inference orchestrator


_CACHE_TTL_S = 5.0
# Monotonic timestamps are nonnegative, so a negative stamp encodes "additions-only
# invalidated at -stamp". Keeps that trust inside the atomically published _scan
# tuple instead of a second global, and keeps it time-bounded like any other.
_lock = threading.Lock()
_scan: tuple[float, dict[str, _LocalGgufEntry]] = (0.0, {})
# Not _lock: that is held for the whole scan, so the request path would wait on it.
_warm_lock = threading.Lock()
# Repos that finished downloading but are not in the published index yet: nothing
# else covers them until the next scan, and the request path must not call them absent.
_just_downloaded: set[str] = set()
_warming = False
# An invalidation landing while a warmer owns the slot asks it for another pass, so
# a snapshot published already-stale is rebuilt off the request path. Callers still
# pair invalidate_index() with warm_index_soon() for the case where it has retired.
_warm_pending = False
_last_scan_s = 0.0
# Rescan at most a tenth of the time: on the TTL alone a slow scan would run continuously.
_WARM_DUTY = 10.0


def _is_abs_path_id(value: str) -> bool:
    """True when an id is an absolute filesystem path (the ./models and LM Studio
    scanners use the on-disk path as the id) rather than a repo id like org/name.

    Both spellings count on every host. Path() follows the running OS, so a
    Windows backend read "/home/me/x.gguf" as relative and a POSIX one read
    "C:\\models\\x.gguf" the same way, and either then reached /v1/models as a
    published id. Ids outlive the machine that wrote them: settings sync, a WSL
    session and a copied config all carry the other platform's spelling, and the
    model-override identity already folds both. Neither reading can misfire on a
    repo id, which has no leading separator, drive or UNC prefix."""
    from pathlib import PurePosixPath, PureWindowsPath
    try:
        return PurePosixPath(value).is_absolute() or PureWindowsPath(value).is_absolute()
    except Exception:
        return False


def _advertised_loader_id(info) -> Optional[str]:
    """The id to advertise for a scanned model: prefer a client-facing alias over
    an absolute filesystem path so /v1/models and the override key never expose a
    host path (the ./models and LM Studio scanners report the path as info.id)."""
    raw_id = getattr(info, "id", None)
    if not raw_id or not _is_abs_path_id(raw_id):
        return raw_id
    for alt in (getattr(info, "model_id", None), getattr(info, "display_name", None)):
        if alt and not _is_abs_path_id(alt):
            return alt
    # No clean alias: strip to a path-free public id so a host path is never advertised.
    return public_model_id(raw_id) or raw_id


def _resolve_load_dir(p):
    """The concrete dir holding the GGUFs. For an HF cache repo (``models--*``
    with ``snapshots/``) this is the latest snapshot dir, so /load takes the
    local branch instead of the download-capable repo-id branch."""
    from pathlib import Path

    try:
        if (p / "snapshots").is_dir():
            from routes.models import _resolve_hf_cache_realpath
            real = _resolve_hf_cache_realpath(p)
            if real:
                return Path(real)
    except Exception:
        pass
    return p


def _local_gguf_entry(loader_id: str, info) -> Optional[_LocalGgufEntry]:
    """Build an entry only when GGUF quants are on disk (not Transformers/
    safetensors), listing only on-disk quants. ``load_path`` is a concrete local
    path so /load resolves the variant locally and never fetches a remote one."""
    from pathlib import Path
    from utils.models.model_config import detect_gguf_model, list_local_gguf_variants

    path = getattr(info, "path", None)
    if not isinstance(path, str):
        return None
    p = Path(path)
    try:
        if p.is_file():
            # A standalone .gguf loads by its own path; no quant sub-selection. An
            # mmproj companion (vision/audio projector) is not a servable model on
            # its own: _scan_models_dir's standalone-file pass does not filter it
            # the way the directory scan does, so reject it here or /v1/models would
            # advertise a projector and a switch could load it instead of the weights,
            # evicting the loaded model. The directory branch below is already mmproj
            # free (list_local_gguf_variants drops mmproj quants).
            if p.suffix.lower() != ".gguf" or detect_gguf_model(str(p)) is None:
                return None
            return _LocalGgufEntry(loader_id, str(p), ())
        load_dir = _resolve_load_dir(p)
        variants, _ = list_local_gguf_variants(str(load_dir))
        quants = tuple(v.quant for v in variants if getattr(v, "quant", None))
        if not quants:
            return None
        # That call orders by descending size, so the head is the biggest quant (often
        # F16). Downstream reads [0], and a bare id must mean whichever quant a plain
        # load would take: answering with the largest can evict a model and then OOM.
        from core.inference.openai_auto_download import preferred_quant

        # Rank the ROOT checkpoints alone when there are any. A plain local load resolves
        # through non-recursive detect_gguf_model and so always takes the repo root, while
        # preferred_quant ranks on the key text and would hand a bare id an equally-good
        # ``distilled/...`` row that sorts earlier -- the same id serving different weights
        # depending on which resolver answered it. The qualified rows stay advertised; they
        # simply are not what a bare id means.
        unqualified = tuple(q for q in quants if "/" not in q)
        best = preferred_quant(unqualified or quants)
        if best and quants[0] != best:
            quants = (best, *(q for q in quants if q != best))
        return _LocalGgufEntry(loader_id, str(load_dir), quants)
    except Exception:
        return None


# A LoRA directory can carry a copied config.json and tokenizer beside these, and
# ModelConfig would then resolve its base model and fetch weights this resolver
# promises never to download.
_ADAPTER_MARKERS = ("adapter_config.json", "adapter_model.safetensors", "adapter_model.bin")
# The multimodal sub-configs the repo's own vision detector reads, which is what tells
# a served VLM apart from a plain seq2seq wearing the same architecture suffix.
_MULTIMODAL_CONFIG_KEYS = (
    "vision_config",
    "img_processor",
    "image_token_index",
    "projector_config",
    "audio_config",
)
_SUPPORTED_CONDITIONAL_AUDIO_MODEL_TYPES = frozenset({"csm", "whisper"})


def _read_json(path):
    """Parsed JSON for *path*, or None when it is absent or unreadable."""
    import json
    try:
        with path.open(encoding = "utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


def _host_serves_mlx() -> bool:
    """Whether this host's inference worker would pick MLXInferenceBackend.

    Reads the detected verdict without triggering detection, so an unknown device
    reads as "not MLX" and the entry is simply withheld until startup has decided.
    """
    try:
        from utils.hardware import hardware as hw
        return hw.DEVICE == hw.DeviceType.MLX
    except Exception:
        return False


def _host_has_a_non_gguf_backend() -> bool:
    """Whether this host can serve non-GGUF weights at all.

    The worker picks MLX on Apple Silicon and Transformers everywhere else, so a host
    with neither leaves the load to fail after the swap has unloaded the resident GGUF.
    An installed torch is not enough: the Transformers worker imports unsloth, whose
    get_device_type raises outright on a host with no NVIDIA, AMD or Intel accelerator,
    so a CPU-only or Vulkan-only box with a CPU wheel serves GGUF alone. Read the
    detected device rather than importing torch, which costs seconds on this path.
    """
    if _host_serves_mlx():
        return True
    try:
        from utils.hardware import hardware as hw

        if hw.DEVICE not in (hw.DeviceType.CUDA, hw.DeviceType.XPU):
            return False
        from importlib.util import find_spec

        return find_spec("torch") is not None
    except Exception:
        return False


def _has_safetensors_weights(load_dir) -> bool:
    """Whether safetensors weights sit under the names the loader looks for.

    Safetensors only: a .bin checkpoint is pickle-backed, and an API request must not
    execute one without an explicit load action. The loader receives no variant, so
    ``model.fp16.safetensors`` or a stray ``optimizer.safetensors`` is not weights it
    can open.
    """
    import re

    if (load_dir / "model.safetensors").is_file():
        return True
    if (load_dir / "model.safetensors.index.json").is_file():
        return True
    try:
        return any(
            re.fullmatch(r"model-\d+-of-\d+\.safetensors", f.name) for f in load_dir.iterdir()
        )
    except OSError:
        return False


def _is_generative_chat_config(config: dict) -> bool:
    """Whether a config.json describes a checkpoint the chat loader can generate with."""
    architectures = config.get("architectures")
    # model_type cannot stand in for the list: transformers' causal mapping lists bert and bart.
    if not isinstance(architectures, list) or not architectures:
        return False
    names = [name for name in architectures if isinstance(name, str)]
    # Not every causal checkpoint wears ForCausalLM: GPT2LMHeadModel and friends are
    # selected from model_type by the loader, and withholding them is the invisibility
    # this whole path exists to remove.
    if any(name.endswith(("ForCausalLM", "LMHeadModel")) for name in names):
        return True
    if not any(name.endswith("ForConditionalGeneration") for name in names):
        return False
    # ForConditionalGeneration is overloaded: T5 and BART wear it too, and the serving
    # path has no AutoModelForSeq2SeqLM branch, so require a multimodal sub-config.
    if any(key in config for key in _MULTIMODAL_CONFIG_KEYS):
        return True
    # whisper is the audio model rather than wearing one, so it carries no such sub-config.
    # the MLX worker refuses ASR and TTS outright, so only a Transformers host serves these.
    return not _host_serves_mlx() and _model_type_is_audio(config.get("model_type"))


def _model_type_is_audio(model_type) -> bool:
    """Whether *model_type* has a supported conditional audio serving path."""
    return isinstance(model_type, str) and model_type in _SUPPORTED_CONDITIONAL_AUDIO_MODEL_TYPES


def _weights_are_servable(load_dir) -> bool:
    """Whether *load_dir* holds weights the chat loader can be pointed at.

    On-disk shape only, nothing parsed from the config: safetensors weights, and none
    of the markers that make a directory something other than a chat model (a diffusers
    pipeline, a bare LoRA adapter, an embedder).
    """
    from routes.models import _local_pipeline_index

    if _local_pipeline_index(load_dir) or not (load_dir / "config.json").is_file():
        return False
    if not _has_safetensors_weights(load_dir):
        return False
    if any((load_dir / name).is_file() for name in _ADAPTER_MARKERS):
        return False
    # Same marker is_embedding_model reads for a local path, without its memo, which
    # would pin a verdict for the process from one scan.
    return not (load_dir / "modules.json").is_file()


def _config_is_servable_here(load_dir, config: dict) -> bool:
    """Whether *config* describes a chat model an API request may load on its own.

    Not whether the load will succeed: /v1/models advertises what this server has and
    could serve, and a load that then fails is an honest error to the caller. Only the
    approval boundary and the category are decided here.
    """
    from utils.security.remote_code_scan import REMOTE_CODE_CONFIG_FILES

    # trust_remote_code needs an approval fingerprint a switch has not got; read as data only.
    for name in REMOTE_CODE_CONFIG_FILES:
        candidate = config if name == "config.json" else _read_json(load_dir / name)
        # truthiness like the consent gate's _config_has_auto_map: an empty map runs nothing.
        if isinstance(candidate, dict) and candidate.get("auto_map"):
            return False
    return _is_generative_chat_config(config)


def _local_weights_entry(loader_id: str, info) -> Optional[_LocalGgufEntry]:
    """Build an entry for a local non-GGUF checkpoint (safetensors, MLX) the
    inference orchestrator can serve, else None.

    Requires a root ``config.json`` beside safetensors weights, so a bare LoRA adapter
    or a tokenizer-only snapshot is never offered as a model. Diffusers pipelines,
    embedders, custom-code repos and partial downloads are rejected too.
    """
    from pathlib import Path

    path = getattr(info, "path", None)
    if not isinstance(path, str) or getattr(info, "partial", False):
        return None
    # model-independent, so first: with no backend every row is withheld anyway.
    if not _host_has_a_non_gguf_backend():
        return None
    try:
        p = Path(path)
        if not p.is_dir():
            return None
        load_dir = _resolve_load_dir(p)
        if not _weights_are_servable(load_dir):
            return None
        config = _read_json(load_dir / "config.json")
        if not isinstance(config, dict) or not _config_is_servable_here(load_dir, config):
            return None
        # No quants: quantization is baked in, so there is no ":<quant>" to pin.
        return _LocalGgufEntry(loader_id, str(load_dir), (), is_gguf = False)
    except Exception:
        return None


def _local_servable_entry(loader_id: str, info) -> Optional[_LocalGgufEntry]:
    """Entry for whichever backend can serve *info* from disk, GGUF first."""
    return _local_gguf_entry(loader_id, info) or _local_weights_entry(loader_id, info)


def local_servable_model(info) -> Optional[tuple[bool, tuple[str, ...]]]:
    """``(is_gguf, on-disk quant labels)`` when this server can serve *info* straight
    from disk, else None. Non-GGUF weights carry no quant labels.

    Read from the files, not ``info.model_format``: the HF-cache scanner leaves that
    unset for GGUF snapshots, so filtering on it drops every cached GGUF. One scan
    tells /v1/models what it can serve and which quant to name.
    """
    from pathlib import Path

    path = getattr(info, "path", None)
    # Ollama-link entries come from a scanner _build_index intentionally skips (it
    # creates symlinks on the request path), so their advertised ids never resolve.
    # Don't report them as servable, or /v1/models would list unswitchable models.
    if isinstance(path, str) and any(
        seg in (".studio_links", "ollama_links") for seg in Path(path).parts
    ):
        return None
    entry = _local_servable_entry(getattr(info, "id", "") or "", info)
    return (entry.is_gguf, entry.variants) if entry is not None else None


def local_load_dir(path: Optional[str]) -> Optional[str]:
    """The concrete directory *path* loads from, or None when it names no directory.

    An HF cache repo resolves to its snapshot, which is what the orchestrator records as
    the active model, so a caller comparing a scanned catalog path against resident state
    is comparing the same string the loader used. Touches the filesystem.
    """
    from pathlib import Path

    if not isinstance(path, str) or not path:
        return None
    try:
        return str(_resolve_load_dir(Path(path)))
    except Exception:
        return path


def _build_index() -> dict[str, _LocalGgufEntry]:
    """Map normalized id/model_id/display_name -> local model entry.

    Scans the same roots Unsloth's model picker lists (./models, the active plus
    legacy/default HF caches, LM Studio dirs, and user scan folders) so a named
    local model is never missed and silently served as the loaded one. Ollama's
    scanner is skipped: it creates symlinks as a side effect and this runs on the
    request path.
    """
    # Lazy import: routes.models imports core.inference, so import at call time.
    from pathlib import Path
    from routes.models import (
        _scan_models_dir,
        _scan_hf_cache,
        _scan_lmstudio_dir,
        _resolve_hf_cache_dir,
        _is_hidden_model,
    )
    from utils.paths import legacy_hf_cache_dir, hf_default_cache_dir, lmstudio_model_dirs
    from utils.hf_cache_settings import known_hf_hub_caches
    from core.inference.model_ids import public_model_id

    index: dict[str, _LocalGgufEntry] = {}
    seen_hf: set[str] = set()

    try:
        active_root = str(Path(_resolve_hf_cache_dir()).resolve())
    except Exception:
        active_root = None

    def _scan_hf_once(directory) -> list:
        if directory is None:
            return []
        try:
            d = Path(directory)
            if not d.is_dir():
                return []
            rp = str(d.resolve())
            if rp in seen_hf:
                return []
            seen_hf.add(rp)
            # Only the active cache loads by repo id. Say so, or an inactive repo is
            # indexed under an id it cannot load by, and its snapshot basename (what
            # /v1/models advertises once loaded by path) is never a key at all.
            # No format classification here: nothing on this path reads model_format,
            # and its recursive walk would duplicate the one _local_gguf_entry already
            # does per snapshot, on the request path.
            return _scan_hf_cache(directory, active_cache = rp == active_root, classify_format = False)
        except Exception as exc:  # a missing/malformed root must skip, never crash the index
            logger.debug("auto-switch: skipping HF cache dir %r: %s", directory, exc)
            return []

    # Each source is guarded on its own so one bad root (a permission error, a
    # malformed cache) drops only that source, not the whole index.
    found: list = []
    try:
        found += _scan_models_dir(Path("./models").resolve())
    except Exception as exc:
        logger.debug("auto-switch: ./models scan failed: %s", exc)
    try:
        for hf_dir in (
            *known_hf_hub_caches(),
            _resolve_hf_cache_dir(),
            legacy_hf_cache_dir(),
            hf_default_cache_dir(),
        ):
            found += _scan_hf_once(hf_dir)
    except Exception as exc:
        logger.debug("auto-switch: HF cache scan failed: %s", exc)
    try:
        for lm_dir in lmstudio_model_dirs():
            found += _scan_lmstudio_dir(lm_dir)
    except Exception as exc:
        logger.debug("auto-switch: LM Studio scan failed: %s", exc)
    try:
        from storage.studio_db import list_scan_folders
        for folder in list_scan_folders():
            try:
                fp = Path(folder["path"])
                found += (
                    _scan_models_dir(fp, limit = 200) + _scan_hf_once(fp) + _scan_lmstudio_dir(fp)
                )
            except Exception as exc:
                logger.debug("auto-switch: scan folder %r failed: %s", folder, exc)
    except Exception as exc:
        logger.debug("auto-switch: scan folders enumerate failed: %s", exc)
    for info in found:
        raw_id = getattr(info, "id", None)
        if not raw_id:
            continue
        # Skip what Unsloth hides from its pickers (validation probe, RAG embed
        # weights): not chat models, so never an auto-switch target.
        if _is_hidden_model(
            raw_id,
            getattr(info, "model_id", None),
            getattr(info, "path", None),
        ):
            continue
        # Advertise a client-facing alias, not an absolute filesystem path.
        loader_id = _advertised_loader_id(info)
        entry = _local_servable_entry(loader_id, info)
        if entry is None:
            continue
        # Index every alias (including the path) so a client can resolve by any of
        # them, even though only the non-path loader_id is advertised.
        for key in (
            raw_id,
            getattr(info, "model_id", None),
            getattr(info, "display_name", None),
            public_model_id(raw_id),
        ):
            if key:
                index.setdefault(key.strip().lower(), entry)
        # Other revisions of the same repo resolve to their own weights, so a pin on
        # one keeps working after Hugging Face writes a newer snapshot.
        if entry.is_gguf:
            for name, sibling_entry in _sibling_revision_entries(raw_id, loader_id):
                index.setdefault(name.strip().lower(), sibling_entry)
    return index


def _sibling_revision_entries(raw_id: str, loader_id: str):
    """Yield ``(revision_name, entry)`` for the repo's OTHER cached revisions.

    An inactive-cache repo carries its snapshot path as the id, and /v1/models
    advertises only that directory's basename once loaded, so anything durable
    pinned to it (a subagent config) holds one revision hash. Hugging Face writes a
    new snapshot dir on every update, and the scan emits a single entry per repo
    pointed at the newest one, so that pin would otherwise stop resolving and drop
    through to whatever model is loaded.

    Each revision gets an entry for its OWN directory rather than an alias onto the
    scanned one: aliasing would redirect a pin that names an older complete revision
    onto a newer half-downloaded snapshot and break a request that works today.
    Incomplete revisions are skipped for the same reason.

    Sibling names are only revisions inside a real cache repo
    (``<root>/models--org--name/snapshots/<rev>``). A scan folder that merely happens
    to be called ``snapshots`` holds unrelated models, and treating those as
    revisions would silently serve one model in place of another.

    GGUF only: ``snapshot_variants_all_complete`` reports a revision offering no quants
    as incomplete, so a non-GGUF repo pins to its scanned revision alone.
    """
    from pathlib import Path
    from types import SimpleNamespace

    snapshots = Path(raw_id).parent
    if snapshots.name != "snapshots" or not snapshots.parent.name.startswith("models--"):
        return
    from routes.models import snapshot_variants_all_complete

    try:
        siblings = [p for p in snapshots.iterdir() if p.is_dir() and p.name != Path(raw_id).name]
    except OSError:
        return
    for sibling in siblings:
        if not snapshot_variants_all_complete(str(sibling)):
            continue
        entry = _local_gguf_entry(loader_id, SimpleNamespace(path = str(sibling)))
        if entry is not None:
            yield sibling.name, entry


def note_downloaded(repo_id: Optional[str]) -> None:
    """Record a repo as present ahead of the scan that will index it."""
    if not repo_id:
        return
    with _lock:
        _just_downloaded.add(repo_id.strip().lower())


def recently_downloaded(repo_id: str) -> bool:
    """Whether *repo_id* finished downloading since the last completed scan."""
    if not isinstance(repo_id, str) or not repo_id.strip():
        return False
    return repo_id.strip().lower() in _just_downloaded


def _snapshot_is_trusted(timestamp: float, now: float) -> bool:
    """Whether a snapshot stamped *timestamp* may answer a model switch at *now*.

    Positive is an ordinary scan, trusted for the TTL. Negative is when an
    additions-only download invalidated it, trusted only while its rebuild could
    still be running: one that keeps failing must not leave entries trusted forever,
    or a model deleted on disk could still trigger a switch. Zero is revoked.
    """
    if timestamp > 0.0:
        return now - timestamp < _CACHE_TTL_S
    if timestamp < 0.0:
        return now + timestamp < max(_CACHE_TTL_S, _last_scan_s * _WARM_DUTY)
    return False


def invalidate_index(*, additions_only: bool = False) -> None:
    """Mark the cached scan stale.

    Entries stay available so an additions-only download invalidation can keep
    serving known positive hits while its background rebuild adds the new model.
    Other invalidations retain the allocation but revoke that trust, since a scan
    root may have been removed. Ordinary TTL expiry is likewise not additions-only.
    """
    global _scan, _warm_pending
    with _lock:
        now = time.monotonic()
        timestamp, retained = _scan
        # Publish entries and their trust state together. A lock-free reader sees
        # either the complete old snapshot or the complete invalidated one, never a
        # fresh timestamp paired with already-revoked trust.
        stamp = -now if additions_only and _snapshot_is_trusted(timestamp, now) else 0.0
        _scan = (stamp, retained)
    # This may have waited out a scan on _lock, so the warmer that just published can
    # still own the slot with a snapshot that is stale again. See _warm_pending.
    with _warm_lock:
        if _warming:
            _warm_pending = True


def _index() -> dict[str, _LocalGgufEntry]:
    global _scan
    # Build under the lock so concurrent callers with an expired cache don't all
    # run the (multi-dir) scan at once; the rest wait and reuse the fresh result.
    with _lock:
        now = time.monotonic()
        ts, cached = _scan
        # ``ts > 0``: monotonic() counts from boot, so under a TTL of uptime an
        # invalidated stamp reads as recent and would serve what was just revoked.
        if ts > 0.0 and now - ts < _CACHE_TTL_S:
            return cached
        fresh = _build_index()
        # Stamp AFTER the scan, not with the pre-scan ``now``: a multi-root scan on
        # an install with many local models can itself exceed the TTL, which would
        # store the cache already expired and make every request rebuild the index.
        _scan = (time.monotonic(), fresh)
        # The scan supersedes the notes: whatever landed is in the index now.
        _just_downloaded.clear()
        return fresh


def index_is_built() -> bool:
    """Whether a scan has ever completed, freshness aside.

    Lock-free on purpose: ``_lock`` is held for the whole scan, so taking it would
    park the request path on the scan it is trying to stay off. Safe because
    ``_scan`` is only ever rebound, never mutated.
    """
    return _scan[0] > 0.0


def resolve_trusted_cached_local_gguf(requested: str) -> Optional[tuple[str, Optional[str], str]]:
    """Resolve a positive cache hit only when its snapshot is safe to trust.

    A snapshot is trustworthy while fresh, or after an explicit additions-only
    invalidation. A positive hit from ordinary TTL expiry or a scan-root change
    must be rebuilt before it can trigger a model switch. The identity checks close
    the race where invalidation publishes a different snapshot during resolution or
    while the trust state is being evaluated.
    """
    snapshot = _scan
    resolved = _resolve_from_index(requested, snapshot[1])
    if resolved is None or _scan is not snapshot:
        return None
    trusted = _snapshot_is_trusted(snapshot[0], time.monotonic())
    return resolved if trusted and _scan is snapshot else None


def warm_index_soon() -> None:
    """(Re)build the index off the request path when it is missing or past its TTL.

    The only refresh for callers using ``allow_scan=False``. Covers a stale index,
    not just an absent one: a model downloaded through the Hub UI or dropped into a
    scan folder has no invalidation hook and would otherwise stay invisible to them
    for the life of the process. Never blocks, and never touches ``_lock``.
    """
    global _warming, _warm_pending
    stamp = _scan[0]
    if stamp > 0.0 and time.monotonic() - stamp < max(_CACHE_TTL_S, _last_scan_s * _WARM_DUTY):
        return
    with _warm_lock:
        if _warming:
            return
        _warming = True
        _warm_pending = False

    def _run() -> None:
        global _warming, _warm_pending, _last_scan_s
        released = False
        try:
            while True:
                started = time.monotonic()
                try:
                    _index()
                except Exception:
                    pass
                _last_scan_s = time.monotonic() - started
                with _warm_lock:
                    if _warm_pending:
                        _warm_pending = False
                        continue
                    _warming, released = False, True
                    return
        finally:
            # Only on a BaseException: leaving the slot held would kill background
            # warming for the life of the process and put scans back on requests.
            if not released:
                with _warm_lock:
                    _warming = _warm_pending = False

    threading.Thread(target = _run, name = "local-model-index-warm", daemon = True).start()


def resolve_local_gguf(
    requested: str, *, allow_scan: bool = True
) -> Optional[tuple[str, Optional[str], str]]:
    """Return ``(load_path, gguf_variant, loader_id)`` for a local match, else None.

    ``load_path`` is the concrete on-disk path to hand /load (so it never fetches
    a remote), ``loader_id`` is the advertised id used as the launch-override key.
    ``gguf_variant`` is None for a non-GGUF checkpoint, which has no quant to pin.
    ``requested`` is ``repo`` or ``repo:VARIANT``. An exact id match wins first
    (so ids containing a colon still resolve); else the last ``:VARIANT`` is split
    off and resolves only when that quant is on disk, unless it names no quant at
    all (an Ollama-style ":latest"), which means the repo.

    ``allow_scan=False`` answers from the last built index and never rebuilds. It is
    a raw snapshot read for callers that separately decide whether the snapshot is
    trustworthy; use :func:`resolve_trusted_cached_local_gguf` for model switching.
    """
    if not isinstance(requested, str) or not requested.strip():
        return None
    requested = requested.strip()
    try:
        index = _index() if allow_scan else _scan[1]
        return _resolve_from_index(requested, index)
    except Exception:
        # Best-effort: any resolver failure falls through to the loaded model,
        # so a malformed name can never turn a servable request into a 500.
        return None


def _resolve_from_index(
    requested: str, index: dict[str, _LocalGgufEntry]
) -> Optional[tuple[str, Optional[str], str]]:
    """Resolve *requested* against one immutable published index mapping."""
    try:
        entry = index.get(requested.lower())
        if entry is not None:
            variant = entry.variants[0] if entry.variants else None
            return entry.load_path, variant, entry.loader_id

        base, sep, variant = requested.rpartition(":")
        if not sep:
            return None
        entry = index.get(base.strip().lower())
        if entry is None:
            return None
        wanted = variant.strip().lower()
        for v in entry.variants:
            if v.lower() == wanted:
                return entry.load_path, v, entry.loader_id
        from core.inference.openai_auto_download import looks_like_quant

        if looks_like_quant(variant):
            return None
        # ":latest" or ":8b" names no file, so it means the repo; a real quant that
        # is not on disk still misses, or a swap would serve the wrong weights.
        return entry.load_path, (entry.variants[0] if entry.variants else None), entry.loader_id
    except Exception:
        return None


def local_target_is_gguf(load_path: Optional[str], loader_id: Optional[str] = None) -> bool:
    """Whether an auto-switch target is served by llama.cpp rather than the orchestrator.

    Reads the weights the load will actually open, so a rebuilt index that no longer
    carries the entry cannot flip the answer mid-switch. Falls back to the indexed
    entry, then to True, which is what the callers need for a target that is not a
    concrete path: llama.cpp is the only backend auto-switch used before non-GGUF
    support, and the idle-unload reload stash only ever holds a freed GGUF.

    Touches the filesystem, so call it off the event loop.
    """
    from pathlib import Path
    from types import SimpleNamespace

    if isinstance(load_path, str) and load_path:
        try:
            if Path(load_path).exists():
                return _local_gguf_entry("", SimpleNamespace(path = load_path)) is not None
        except OSError:
            pass
    if not isinstance(loader_id, str) or not loader_id.strip():
        return True
    entry = _scan[1].get(loader_id.strip().lower())
    return entry.is_gguf if entry is not None else True


MISS_MODEL_NOT_FOUND = "model_not_found"
MISS_VARIANT_NOT_FOUND = "variant_not_found"


def describe_local_miss(requested: str) -> tuple[str, tuple[str, ...]]:
    """Why :func:`resolve_local_gguf` missed, so an error can say "wrong quant"
    instead of "no such model".

    ``(MISS_VARIANT_NOT_FOUND, <local quants>)`` when the repo is downloaded but the
    requested ``:VARIANT`` is not, else ``(MISS_MODEL_NOT_FOUND, ())``. Fail-safe: a
    scan failure reports the generic miss rather than raising into the handler.
    """
    if not isinstance(requested, str) or not requested.strip():
        return MISS_MODEL_NOT_FOUND, ()
    base, sep, variant = requested.strip().rpartition(":")
    from core.inference.openai_auto_download import looks_like_quant

    # Split like the resolver or the two disagree: a tag naming no quant means the
    # repo there, so reporting a missing quant for it would name one nobody asked for.
    if not sep or not looks_like_quant(variant):
        return MISS_MODEL_NOT_FOUND, ()
    try:
        entry = _index().get(base.strip().lower())
    except Exception:
        return MISS_MODEL_NOT_FOUND, ()
    if entry is None or not entry.variants:
        return MISS_MODEL_NOT_FOUND, ()
    return MISS_VARIANT_NOT_FOUND, entry.variants
