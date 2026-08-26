# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Malware / unsafe-file gate for model loads.

The ``trust_remote_code`` consent gate covers the ``auto_map`` Python vector; this
covers the other one -- a malicious pickle inside a weight file, which executes
during ``from_pretrained`` deserialization even with ``trust_remote_code=False``.
It reads Hugging Face's OWN scan (picklescan + ClamAV) via
``model_info(securityStatus=True).security_repo_status``. METADATA-ONLY: it never
downloads, opens, or unpickles the flagged files.

Policy:
  * Hard block, non-approvable.
  * Block whenever ``filesWithIssues`` lists a non-``safe`` level, regardless of
    ``scansDone`` (often false even for clean repos). Unknown/future levels fail
    CLOSED (block) so Hub schema drift cannot silently allow a bad verdict; only a
    small allowlist of clean / not-yet-scanned levels is non-blocking. An unavailable
    status falls back to local inspection for a pinned cache snapshot and otherwise
    fails open.
  * Scope to the load-path RCE vector: a root-level (or load-subdir-level),
    code-executing file. Inert formats (safetensors / gguf / config / text) and
    subdirectory pickles that no root weight-index references are NOT loaded, so
    they do not block; an index-referenced shard does, wherever it lives. This
    blocks real malware (eicar's root ``*.pkl``/``*.dat``) without false-blocking
    repos like ``nvidia/Nemotron-H-8B-Base-8K`` (flagged NeMo pickles under
    ``nemo/`` that no index lists).
  * No first-party exemption (scoping is by load path/format, not org).
  * Local paths are skipped (no Hub scan); a remote ``*.gguf``-named repo is still
    scanned so a repo cannot dodge the gate by suffixing its name.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from loggers import get_logger

logger = get_logger(__name__)

# Pickle-format weight files (plain or sharded) that execute code on load; safetensors/gguf are
# inert. Grouped by weight family so an inert safetensors only suppresses the pickle it replaces.
_PICKLE_WEIGHT_RE = re.compile(
    r"^(model|pytorch_model|adapter_model|consolidated)(-\d+-of-\d+)?"
    r"\.(bin|pt|pth|ckpt|pkl|pickle)$",
    re.IGNORECASE,
)

# Non-blocking levels: clean or not-yet-finished. Anything else blocks, so schema drift fails CLOSED.
_NONBLOCKING_LEVELS = frozenset(
    {"", "safe", "pending", "scanning", "queued", "unscanned", "error", "unknown", "none"}
)

# Suffixes that cannot execute code on load (safetensors, gguf, text/markup/images).
_INERT_SUFFIXES = frozenset(
    {
        ".safetensors",
        ".gguf",
        ".json",
        ".txt",
        ".md",
        ".rst",
        ".yaml",
        ".yml",
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".webp",
        ".svg",
        ".bmp",
        ".gitattributes",
        ".gitignore",
    }
)

# Source files are not deserialized by a weight load; executable repo code runs only via
# auto_map, the consent gate's domain, else a flagged helper/train script would false-block.
_SOURCE_SUFFIXES = frozenset({".py", ".pyc", ".pyx", ".pyi"})


# Torch-family weight indexes: from_pretrained feeds each shard they name to load_state_dict,
# which torch.load()s (pickle) any shard not ending in .safetensors. A pytorch index is
# superseded by a base safetensors; a safetensors index IS the chosen archive. tf/flax are inert.
_TORCH_INDEX_FILES = ("pytorch_model.bin.index.json", "model.safetensors.index.json")

# Root weight-index files: a flagged subdir pickle is a load vector iff a root index names it.
_TRANSFORMERS_INDEX_FILES = (
    "pytorch_model.bin.index.json",
    "model.safetensors.index.json",
    "tf_model.h5.index.json",
    "flax_model.msgpack.index.json",
)


def _normalize_repo_path(path: str) -> str:
    """Strip ``./`` prefixes and normalize separators for repo-relative comparison."""
    p = (path or "").strip().replace("\\", "/")
    while p.startswith("./"):
        p = p[2:]
    return p


def _file_suffix(path: str) -> str:
    """Lowercase ``.ext`` of the basename, or ``""`` if none."""
    base = _normalize_repo_path(path).rsplit("/", 1)[-1]
    return "." + base.rsplit(".", 1)[1].lower() if "." in base else ""


def _hf_cache_snapshot_ref(local_path: str) -> Optional[tuple[str, str, Path]]:
    """Return provenance for an HF-cache snapshot path, else ``None``.

    An inactive Unsloth cache loads by its snapshot path but keeps the
    ``models--org--repo/snapshots/<rev>`` layout, so the gate recovers its provenance
    and scans that exact commit instead of exempting it.
    """
    try:
        path = Path(local_path).resolve(strict = False)
    except (OSError, ValueError):
        return None
    for parent in path.parents:
        if parent.name != "snapshots":
            continue
        encoded = parent.parent.name
        if not encoded.startswith("models--"):
            return None
        repo_id = encoded.removeprefix("models--").replace("--", "/")
        if not repo_id:
            return None
        revision = path.relative_to(parent).parts[0]
        return repo_id, revision, parent / revision
    return None


def _load_relative_path(norm: str, load_subdirs) -> str:
    """``norm`` relative to a ``from_pretrained`` load root. Some loads read from a
    snapshot SUBDIRECTORY (Spark-TTS / BiCodec load ``<snapshot>/LLM``), where a file
    directly under the subdir is root-level, not nested. Strips the matching load-subdir
    prefix, or returns ``norm`` unchanged when it is not under one.
    """
    for subdir in load_subdirs or ():
        prefix = _normalize_repo_path(subdir).strip("/")
        if prefix and norm.startswith(prefix + "/"):
            return norm[len(prefix) + 1 :]
    return norm


def _index_prefixes(load_subdirs) -> tuple:
    """Prefixes to look for weight-index files under: repo root plus each load subdir."""
    prefixes = [""]
    for subdir in load_subdirs or ():
        p = _normalize_repo_path(subdir).strip("/")
        if p:
            prefixes.append(p + "/")
    return tuple(prefixes)


def _indexed_shard_paths(
    model_name: str,
    hf_token: Optional[str],
    load_subdirs = (),
    revision: Optional[str] = None,
):
    """Repo-relative weight paths a load could fetch via weight-index files. Returns a
    set (empty when the repo ships no index files -- a definitive "nothing sharded"), or
    None when the lookup was inconclusive (transient error) so the caller treats a
    flagged subdir pickle conservatively. Reads only small JSON indexes, never weights.
    Indexes are looked up at the root and each ``load_subdirs`` root, with ``weight_map``
    entries re-prefixed to repo-relative paths. ``revision`` scopes to a cached commit.
    """
    import json

    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.utils import EntryNotFoundError
        from utils.hf_cache_settings import active_hf_hub_cache
        from utils.hf_probe import hf_file_definitely_absent
    except Exception:
        return None

    paths: set = set()
    inconclusive = False
    for prefix in _index_prefixes(load_subdirs):
        for filename in _TRANSFORMERS_INDEX_FILES:
            # Most repos are unsharded, so avoid caching expected 404s for optional indexes.
            if hf_file_definitely_absent(
                model_name, prefix + filename, revision = revision, token = hf_token or None
            ):
                continue
            try:
                index_path = hf_hub_download(
                    model_name,
                    prefix + filename,
                    revision = revision,
                    token = hf_token or None,
                    cache_dir = active_hf_hub_cache(),
                )
            except EntryNotFoundError:
                continue  # definitively absent, not an error
            except Exception:
                inconclusive = True  # transient: an index that might exist could not be read
                continue
            try:
                weight_map = (json.loads(open(index_path, encoding = "utf-8-sig").read()) or {}).get(
                    "weight_map"
                ) or {}
                for shard in weight_map.values():
                    shard_norm = _normalize_repo_path(str(shard))
                    # weight_map paths are relative to the index file's directory.
                    if prefix and not shard_norm.startswith(prefix):
                        shard_norm = prefix + shard_norm
                    paths.add(shard_norm)
            except Exception:
                inconclusive = True
    # Any transient failure -> inconclusive (the shard could be listed only by the index we could
    # not read), so fail closed. No index files -> empty set, a definitive "nothing sharded".
    if inconclusive:
        return None
    return paths


# Two-timeout metadata fetch, mirroring hub.workers.hf_download._retry_metadata_fetch.
_REQUEST_TIMEOUT = 10.0
_RETRY_TIMEOUT = 20.0


@dataclass
class FileSecurityDecision:
    """Outcome of the Hub security scan for one model repo."""

    model_name: str
    blocked: bool
    unsafe_files: list = field(default_factory = list)  # [{"path", "level"}]
    reason: str = ""

    def response_payload(self) -> dict:
        """Machine-readable detail merged into the preflight payload the dialog reads."""
        return {
            "unsafe_files": self.unsafe_files,
            "security_blocked": self.blocked,
            "reason": self.reason,
        }


def security_load_subdirs(
    model_name: str,
    hf_token: Optional[str] = None,
    local_files_only: bool = False,
) -> tuple:
    """Snapshot subdirectories a load calls ``from_pretrained`` on, for scoping the scan.
    Most models load from the root (``()``); Spark-TTS / BiCodec load ``<snapshot>/LLM``,
    so ``LLM/`` is a load root for them. Metadata-only (tokenizer special tokens), cached.

    ``local_files_only`` skips the remote tokenizer fetch. Callers deciding whether a
    cache already on disk is usable must pass it: that work is meant to be pure
    filesystem, and a hung hub would otherwise block local snapshot resolution.
    """
    try:
        from utils.models.model_config import detect_audio_type, load_model_defaults
        if (
            detect_audio_type(model_name, hf_token = hf_token, local_files_only = local_files_only)
            == "bicodec"
        ):
            return ("LLM",)
        # Tokenizer detection can fail (network/gated/unresolved alias); the YAML default also pins the
        # audio type, so fall back to it, else a flagged LLM/ pickle reads as an ignored subdir artifact.
        if (load_model_defaults(model_name) or {}).get("audio_type") == "bicodec":
            return ("LLM",)
    except Exception:
        pass
    return ()


def load_scan_target(model_name: str, load_subdirs: tuple) -> tuple:
    """Map a load alias to the ``(repo_id, load_subdirs)`` the load actually fetches. The
    Spark-TTS / BiCodec alias ``<parent>/LLM`` is downloaded by the trainer as
    ``unsloth/<parent>`` and loaded from ``LLM/``, so scan that repo with ``LLM`` as a
    load root (the literal alias 404s and fails open). Everything else is unchanged.
    """
    try:
        from utils.paths import is_local_path
        if is_local_path(model_name):
            return model_name, load_subdirs
    except Exception:
        return model_name, load_subdirs
    name = (model_name or "").strip().strip("/")
    # Rewrite ONLY a registry-known bicodec alias: "evil/LLM" would scan unsloth/evil and fail open.
    if name.endswith("/LLM") and name.count("/") == 1:
        try:
            from utils.models.model_config import load_model_defaults
            if (load_model_defaults(name) or {}).get("audio_type") == "bicodec":
                parent = name[: -len("/LLM")]
                return f"unsloth/{parent}", tuple(dict.fromkeys((*load_subdirs, "LLM")))
        except Exception:
            pass
    return model_name, load_subdirs


def _fetch_security_status(
    model_name: str,
    hf_token: Optional[str],
    revision: Optional[str] = None,
):
    """``security_repo_status`` (a dict) or None if unavailable. Hub metadata only;
    retries once on a transient error, then returns None so the caller can apply its
    local-fallback policy. ``revision`` scopes the scan to a specific cached commit
    (else the default branch).
    """
    from huggingface_hub import model_info as hf_model_info

    token_arg = hf_token if hf_token else False
    last_exc = None
    for attempt, timeout in enumerate((_REQUEST_TIMEOUT, _RETRY_TIMEOUT)):
        try:
            info = hf_model_info(
                model_name,
                revision = revision,
                token = token_arg,
                securityStatus = True,
                timeout = timeout,
            )
            return getattr(info, "security_repo_status", None)
        except Exception as exc:  # network/offline/gated/404/unsupported-client
            last_exc = exc
            if attempt == 0:
                continue
    logger.debug(
        "HF security scan unavailable for '%s' (%s).",
        model_name,
        type(last_exc).__name__ if last_exc else "unknown",
    )
    return None


def _st_load_roots(snapshot: Path, load_subdirs = ()) -> list:
    """Directories a SentenceTransformer load deserializes weights from: the snapshot root plus
    each module path in modules.json and each explicit ``from_pretrained`` load subdirectory.
    Local, no network. Mirrors the online gate (which ignores unreferenced nested pickles the
    loader never opens) so the offline gate doesn't over-block."""
    roots = [snapshot]

    def _add_repo_root(raw_path: Any) -> None:
        path = _normalize_repo_path(str(raw_path)).strip("/")
        if not path or ".." in path.split("/"):
            return
        candidate = snapshot.joinpath(*path.split("/"))
        if candidate not in roots:
            roots.append(candidate)

    for subdir in load_subdirs or ():
        _add_repo_root(subdir)
    try:
        import json
        modules = json.loads((snapshot / "modules.json").read_text(encoding = "utf-8-sig"))
    except (OSError, ValueError):
        return roots  # no / invalid modules.json -> only declared load roots apply
    for module in modules if isinstance(modules, list) else ():
        if not isinstance(module, dict):
            continue
        path = str(module.get("path", "")).strip()
        native_path = Path(path)
        if not path:
            continue
        if (
            native_path.is_absolute()
            or native_path.drive
            or native_path.root
            or ".." in native_path.parts
        ):
            raise OSError("unsafe SentenceTransformer module path")
        candidate = snapshot / native_path
        if candidate not in roots:
            roots.append(candidate)
    return roots


def _snapshot_module_subdirs(snapshot: Path) -> tuple[str, ...]:
    return tuple(root.relative_to(snapshot).as_posix() for root in _st_load_roots(snapshot)[1:])


def _indexed_pickle_shards(index_path: Path, root: Path, snapshot: Path) -> list:
    """Shards a torch weight index points a ``from_pretrained`` load at that load_state_dict would
    torch.load (pickle): every ``weight_map`` target NOT ending in ``.safetensors``, whatever its
    stem (an arbitrary name like ``shards/payload`` still deserializes). Resolved relative to the
    index dir (``root``) like the loader, so a shard in a nested dir is followed (iterdir misses it).
    Lexical only, never ``Path.resolve()`` (HF snapshot files symlink into ``blobs/``, so resolving
    escapes the snapshot and false-blocks every shard). Raises OSError -> caller fails CLOSED on an
    unreadable/invalid index or a target escaping the snapshot."""
    import json
    import os

    try:
        # JSON is UTF-8 by spec; pin it so a non-ASCII index is not misdecoded under Windows' cp1252.
        parsed = json.loads(index_path.read_text(encoding = "utf-8-sig"))
    except (OSError, ValueError) as exc:
        raise OSError(f"unreadable weight index: {index_path}") from exc
    weight_map = parsed.get("weight_map") if isinstance(parsed, dict) else None
    if not isinstance(weight_map, dict):
        return []  # no dict weight_map -> the loader resolves no shards from this index
    snapshot_norm = os.path.normpath(str(snapshot))
    shards = []
    for shard in weight_map.values():
        raw = str(shard)
        if not raw:
            continue
        # Join the RAW weight_map value like from_pretrained's os.path.join: on POSIX a backslash is a
        # literal filename char, so normalizing it would probe a different path than the loader opens.
        joined = os.path.normpath(os.path.join(str(root), raw))
        if joined != snapshot_norm and not joined.startswith(snapshot_norm + os.sep):
            raise OSError(f"weight index escapes the snapshot: {index_path}")
        shard_path = Path(joined)
        # Case-SENSITIVE, mirroring load_state_dict's own endswith(".safetensors"): payload.SAFETENSORS falls to torch.load.
        if not shard_path.name.endswith(".safetensors") and shard_path.is_file():
            shards.append(shard_path)
    return shards


def _loader_resolves(root: Path, name: str) -> bool:
    """True iff from_pretrained would open ``name`` under ``root``. ``is_file()`` honors the platform
    (case-sensitive on Linux, case-insensitive on Windows/macOS), so it mirrors the loader's own
    lookup: an oddly-cased decoy counts as an alternative only where the loader would truly open it.
    A name-fold instead would let an uppercase MODEL.SAFETENSORS suppress the scan on Linux while the
    loader, asking for the canonical lowercase name, silently falls through to a pickle index."""
    return (root / name).is_file()


def _cached_pickle_weight_files(snapshot: Path, load_subdirs = ()) -> list:
    """Pickle weight files a SentenceTransformer/Transformers load deserializes from snapshot's ST
    load roots, EXCLUDING those whose weight family also ships an inert safetensors in the same dir
    (the loader prefers it): a base pickle is suppressed only by a base model.safetensors, an adapter
    pickle only by adapter_model.safetensors -- an unrelated safetensors is no substitute. Covers
    both direct-child pickles AND pickle shards referenced by a local weight index (which the loader
    follows into nested dirs, matching the online gate). Raises OSError -- caller fails CLOSED -- if
    the snapshot root or a weight index is unreadable, or an index reference escapes the snapshot."""
    blocked = []
    seen = set()

    def _add(path: Path):
        key = str(path)
        if key not in seen:
            seen.add(key)
            blocked.append(path)

    for root in _st_load_roots(snapshot, load_subdirs):
        try:
            entries = [p for p in root.iterdir() if p.is_file()]
        except OSError:
            if root == snapshot:
                raise  # top-level unreadable -> fail closed
            continue  # unreadable module subdir: nothing loadable to attest here
        # Safetensors alternatives the loader would actually resolve (never a bare name-fold, which fails
        # OPEN). A base pickle is replaced only by a base safetensors; model.safetensors outranks both indexes.
        has_direct_base_safetensors = _loader_resolves(root, "model.safetensors")
        has_base_safetensors = has_direct_base_safetensors or _loader_resolves(
            root, "model.safetensors.index.json"
        )
        has_adapter_safetensors = _loader_resolves(root, "adapter_model.safetensors")
        for path in entries:
            if not _PICKLE_WEIGHT_RE.match(path.name):
                continue
            is_adapter = path.name.lower().startswith("adapter_model")
            has_alternative = has_adapter_safetensors if is_adapter else has_base_safetensors
            if not has_alternative:
                _add(path)
        # A torch weight index makes from_pretrained load nested shards iterdir never sees, torch.loading
        # any not ending in .safetensors. Probe the canonical index name with the loader's own lookup so
        # an oddly-cased artifact it would never open does not block. model.safetensors wins over both.
        for index_name in _TORCH_INDEX_FILES:
            if not _loader_resolves(root, index_name):
                continue
            if has_direct_base_safetensors:
                continue
            if index_name == "pytorch_model.bin.index.json" and has_base_safetensors:
                continue
            for shard_path in _indexed_pickle_shards(root / index_name, root, snapshot):
                _add(shard_path)
    return blocked


def _evaluate_local_snapshot(
    model_name: str,
    snapshot_path: Optional[Path] = None,
    *,
    context: str,
    load_subdirs = (),
) -> FileSecurityDecision:
    """Inspect cached weights when the Hub scan cannot be used."""
    if snapshot_path is None:
        from utils.utils import hf_cache_snapshot_dir
        try:
            snapshot = hf_cache_snapshot_dir(model_name)
        except Exception:
            logger.warning(
                "Local security fallback (%s): could not resolve the cache for '%s'; blocking.",
                context,
                model_name,
            )
            return FileSecurityDecision(
                model_name,
                True,
                reason = f"{context}; could not inspect the local cache",
            )
    else:
        try:
            snapshot = snapshot_path.resolve(strict = True)
            if not snapshot.is_dir():
                raise OSError("snapshot is not a directory")
        except (OSError, RuntimeError, ValueError):
            logger.warning(
                "Local security fallback (%s): could not resolve selected snapshot for "
                "'%s'; blocking.",
                context,
                model_name,
            )
            return FileSecurityDecision(
                model_name,
                True,
                reason = f"{context}; could not inspect the selected model snapshot",
            )

    if snapshot is None:
        return FileSecurityDecision(
            model_name,
            False,
            reason = f"{context}; nothing cached to load",
        )

    try:
        pickles = _cached_pickle_weight_files(snapshot, load_subdirs)
    except OSError:
        logger.warning(
            "Local security fallback (%s): could not read the cache for '%s'; blocking.",
            context,
            model_name,
        )
        return FileSecurityDecision(
            model_name,
            True,
            reason = f"{context}; could not read the local cache",
        )

    if not pickles:
        return FileSecurityDecision(
            model_name,
            False,
            reason = f"{context}; cached weights are inert (safetensors/gguf)",
        )

    # Snapshot-relative posix paths (match the online gate; disambiguate same-named pickles).
    rel_paths = sorted(p.relative_to(snapshot).as_posix() for p in pickles)
    names = ", ".join(rel_paths)
    logger.warning(
        "Blocking load of '%s': cached pickle weight(s) could not be malware-scanned "
        "(%s) and have no safetensors alternative (%s).",
        model_name,
        context,
        names,
    )
    return FileSecurityDecision(
        model_name,
        True,
        unsafe_files = [{"path": rel, "level": "unscanned"} for rel in rel_paths],
        reason = (f"{context}; unscanned pickle weights with no safetensors alternative: {names}"),
    )


def evaluate_file_security(
    model_name: str,
    hf_token: Optional[str] = None,
    *,
    load_subdirs = (),
    local_only_load: bool = False,
) -> FileSecurityDecision:
    """Block a load when HF's security scan flags unsafe serialized files.

    Call UNCONDITIONALLY before any load (independent of trust_remote_code): a malicious
    pickle deserializes during ``from_pretrained`` regardless. Metadata-only; when the
    scan is unavailable, exact cached snapshots receive a local fail-closed inspection
    while unresolved remote refs remain fail-open.

    ``load_subdirs`` names subdirs the load calls ``from_pretrained`` on (e.g. ``("LLM",)``
    for Spark-TTS / BiCodec, loading ``<snapshot>/LLM``): a flagged file directly under one
    is root-level there and blocks, and an index inside it is honored when scoping shards.

    ``local_only_load`` marks an offline load and skips the Hub request.
    """
    # Scan the repo the load actually fetches, not the literal alias (which 404s and fails open).
    model_name, load_subdirs = load_scan_target(model_name, tuple(load_subdirs))

    # Local paths have no Hub scan, EXCEPT an HF-cache snapshot whose canonical path encodes a repo
    # id + commit: scan that exact commit so an inactive-cache load can't dodge the gate.
    snapshot_revision = None
    selected_snapshot = None
    try:
        from utils.paths import is_local_path
        if is_local_path(model_name):
            cache_ref = _hf_cache_snapshot_ref(model_name)
            if cache_ref is None:
                return FileSecurityDecision(model_name, False, reason = "local path; no Hub scan")
            model_name, snapshot_revision, selected_snapshot = cache_ref
    except Exception:
        # Cannot classify the path -> do not block on that account.
        return FileSecurityDecision(model_name, False, reason = "path check failed; not blocked")

    # Offline: inspect the local cache and fail closed rather than hang on model_info or fail open.
    if local_only_load:
        return _evaluate_local_snapshot(
            model_name,
            selected_snapshot,
            context = "offline",
            load_subdirs = load_subdirs,
        )

    status = _fetch_security_status(model_name, hf_token, revision = snapshot_revision)
    if not isinstance(status, dict):
        if selected_snapshot is not None:
            return _evaluate_local_snapshot(
                model_name,
                selected_snapshot,
                context = "Hub scan unavailable",
                load_subdirs = load_subdirs,
            )
        return FileSecurityDecision(
            model_name, False, reason = "scan unavailable; allowed (fail-open)"
        )
    if selected_snapshot is not None:
        try:
            load_subdirs = tuple(
                dict.fromkeys((*load_subdirs, *_snapshot_module_subdirs(selected_snapshot)))
            )
        except OSError:
            return _evaluate_local_snapshot(
                model_name,
                selected_snapshot,
                context = "invalid cached model metadata",
                load_subdirs = load_subdirs,
            )

    # Block a non-``safe`` flagged file scoped to the load-path RCE vector (root-level,
    # code-executing). Not gated on ``scansDone`` (often false even when clean). Unknown levels fail
    # closed. Subdir pickles and inert formats are not loaded by from_pretrained and do not block; an
    # unavailable status is fail-open only for an unresolved remote ref.
    unsafe = []
    skipped = []  # flagged, but not a load-path RCE vector (subdir artifact / inert)
    maybe_shard = []  # flagged subdir pickle: a load vector ONLY if a root index lists it
    for entry in status.get("filesWithIssues") or []:
        if not isinstance(entry, dict):
            continue
        level = str(entry.get("level", "")).lower()
        if level in _NONBLOCKING_LEVELS:
            continue
        path = entry.get("path", "")
        norm = _normalize_repo_path(path)
        suffix = _file_suffix(norm)
        # Path relative to the load root: a file under a load subdir (e.g. LLM/) is root-level there.
        load_rel = _load_relative_path(norm, load_subdirs)
        if not norm or suffix in _INERT_SUFFIXES or suffix in _SOURCE_SUFFIXES:
            # Inert formats cannot execute on load; source code is the consent gate's domain (auto_map).
            skipped.append({"path": path, "level": level})
        elif "/" not in load_rel:
            unsafe.append({"path": path, "level": level})  # root pickle -> load vector
        else:
            # Subdir pickle: deserialized only if a weight index references it.
            maybe_shard.append({"path": path, "level": level, "norm": norm})

    if maybe_shard:
        indexed = _indexed_shard_paths(
            model_name, hf_token, load_subdirs, revision = snapshot_revision
        )
        for m in maybe_shard:
            # Block if a root index lists this shard, or if the lookup was inconclusive. A definitive
            # "no index / not listed" stays non-blocking (e.g. NeMo nemo/*.distcp).
            if indexed is None or m["norm"] in indexed:
                unsafe.append({"path": m["path"], "level": m["level"]})
            else:
                skipped.append({"path": m["path"], "level": m["level"]})

    if not unsafe:
        if skipped:
            # Flagged files exist, but none the load deserializes -> allow, but log them.
            logger.info(
                "'%s': Hugging Face flagged files, but none are a load-path RCE "
                "vector (subdir/inert); allowing the load. Flagged: %s",
                model_name,
                ", ".join(f"{s['path']}({s['level']})" for s in skipped),
            )
        return FileSecurityDecision(model_name, False, reason = "no unsafe files in the load path")

    names = ", ".join(u["path"] for u in unsafe if u["path"]) or "unknown files"
    logger.warning(
        "Blocking load of '%s': Hugging Face security scan flagged unsafe files (%s).",
        model_name,
        names,
    )
    return FileSecurityDecision(
        model_name,
        True,
        unsafe_files = unsafe,
        reason = f"Hugging Face security scan flagged unsafe files: {names}",
    )
