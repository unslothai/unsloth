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
    small allowlist of clean / not-yet-scanned levels is non-blocking. An
    unavailable status (missing field / error) fails open, but an explicit
    local-only (offline) load fails CLOSED against the cached files instead: an
    unscanned cached pickle is blocked, a pickle-free (safetensors) cache allowed.
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
from typing import Optional

from loggers import get_logger

logger = get_logger(__name__)

# Non-blocking levels: clean or not-yet-finished. Anything else (unsafe/suspicious/
# malicious or a future label) blocks, so Hub schema drift fails CLOSED.
_NONBLOCKING_LEVELS = frozenset(
    {"", "safe", "pending", "scanning", "queued", "unscanned", "error", "unknown", "none"}
)

# Suffixes that cannot execute code on load (tensor-only safetensors, non-pickle gguf,
# text/markup/images), so a flag on one is never an RCE vector.
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

# Source files are not deserialized by a weight load; executable repo code runs only
# via auto_map, which is the consent gate's domain. So a flag on a .py is not this
# gate's vector (else a flagged helper/train script would false-block).
_SOURCE_SUFFIXES = frozenset({".py", ".pyc", ".pyx", ".pyi"})


# Root weight-index files. from_pretrained reads these to find sharded weights, so a
# flagged subdir pickle is a load vector iff a root index references it.
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


def _canonical_rel(rel: str):
    """A repo-relative path with ``.`` / ``..`` components collapsed lexically (no filesystem
    access), or None when it is empty, the repo root, or escapes the root (a leading ``..``
    after normalization).

    A ``modules.json`` / ``router_config.json`` / load-subdir / ``weight_map`` entry is
    repo-controlled. The loader resolves an entry such as ``0/../evil`` to ``evil/`` and
    deserializes ``evil/pytorch_model.bin`` -- the offline gate already scopes that SAME
    normalized directory (via :func:`_canonical_load_dir`) and the ONLINE scan must agree, or
    the raw ``0/../evil`` never equals the canonical ``evil/...`` the Hub reports for the
    flagged file and the pickle slips the gate. A legitimate declared path never traverses
    upward, so an escaping path is rejected."""
    import posixpath

    norm = posixpath.normpath(_normalize_repo_path(rel).strip("/"))
    if norm in ("", ".") or norm == ".." or norm.startswith("../"):
        return None
    return norm


def _canonical_load_dir(base, rel: str):
    """``base`` / ``rel`` with ``.`` / ``..`` collapsed lexically, or None when ``rel`` is
    empty, the base itself, or escapes ``base``. String form: :func:`_canonical_rel`."""
    norm = _canonical_rel(rel)
    return None if norm is None else base / norm


def _file_suffix(path: str) -> str:
    """Lowercase ``.ext`` of the basename, or ``""`` if none."""
    base = _normalize_repo_path(path).rsplit("/", 1)[-1]
    return "." + base.rsplit(".", 1)[1].lower() if "." in base else ""


def _load_relative_path(norm: str, load_subdirs) -> str:
    """``norm`` relative to a ``from_pretrained`` load root. Some loads read from a
    snapshot SUBDIRECTORY (Spark-TTS / BiCodec load ``<snapshot>/LLM``), where a file
    directly under the subdir is root-level, not nested. Strips the matching load-subdir
    prefix, or returns ``norm`` unchanged when it is not under one.

    A load-subdir is repo-controlled (the RAG guard unions in each ``modules.json`` module
    ``path``), so it is canonicalized (``0/../evil`` -> ``evil``) to match the canonical repo
    path the Hub reports for a flagged file -- the raw ``0/../evil`` would never prefix
    ``evil/pytorch_model.bin`` and the file would slip through as an unreferenced nested shard,
    though the offline gate (which canonicalizes the same path) blocks it.
    """
    for subdir in load_subdirs or ():
        prefix = _canonical_rel(subdir)
        if prefix and norm.startswith(prefix + "/"):
            return norm[len(prefix) + 1 :]
    return norm


def _index_prefixes(load_subdirs) -> tuple:
    """Prefixes to look for weight-index files under: repo root plus each load subdir. Each
    subdir is canonicalized (see :func:`_load_relative_path`) so a traversing declared path
    resolves to the same directory offline and online."""
    prefixes = [""]
    for subdir in load_subdirs or ():
        p = _canonical_rel(subdir)
        if p:
            prefixes.append(p + "/")
    return tuple(prefixes)


def _indexed_shard_paths(
    model_name: str,
    hf_token: Optional[str],
    load_subdirs = (),
):
    """Repo-relative weight paths a load could fetch via weight-index files. Returns a
    set (empty when the repo ships no index files -- a definitive "nothing sharded"), or
    None when the lookup was inconclusive (transient error) so the caller treats a
    flagged subdir pickle conservatively. Reads only small JSON indexes, never weights.
    Indexes are looked up at the root and each ``load_subdirs`` root, with ``weight_map``
    entries re-prefixed to repo-relative paths.
    """
    import json

    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.utils import EntryNotFoundError
    except Exception:
        return None

    paths: set = set()
    inconclusive = False
    for prefix in _index_prefixes(load_subdirs):
        for filename in _TRANSFORMERS_INDEX_FILES:
            try:
                index_path = hf_hub_download(model_name, prefix + filename, token = hf_token or None)
            except EntryNotFoundError:
                continue  # definitively absent, not an error
            except Exception:
                inconclusive = True  # transient: an index that might exist could not be read
                continue
            try:
                weight_map = (json.loads(open(index_path).read()) or {}).get("weight_map") or {}
                for shard in weight_map.values():
                    shard_norm = _normalize_repo_path(str(shard))
                    # weight_map paths are relative to the index file's directory.
                    if prefix and not shard_norm.startswith(prefix):
                        shard_norm = prefix + shard_norm
                    # Collapse . / .. (a repo-controlled weight_map may traverse, e.g.
                    # "sub/../evil/pytorch_model.bin") so the recorded path is the canonical
                    # one the Hub reports for the flagged shard -- mirrors the offline gate,
                    # which resolves the same traversal on disk. An escaping shard (leading
                    # ..) can never name a repo file, so drop it.
                    shard_canon = _canonical_rel(shard_norm)
                    if shard_canon is not None:
                        paths.add(shard_canon)
            except Exception:
                inconclusive = True
    # Any transient failure -> inconclusive (the shard could be listed only by the index
    # we could not read), so fail closed (None) and let the caller block. Ships no index
    # files -> EntryNotFoundError for each, empty set, a definitive "nothing sharded".
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
    # The commit SHA the Hub scan reported (online only), and whether that scan was a DEFINITIVE
    # clean verdict (a completed scan with no load-path issue) -- not merely a non-block. The
    # embedding recorder persists a clean verdict only when ``scanned_clean`` and the loaded commit
    # equals ``commit``; both stay unset for local-only and fail-open (unavailable) decisions.
    commit: Optional[str] = None
    scanned_clean: bool = False

    def response_payload(self) -> dict:
        """Machine-readable detail merged into the preflight payload the dialog reads."""
        return {
            "unsafe_files": self.unsafe_files,
            "security_blocked": self.blocked,
            "reason": self.reason,
        }


def security_load_subdirs(model_name: str, hf_token: Optional[str] = None) -> tuple:
    """Snapshot subdirectories a load calls ``from_pretrained`` on, for scoping the scan.
    Most models load from the root (``()``); Spark-TTS / BiCodec load ``<snapshot>/LLM``,
    so ``LLM/`` is a load root for them. Metadata-only (tokenizer special tokens), cached.
    """
    try:
        from utils.models.model_config import detect_audio_type, load_model_defaults
        if detect_audio_type(model_name, hf_token = hf_token) == "bicodec":
            return ("LLM",)
        # Tokenizer detection can fail (network/gated/unresolved alias); the YAML default
        # also pins the audio type, so fall back to it (else a flagged LLM/ pickle is
        # treated as an ignored subdir artifact).
        if (load_model_defaults(model_name) or {}).get("audio_type") == "bicodec":
            return ("LLM",)
    except Exception:
        pass
    return ()


def _load_scan_target(model_name: str, load_subdirs: tuple) -> tuple:
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
    # Rewrite ONLY a registry-known bicodec alias, never any repo ending in "/LLM"
    # (e.g. "evil/LLM" would scan unsloth/evil and fail open on the real repo).
    if name.endswith("/LLM") and name.count("/") == 1:
        try:
            from utils.models.model_config import load_model_defaults
            if (load_model_defaults(name) or {}).get("audio_type") == "bicodec":
                parent = name[: -len("/LLM")]
                return f"unsloth/{parent}", tuple(dict.fromkeys((*load_subdirs, "LLM")))
        except Exception:
            pass
    return model_name, load_subdirs


# Pickle weight formats a from_pretrained load deserializes (the RCE vector); safetensors
# and gguf are inert. Matched by base-model NAME so training_args.bin / optimizer.pt (not
# loaded as model weights) do not trip the offline gate.
_PICKLE_WEIGHT_RE = re.compile(
    r"^(model|pytorch_model)(-\d+-of-\d+)?\.(bin|pt|pth|ckpt|pkl|pickle)$"
)

# PEFT adapter pickle weights. from_pretrained auto-detects an adapter_config.json in the load
# root and deserializes the adapter weights on top of the base model, so an adapter pickle is a
# SEPARATE RCE vector from the base weights -- a safetensors base does not cover it. It has its
# own safetensors preference: adapter_model.safetensors is loaded in place of the pickle.
_ADAPTER_PICKLE_RE = re.compile(r"^adapter_model\.(bin|pt|pth|ckpt|pkl|pickle)$")

# Suffixes a from_pretrained load deserializes as a pickle (the RCE vector). Used for shards a
# weight-index maps, which are model weights regardless of their exact filename.
_PICKLE_SUFFIXES = frozenset({".bin", ".pt", ".pth", ".ckpt", ".pkl", ".pickle"})

# Root pickle weight-index. Its weight_map can point shards into SUBDIRECTORIES; from_pretrained
# follows the map and deserializes them wherever they live.
_PICKLE_INDEX_FILE = "pytorch_model.bin.index.json"

# SentenceTransformer Router (a.k.a. the legacy Asym) config. It declares the router's child
# sub-modules in its ``types`` map ({route}_{idx}_{ClassName} -> module class) and is the ONLY
# place they are declared -- the top-level modules.json lists just the Router. Router.load()
# deserializes each child from its own subdir, so those subdirs are load roots too.
_ROUTER_CONFIG_FILE = "router_config.json"


def _index_weight_map_values(index_path) -> set:
    """Repo-relative shard paths a weight-index maps (its ``weight_map`` values), or empty on
    an unreadable / malformed index."""
    import json

    try:
        weight_map = (json.loads(index_path.read_text(encoding = "utf-8")) or {}).get(
            "weight_map"
        ) or {}
    except (OSError, ValueError):
        return set()
    return {_normalize_repo_path(str(shard)) for shard in weight_map.values()}


# A from_pretrained load prefers safetensors over a pickle ONLY when the directory holds a
# safetensors weight it can actually load in its place: an unsharded base file, or an index
# whose every referenced shard is present. A bare adapter (``adapter_model.safetensors``) or
# an orphan shard with no index is NOT a loadable base weight -- the loader falls back to and
# deserializes the pickle, which therefore stays the live RCE vector.
#
# Only the EXACT names the loader resolves count. transformers' SAFE_WEIGHTS_NAME /
# SAFE_WEIGHTS_INDEX_NAME and sentence-transformers' Module.load_torch_weights both look up
# ``model.safetensors`` (then its index); neither ever looks up ``pytorch_model.safetensors``,
# so crediting that name would let a repo shipping an inert ``pytorch_model.safetensors`` decoy
# beside a live ``pytorch_model.bin`` pass unblocked while the loader deserialized the pickle.
_SAFETENSORS_BASE_UNSHARDED = ("model.safetensors",)
_SAFETENSORS_BASE_INDEX = ("model.safetensors.index.json",)


def _safetensors_index_complete(index_path) -> bool:
    """True when every shard the safetensors index maps is present, resolved RELATIVE TO the
    index's own directory -- a ``weight_map`` value may name a subdirectory
    (``weights/model-00001-of-00002.safetensors``), so comparing basenames alone would miss a
    complete set and wrongly treat the inert safetensors as absent."""
    import json

    try:
        weight_map = (json.loads(index_path.read_text(encoding = "utf-8")) or {}).get(
            "weight_map"
        ) or {}
    except (OSError, ValueError):
        return False  # unreadable index -> not a usable safetensors set -> keep the pickle blocked
    shards = {_normalize_repo_path(str(shard)) for shard in weight_map.values()}
    if not shards:
        return False
    base = index_path.parent
    for shard_rel in shards:
        try:
            if not base.joinpath(*shard_rel.split("/")).is_file():
                return False
        except OSError:
            return False
    return True


def _exact_named(files: dict, name: str):
    """The Path in *files* (lower-name -> Path for one directory) whose REAL basename is
    exactly *name* (case-sensitive), or None. A safetensors CREDIT must match the loader's
    exact lookup: transformers / sentence-transformers request ``model.safetensors`` verbatim,
    so on a case-sensitive filesystem (the Studio default) a mixed-case ``Model.SafeTensors``
    decoy is NOT the file the loader reads -- it falls back to and deserializes the pickle. The
    lower-name key would fold the decoy in and fail OPEN, so credit is decided by ``Path.name``.
    """
    for path in files.values():
        if path.name == name:
            return path
    return None


def _dir_has_loadable_safetensors(files: dict) -> bool:
    """True when *files* (lower-name -> Path for one directory) hold a safetensors weight a
    from_pretrained load will read INSTEAD of a pickle sibling: an unsharded base file, or a
    complete indexed shard set. A bare adapter or an orphan shard does not qualify. The
    safetensors credit is case-SENSITIVE (see :func:`_exact_named`) so a mis-cased decoy the
    loader would skip cannot vouch for a live pickle."""
    if any(_exact_named(files, name) is not None for name in _SAFETENSORS_BASE_UNSHARDED):
        return True
    for index_name in _SAFETENSORS_BASE_INDEX:
        index_path = _exact_named(files, index_name)
        if index_path is not None and _safetensors_index_complete(index_path):
            return True
    return False


def _router_child_dirs(root) -> set:
    """Child sub-module directories a SentenceTransformer ``Router`` (legacy ``Asym``) at *root*
    deserializes. A Router declares its children only in ``router_config.json`` (``types`` maps
    ``{route}_{idx}_{ClassName}`` -> module class), NOT in the top-level ``modules.json``, and
    ``Router.load()`` calls ``module_class.load(subfolder=model_id)`` on each. A child such as
    ``query_0_WordEmbeddings/`` holds ``wordembedding_config.json`` + ``pytorch_model.bin`` and no
    ``config.json``, so its pickle is still deserialized and its dir must be a load root. Returns
    an empty set when there is no readable router config."""
    import json

    try:
        config = json.loads((root / _ROUTER_CONFIG_FILE).read_text(encoding = "utf-8"))
    except (OSError, ValueError):
        return set()
    if not isinstance(config, dict):
        return set()
    types = config.get("types")
    children: set = set()
    if isinstance(types, dict):
        for model_id in types:
            child = _canonical_load_dir(root, str(model_id))
            if child is not None:
                children.add(child)
    return children


def _st_load_roots(snap, load_subdirs = ()) -> set:
    """Directories a load opens ``from_pretrained`` on: the snapshot root, every module path
    ``modules.json`` declares, and each passed-in ``load_subdirs`` entry. A SentenceTransformer
    module can load from a directory without ``config.json`` -- e.g. a ``0_WordEmbeddings/``
    module with ``wordembedding_config.json`` + ``pytorch_model.bin`` -- so a pickle there is
    still deserialized and must be treated as a load root."""
    roots = {snap}
    for subdir in load_subdirs or ():
        root = _canonical_load_dir(snap, str(subdir))
        if root is not None:
            roots.add(root)
    try:
        import json
        modules = json.loads((snap / "modules.json").read_text(encoding = "utf-8"))
    except (OSError, ValueError):
        modules = None
    if isinstance(modules, list):
        for module in modules:
            if isinstance(module, dict):
                root = _canonical_load_dir(snap, str(module.get("path") or ""))
                if root is not None:
                    roots.add(root)
    # A Router/Asym module declares its child sub-modules in router_config.json, not modules.json,
    # and Router.load() deserializes each child's weights from its own subdir. Treat those child
    # dirs as load roots too so a pickle in a config.json-less child (e.g.
    # query_0_WordEmbeddings/pytorch_model.bin) is scanned. Bounded BFS: every child is a strict
    # subpath and a visited set stops any cycle; scanning extra dirs only tightens the gate.
    pending = list(roots)
    while pending:
        current = pending.pop()
        for child in _router_child_dirs(current):
            if child not in roots:
                roots.add(child)
                pending.append(child)
    return roots


def _cached_pickle_weight_paths(snap, load_subdirs = ()) -> list:
    """The pickle weight FILES (as ``Path`` objects under *snap*) a from_pretrained load actually
    deserializes: at a real load root and with NO loadable safetensors alternative there. A load
    root is the snapshot root, a directory ``modules.json`` / ``load_subdirs`` declares, or a plain
    from_pretrained root (holds ``config.json``). A stray pickle in a non-load subdir (``archive/``,
    ``nemo/``) that no load opens is not a vector, matching the online scan's load-path scoping. Two
    weight classes are scoped independently: a BASE pickle (``pytorch_model.bin`` ...) is covered by
    a loadable base safetensors the loader picks instead; a PEFT ADAPTER pickle (``adapter_model.bin``),
    which from_pretrained auto-loads when ``adapter_config.json`` is present, is covered only by
    ``adapter_model.safetensors`` -- a safetensors base does NOT cover it. A bare adapter or an
    orphan shard leaves its pickle live. Returns the concrete Paths (not basenames) so the offline
    verdict cache can hash exactly the files the loader reads and never conflate two module dirs that
    ship the same pickle basename."""
    roots = _st_load_roots(snap, load_subdirs)
    by_dir_pickle: dict = {}
    by_dir_files: dict = {}  # directory -> {lower-name: Path}
    try:
        for path in snap.rglob("*"):
            try:
                if not path.is_file():
                    continue
                low = path.name.lower()
                by_dir_files.setdefault(path.parent, {})[low] = path
                if _PICKLE_WEIGHT_RE.match(low) or _ADAPTER_PICKLE_RE.match(low):
                    by_dir_pickle.setdefault(path.parent, []).append(path.name)
            except OSError:
                continue
    except OSError:
        return []
    hits: set = set()
    for directory, names in by_dir_pickle.items():
        files = by_dir_files.get(directory, {})
        # A pickle is deserialized only at an actual load root: the snapshot root, a declared
        # modules.json / load_subdirs dir, or a Router child (all resolved by _st_load_roots).
        # A stray config.json in an UNREFERENCED subdir (a nested checkpoint-500/ or archive/)
        # does NOT make it a load root -- from_pretrained never descends into it and the ST load
        # opens only declared modules -- so it must not block, matching the online scan which
        # ignores the same unindexed subdir pickle.
        if directory not in roots:
            continue
        base = [n for n in names if _PICKLE_WEIGHT_RE.match(n.lower())]
        if base and not _dir_has_loadable_safetensors(files):
            hits.update(files[n.lower()] for n in base)
        # An adapter pickle is deserialized only when from_pretrained auto-detects the adapter
        # (adapter_config.json present) and there is no adapter_model.safetensors to load instead.
        # The safetensors credit is case-SENSITIVE (a mixed-case Adapter_Model.SafeTensors decoy
        # is not the file PEFT loads), so it is matched by real basename; the config presence
        # stays case-insensitive (over-blocking a mis-cased adapter is the safe direction).
        adapter = [n for n in names if _ADAPTER_PICKLE_RE.match(n.lower())]
        if (
            adapter
            and "adapter_config.json" in files
            and _exact_named(files, "adapter_model.safetensors") is None
        ):
            hits.update(files[n.lower()] for n in adapter)
    # A load-root pickle index (pytorch_model.bin.index.json) can map shards into SUBDIRECTORIES
    # that are not themselves load roots; from_pretrained follows the map and deserializes them,
    # so include those referenced pickle shards (unless a loadable base safetensors at the index
    # root covers the base weights). Mirrors the online _indexed_shard_paths scan, read from disk.
    for root_dir in roots:
        files = by_dir_files.get(root_dir, {})
        index_path = files.get(_PICKLE_INDEX_FILE)
        if index_path is None or _dir_has_loadable_safetensors(files):
            continue
        for shard_rel in _index_weight_map_values(index_path):
            if _file_suffix(shard_rel) not in _PICKLE_SUFFIXES:
                continue
            shard = root_dir.joinpath(*shard_rel.split("/"))
            try:
                if shard.is_file():
                    hits.add(shard)
            except OSError:
                continue
    return sorted(hits)


def _snapshot_relative(path, snap) -> str:
    """*path* as a snapshot-relative posix string, or its basename if it is somehow not under
    *snap* (only for display / the blocked-file list)."""
    try:
        return path.relative_to(snap).as_posix()
    except ValueError:
        return path.name


def _cached_pickle_weight_files(snap, load_subdirs = ()) -> list:
    """Snapshot-relative posix names of the load-root pickle weights (see
    :func:`_cached_pickle_weight_paths`). Names are relative to the snapshot so two module dirs that
    each ship ``pytorch_model.bin`` are reported (and, in the verdict cache, keyed) distinctly."""
    return [_snapshot_relative(p, snap) for p in _cached_pickle_weight_paths(snap, load_subdirs)]


def _pickle_hash_map(snap, paths):
    """``{snapshot-relative posix name: sha256}`` for *paths*, or None if any file cannot be hashed
    (an unreadable pickle must never verify or record as clean)."""
    from utils.security import embedding_scan_verdicts

    out = {}
    for p in paths:
        digest = embedding_scan_verdicts.sha256_file(p)
        if digest is None:
            return None
        out[_snapshot_relative(p, snap)] = digest
    return out


def _matches_clean_verdict(model_name: str, snap, paths) -> bool:
    """True when a recorded clean Hub verdict covers EXACTLY this load-root pickle set at the active
    cached commit: same commit, same relative-name set, same sha256 for every file. A missing
    record, moved commit, changed / added / unreadable pickle, or any error -> False (fail-closed)."""
    try:
        from utils.models.model_config import _active_commit
        from utils.security import embedding_scan_verdicts

        commit = _active_commit(model_name)
        if not commit:
            return False
        recorded = embedding_scan_verdicts.lookup(model_name, commit)
        if not recorded:
            return False
        current = _pickle_hash_map(snap, paths)
        return current is not None and current == recorded
    except Exception:
        return False


def record_embedding_verdict(model_name: str, scanned_commit, load_subdirs = ()) -> None:
    """Record a clean Hub verdict for an embedding repo just loaded ONLINE, so a later offline load
    of the same content is not fail-closed. Persists the sha256 of every load-root pickle keyed by
    its snapshot-relative name. Records nothing when the snapshot is missing, the active cached
    commit differs from the scanned commit (branch moved -> the loaded content was not what HF
    scanned), there are no load-root pickles (an inert cache is already allowed), or any file cannot
    be hashed. Never raises into the load path."""
    try:
        from utils.models.model_config import _active_commit, _active_snapshot_dir
        from utils.security import embedding_scan_verdicts

        if not scanned_commit or _active_commit(model_name) != scanned_commit:
            return
        snap = _active_snapshot_dir(model_name)
        if snap is None:
            return
        paths = _cached_pickle_weight_paths(snap, load_subdirs)
        if not paths:
            return
        pickles = _pickle_hash_map(snap, paths)
        if pickles is None:
            return
        embedding_scan_verdicts.record_clean(model_name, scanned_commit, pickles)
    except Exception as exc:
        logger.debug("Could not record embedding scan verdict for '%s': %s", model_name, exc)


def _evaluate_local_only(model_name: str, load_subdirs = ()) -> "FileSecurityDecision":
    """Fail-CLOSED security decision for an offline (local_files_only) load.

    The Hub scan cannot be fetched offline, so instead of failing OPEN we inspect the cached
    files we already have: block the actual RCE vector -- a pickle weight the load
    deserializes -- and allow only a pickle-free cache (safetensors / gguf are inert). A
    previously-scanned pickle model must be reloaded online once to pass, or shipped as
    safetensors. Nothing cached means there is nothing to deserialize, so it is not blocked
    (the load fails downstream on its own, which is not a security event).
    """
    try:
        from utils.models.model_config import _active_snapshot_dir
        snap = _active_snapshot_dir(model_name)
    except Exception:
        snap = None
    if snap is None:
        return FileSecurityDecision(model_name, False, reason = "offline; nothing cached to scan")

    paths = _cached_pickle_weight_paths(snap, load_subdirs)
    if not paths:
        return FileSecurityDecision(
            model_name, False, reason = "offline; cached weights are pickle-free (inert)"
        )

    # A pickle model the user already loaded ONLINE (and HF scanned clean) may load offline when
    # its exact content is unchanged. Allow only when the active cached commit and EVERY load-root
    # pickle's sha256 match the recorded clean verdict; a missing record, moved commit, hash
    # mismatch, unreadable file, or any error falls through to the fail-closed block below.
    if _matches_clean_verdict(model_name, snap, paths):
        return FileSecurityDecision(
            model_name,
            False,
            reason = "offline; cached pickle weights match a recorded clean Hub scan",
        )

    pickles = [_snapshot_relative(p, snap) for p in paths]
    names = ", ".join(pickles)
    logger.warning(
        "Blocking offline load of '%s': cached pickle weights cannot be security-scanned "
        "offline (%s). Reconnect once to scan, or use safetensors weights.",
        model_name,
        names,
    )
    return FileSecurityDecision(
        model_name,
        True,
        unsafe_files = [{"path": p, "level": "unscanned"} for p in pickles],
        reason = (
            "offline: cached pickle weights are unscanned and cannot be verified; "
            f"reconnect once to scan, or use safetensors weights ({names})"
        ),
    )


def _fetch_security_status(model_name: str, hf_token: Optional[str]):
    """``(security_repo_status, commit_sha)`` from a single Hub metadata call, or ``(None, None)``
    if unavailable. The commit is the SHA the scan applies to (used to bind a recorded clean
    verdict to immutable content). Metadata only; retries once on a transient error, then returns
    ``(None, None)`` so the caller fails open.
    """
    from huggingface_hub import model_info as hf_model_info

    token_arg = hf_token if hf_token else False
    last_exc = None
    for attempt, timeout in enumerate((_REQUEST_TIMEOUT, _RETRY_TIMEOUT)):
        try:
            info = hf_model_info(
                model_name,
                token = token_arg,
                securityStatus = True,
                timeout = timeout,
            )
            return getattr(info, "security_repo_status", None), getattr(info, "sha", None)
        except Exception as exc:  # network/offline/gated/404/unsupported-client
            last_exc = exc
            if attempt == 0:
                continue
    logger.debug(
        "HF security scan unavailable for '%s' (%s); failing open.",
        model_name,
        type(last_exc).__name__ if last_exc else "unknown",
    )
    return None, None


def evaluate_file_security(
    model_name: str,
    hf_token: Optional[str] = None,
    *,
    load_subdirs = (),
    local_only_load: bool = False,
) -> FileSecurityDecision:
    """Block a load when HF's security scan flags unsafe serialized files.

    Call UNCONDITIONALLY before any load (independent of trust_remote_code): a malicious
    pickle deserializes during ``from_pretrained`` regardless. Metadata-only; fails open
    when the scan is unavailable.

    ``load_subdirs`` names subdirs the load calls ``from_pretrained`` on (e.g. ``("LLM",)``
    for Spark-TTS / BiCodec, loading ``<snapshot>/LLM``): a flagged file directly under one
    is root-level there and blocks, and an index inside it is honored when scoping shards.

    ``local_only_load`` marks a load the caller GUARANTEES cannot fetch (e.g. the RAG
    embedder, which passes ``local_files_only`` to SentenceTransformer from the same
    predicate). It cannot reach the Hub scan, so it is evaluated fail-CLOSED against the
    cached files (:func:`_evaluate_local_only`): a cached pickle weight is blocked, a
    pickle-free (safetensors) cache is allowed. Pass it only with that guarantee; claiming
    local-only while the loader can still fetch changes gate semantics. Default False.
    """
    # Scan the repo the load actually fetches, not the literal alias (which 404s and
    # fails open): the Spark-TTS "<parent>/LLM" alias is really unsloth/<parent> from LLM/.
    model_name, load_subdirs = _load_scan_target(model_name, tuple(load_subdirs))

    # Local paths (including a local .gguf) have no Hub scan. A remote ref is scanned
    # even if named "*.gguf", so a repo cannot dodge the scan via its name.
    try:
        from utils.paths import is_local_path
        if is_local_path(model_name):
            return FileSecurityDecision(model_name, False, reason = "local path; no Hub scan")
    except Exception:
        # Cannot classify the path -> do not block on that account.
        return FileSecurityDecision(model_name, False, reason = "path check failed; not blocked")

    # An offline (local-only) load cannot fetch the Hub scan: fail CLOSED against the cache
    # instead of skipping the gate, so an unscanned cached pickle cannot deserialize.
    if local_only_load:
        return _evaluate_local_only(model_name, load_subdirs)

    status, commit = _fetch_security_status(model_name, hf_token)
    if not isinstance(status, dict):
        return FileSecurityDecision(
            model_name, False, reason = "scan unavailable; allowed (fail-open)"
        )
    # A DEFINITIVE clean verdict (for the durable offline cache) requires a COMPLETED scan, unlike
    # the block decision which is not gated on scansDone. An in-progress scan is allowed online but
    # must not be persisted as clean.
    scans_done = bool(status.get("scansDone"))

    # Block a non-``safe`` flagged file scoped to the load-path RCE vector (root-level,
    # code-executing). Not gated on ``scansDone`` (often false even when clean; a flagged
    # file is flagged regardless). Unknown levels fail closed; in-progress/clean do not.
    # Subdir pickles and inert formats (safetensors/gguf) are not loaded by
    # from_pretrained and do not block. Unavailable status (above) is the only fail-open.
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
        # Path relative to the load root: a file under a load subdir (e.g. LLM/) is
        # root-level there, not nested.
        load_rel = _load_relative_path(norm, load_subdirs)
        if not norm or suffix in _INERT_SUFFIXES or suffix in _SOURCE_SUFFIXES:
            # Inert formats cannot execute on load; source code is the consent gate's
            # domain (auto_map), not a deserialization vector.
            skipped.append({"path": path, "level": level})
        elif "/" not in load_rel:
            unsafe.append({"path": path, "level": level})  # root pickle -> load vector
        else:
            # Subdir pickle: deserialized only if a weight index references it.
            maybe_shard.append({"path": path, "level": level, "norm": norm})

    if maybe_shard:
        indexed = _indexed_shard_paths(model_name, hf_token, load_subdirs)
        for m in maybe_shard:
            # Block if a root index lists this shard, or if the lookup was inconclusive
            # (transient error -> stay conservative). A definitive "no index / not listed"
            # stays non-blocking (e.g. NeMo nemo/*.distcp).
            if indexed is None or m["norm"] in indexed:
                unsafe.append({"path": m["path"], "level": m["level"]})
            else:
                skipped.append({"path": m["path"], "level": m["level"]})

    if not unsafe:
        if skipped:
            # Flagged files exist, but none the load deserializes (subdir pickle or inert
            # format) -> allow, but log them so they stay visible.
            logger.info(
                "'%s': Hugging Face flagged files, but none are a load-path RCE "
                "vector (subdir/inert); allowing the load. Flagged: %s",
                model_name,
                ", ".join(f"{s['path']}({s['level']})" for s in skipped),
            )
        return FileSecurityDecision(
            model_name,
            False,
            reason = "no unsafe files in the load path",
            commit = commit,
            scanned_clean = scans_done,
        )

    names = ", ".join(u["path"] for u in unsafe if u["path"]) or "unknown files"
    logger.warning(
        "Blocking load of '%s': Hugging Face security scan flagged unsafe files (%s).",
        model_name,
        names,
    )
    # An authoritative unsafe verdict revokes any stale clean record for this repo, so a
    # now-flagged commit cannot keep loading offline on a previously recorded verdict.
    try:
        from utils.security import embedding_scan_verdicts
        embedding_scan_verdicts.forget(model_name)
    except Exception:
        pass
    return FileSecurityDecision(
        model_name,
        True,
        unsafe_files = unsafe,
        reason = f"Hugging Face security scan flagged unsafe files: {names}",
        commit = commit,
    )
