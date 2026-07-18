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
    small allowlist of clean / not-yet-scanned levels is non-blocking. The sole
    fail-open path is an unavailable status (missing field / offline / error).
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


def _file_suffix(path: str) -> str:
    """Lowercase ``.ext`` of the basename, or ``""`` if none."""
    base = _normalize_repo_path(path).rsplit("/", 1)[-1]
    return "." + base.rsplit(".", 1)[1].lower() if "." in base else ""


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
                    paths.add(shard_norm)
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


def _fetch_security_status(model_name: str, hf_token: Optional[str]):
    """``security_repo_status`` (a dict) or None if unavailable. Hub metadata only;
    retries once on a transient error, then returns None so the caller fails open.
    """
    from huggingface_hub import model_info as hf_model_info
    from utils.utils import hf_env_offline

    # Effective offline: skip this metadata-only lookup instead of burning both
    # request timeouts (10s + 20s) on a session the user declared offline.
    #
    # SAFETY INVARIANT: this is only sound because every loader behind this gate
    # forces local-only loading from the SAME predicate -- see
    # core/rag/embeddings.py, which passes local_files_only = hf_env_offline() to
    # SentenceTransformer. Nothing can therefore be fetched here, so the scan's
    # job (block a poisoned pickle from being downloaded and deserialized) is
    # already served; the residual case, a model cached BEFORE it was flagged, is
    # the same fail-open this function has always documented for an unavailable
    # scan. If a loader ever stops honoring hf_env_offline(), this skip becomes
    # unsafe -- tests/test_hf_offline_flags.py pins that coupling.
    if hf_env_offline():
        logger.debug("HF security scan skipped for '%s': offline; failing open.", model_name)
        return None

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
            return getattr(info, "security_repo_status", None)
        except Exception as exc:  # network/offline/gated/404/unsupported-client
            last_exc = exc
            if attempt == 0:
                continue
    logger.debug(
        "HF security scan unavailable for '%s' (%s); failing open.",
        model_name,
        type(last_exc).__name__ if last_exc else "unknown",
    )
    return None


def evaluate_file_security(
    model_name: str,
    hf_token: Optional[str] = None,
    *,
    load_subdirs = (),
) -> FileSecurityDecision:
    """Block a load when HF's security scan flags unsafe serialized files.

    Call UNCONDITIONALLY before any load (independent of trust_remote_code): a malicious
    pickle deserializes during ``from_pretrained`` regardless. Metadata-only; fails open
    when the scan is unavailable.

    ``load_subdirs`` names subdirs the load calls ``from_pretrained`` on (e.g. ``("LLM",)``
    for Spark-TTS / BiCodec, loading ``<snapshot>/LLM``): a flagged file directly under one
    is root-level there and blocks, and an index inside it is honored when scoping shards.
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

    status = _fetch_security_status(model_name, hf_token)
    if not isinstance(status, dict):
        return FileSecurityDecision(
            model_name, False, reason = "scan unavailable; allowed (fail-open)"
        )

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
