# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Malware / unsafe-file gate for model loads.

The ``trust_remote_code`` consent gate (``consent.py``) covers one load-time RCE
vector: a repo's ``auto_map`` Python. It does NOT cover the other one -- a
malicious *pickle* inside a weight file (``pytorch_model.bin``, ``*.pkl``,
``*.dat``), which executes during ``from_pretrained`` deserialization *even with*
``trust_remote_code=False``. A repo with a normal ``config.json`` plus a poisoned
pickle would slip straight past the remote-code gate.

This module closes that gap using Hugging Face's OWN security scan (picklescan +
ClamAV), which the Hub runs on every repo and exposes as metadata via
``HfApi.model_info(repo, securityStatus=True).security_repo_status``::

    {"scansDone": true, "filesWithIssues": [{"path": "x.pkl", "level": "unsafe"}, ...]}

The check is strictly METADATA-ONLY: it reads the Hub's scan verdict and surfaces
the flagged file names + levels. It NEVER downloads, opens, ``torch.load``s, or
unpickles the flagged files.

Policy (confirmed product decisions):
  * Hard block, non-approvable -- a flagged file cannot be overridden by the user.
  * Block whenever ``filesWithIssues`` lists a blocking level, regardless of
    ``scansDone``: ``scansDone`` is frequently false even for fully-clean
    safetensors repos, and a file the Hub has ALREADY flagged ``unsafe`` is unsafe
    whether or not the rest of the scan has finished. Unknown/future non-``safe``
    levels fail CLOSED (block) so Hub schema drift cannot silently allow a
    newly-named bad verdict; only a small allowlist of "clean / not-yet-scanned"
    levels is treated as non-blocking. The ONLY fail-open path is an unavailable
    status (missing scan field / offline / any error) => allow (subprocess
    isolation + transformers ``weights_only`` remain the backstop).
  * Scope the block to the LOAD-PATH RCE vector. The gate exists to stop a
    malicious PICKLE deserializing during ``from_pretrained``; that loader reads
    weight files at the repo ROOT. A flag only blocks when it lands on a
    root-level file in a code-executing format. Two exclusions, because they are
    NOT an RCE vector for a Studio load:
      - Non-pickle formats can't execute: ``.safetensors`` is tensor-only by
        design, ``.gguf`` is a non-pickle llama.cpp format, configs/text/images
        cannot deserialize code. (Hub sometimes flags a repo's safetensors at
        ``unsafe`` when picklescan trips on a *sibling* pickle; that safetensors
        still cannot run code.)
      - Files in SUBDIRECTORIES are not read by ``from_pretrained`` UNLESS a root
        weight index (``pytorch_model.bin.index.json`` etc.) references them: a
        sharded ``.bin`` that the ``weight_map`` points at, e.g.
        ``shards/pytorch_model-00001-of-00002.bin``, IS deserialized and so still
        blocks. A flagged subdir pickle that no index lists (NeMo
        ``nemo/weights/*.distcp`` / ``common.pt``, ONNX/TF exports) is never loaded.
        This keeps real malware blocked (eicar ships its ``*.pkl`` / ``*.dat`` /
        ``eicar_test_file`` at the repo root, and an indexed malicious shard blocks
        wherever it lives) while not false-blocking legitimate first-party repos
        like ``nvidia/Nemotron-H-8B-Base-8K`` (root safetensors + flagged NeMo
        pickle checkpoints in ``nemo/`` that no index references).
  * No first-party exemption -- a poisoned pickle in a compromised trusted repo is
    exactly the CRITICAL-class threat we block even for first parties. (The scoping
    above is by load-path/format, not by org, so a root pickle in an
    ``unsloth``/``nvidia`` repo still blocks.)
  * Local paths are skipped (no Hub scan exists; a local ``.gguf`` lands here too).
    A REMOTE reference is always scanned, even one whose name ends in ``.gguf``, so
    a repo cannot dodge the scan by suffixing its name (``evil/model.gguf``); a
    reference that is not a real repo id simply fails open.
"""

from dataclasses import dataclass, field
from typing import Optional

from loggers import get_logger

logger = get_logger(__name__)

# Scan levels that are NOT a block: clean, or a not-yet-finished/non-verdict state.
# ANYTHING else (unsafe/suspicious/malicious, or a future label like "infected")
# is treated as blocking, so Hub schema drift fails CLOSED instead of silently
# allowing a newly-named bad verdict. "pending"/"scanning"/"error"/"" stay
# non-blocking so an in-progress or errored per-file scan does not false-block.
_NONBLOCKING_LEVELS = frozenset(
    {"", "safe", "pending", "scanning", "queued", "unscanned", "error", "unknown", "none"}
)

# File suffixes that cannot execute code when a model loads them, so a flag on one
# is never an RCE vector. safetensors is tensor-only by design; gguf is a non-pickle
# llama.cpp format; the rest are text/markup/images that no loader deserializes.
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

# Source files are not deserialized by ``from_pretrained`` -- a root ``.py`` is never
# imported by a weight load. Executable repo code runs only through ``auto_map`` under
# ``trust_remote_code``, which the remote-code CONSENT gate (consent.py) statically
# scans and gates. So a Hub flag on a Python file is the consent gate's domain, not a
# serialized-weight RCE vector for this gate; treating it as an unsafe load artifact
# would false-block repos that merely ship a flagged helper script (build/train .py).
_SOURCE_SUFFIXES = frozenset({".py", ".pyc", ".pyx", ".pyi"})


# Root weight-index files. ``from_pretrained`` reads these to find sharded weight
# files, and a shard they reference IS deserialized even when it lives in a
# subdirectory (e.g. ``weight_map`` -> ``shards/pytorch_model-00001-of-00002.bin``).
# So a flagged subdirectory pickle is a load-path vector iff a root index lists it.
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
    """``norm`` relative to a known ``from_pretrained`` load root.

    Most loads read weights from the repo root, but some call ``from_pretrained``
    on a SUBDIRECTORY of the snapshot (e.g. Spark-TTS / BiCodec load
    ``<snapshot>/LLM``). For those, a file directly under that subdir is a
    root-level load artifact, not a nested one, so classification must be done
    relative to the load root. Returns the path with the matching load-subdir
    prefix stripped, or ``norm`` unchanged when it is not under one.
    """
    for subdir in load_subdirs or ():
        prefix = _normalize_repo_path(subdir).strip("/")
        if prefix and norm.startswith(prefix + "/"):
            return norm[len(prefix) + 1 :]
    return norm


def _index_prefixes(load_subdirs) -> tuple:
    """Directory prefixes to look for weight-index files under: the repo root plus
    each ``from_pretrained`` load subdir (``""`` and e.g. ``"LLM/"``)."""
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
    """Repo-relative weight paths a load could fetch via weight-index files.

    Returns a ``set`` of normalized paths (possibly empty when the repo simply ships
    no index files -- a definitive "nothing sharded", so a flagged subdir artifact
    like ``nemo/*.distcp`` is NOT a load vector). Returns ``None`` only when the
    lookup was inconclusive (a transient/network error fetching an index that might
    exist), so the caller treats an already-flagged subdir pickle conservatively.
    Reads only small JSON index files; never the weight bytes.

    Index files are looked up at the repo root and under each ``load_subdirs`` load
    root, and ``weight_map`` entries (relative to their index's directory) are
    re-prefixed so they compare against repo-relative flagged paths.
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
                continue  # this index does not exist in the repo -- definitive, not an error
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
    # Any transient failure makes the result inconclusive, even if another index read
    # cleanly: the flagged shard could be listed only by the index we could not read,
    # so a partial path set is not safe to treat as definitive. Fail closed (None) and
    # let the caller block the already-flagged subdir pickle. A repo that simply ships
    # no index files raises EntryNotFoundError for each (never sets inconclusive) and
    # returns an empty set -- a definitive "nothing sharded".
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
    """Snapshot subdirectories a load calls ``from_pretrained`` on, for scoping the
    file-security scan.

    Most models load from the repo root (``()``). Spark-TTS / BiCodec load
    ``<snapshot>/LLM`` (see ``core/training/trainer.py``), so a flagged pickle under
    ``LLM/`` IS a load-path vector for them and the gate must treat ``LLM`` as a load
    root. Detection is metadata-only (tokenizer special tokens) and cached.
    """
    try:
        from utils.models.model_config import detect_audio_type, load_model_defaults

        if detect_audio_type(model_name, hf_token = hf_token) == "bicodec":
            return ("LLM",)
        # Tokenizer detection can fail (network/gated/an unresolved alias), but the
        # Studio YAML default also pins the audio type, so honor it as a fallback --
        # otherwise a flagged LLM/ pickle is treated as an ignored subdir artifact.
        if (load_model_defaults(model_name) or {}).get("audio_type") == "bicodec":
            return ("LLM",)
    except Exception:
        pass
    return ()


def _load_scan_target(model_name: str, load_subdirs: tuple) -> tuple:
    """Map a load alias to the ``(repo_id, load_subdirs)`` the load actually fetches.

    The Spark-TTS / BiCodec alias ``<parent>/LLM`` is downloaded by the trainer as
    ``unsloth/<parent>`` and loaded from ``LLM/`` (see ``core/training/trainer.py``).
    Scanning the literal alias 404s and fails open, so a flagged ``LLM/`` pickle in the
    real repo would be missed -- scan the real repo with ``LLM`` as a load root instead.
    Every other model is returned unchanged.
    """
    try:
        from utils.paths import is_local_path

        if is_local_path(model_name):
            return model_name, load_subdirs
    except Exception:
        return model_name, load_subdirs
    name = (model_name or "").strip().strip("/")
    if name.endswith("/LLM") and name.count("/") == 1:
        parent = name[: -len("/LLM")]
        return f"unsloth/{parent}", tuple(dict.fromkeys((*load_subdirs, "LLM")))
    return model_name, load_subdirs


def _fetch_security_status(model_name: str, hf_token: Optional[str]):
    """Return ``security_repo_status`` (a dict) or ``None`` if unavailable.

    Reads Hub metadata only (no file download). Retries once on a transient error,
    then gives up and returns ``None`` so the caller fails open.
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
    """Block a load when Hugging Face's security scan flags unsafe serialized files.

    Call this UNCONDITIONALLY before any load (independent of trust_remote_code),
    because a malicious pickle deserializes during ``from_pretrained`` regardless.
    Metadata-only: never touches the flagged file bytes. Fails open when the scan
    cannot be obtained.

    ``load_subdirs`` names snapshot subdirectories the load calls ``from_pretrained``
    on (e.g. ``("LLM",)`` for Spark-TTS / BiCodec, which loads ``<snapshot>/LLM``). A
    flagged file directly under such a subdir is a root-level load artifact there and
    blocks, and an index inside that subdir is honored when scoping flagged shards.
    """
    # Resolve a load alias to the repo the load actually fetches from, so the gate
    # scans the same target the loader does (the Spark-TTS "<parent>/LLM" alias is
    # really unsloth/<parent> loaded from LLM/). Scanning the literal alias 404s and
    # fails open. Reporting the resolved repo in the decision is the honest target.
    model_name, load_subdirs = _load_scan_target(model_name, tuple(load_subdirs))

    # Local folders/files (including a local .gguf) have no Hub security scan;
    # nothing to check. A REMOTE reference is scanned even if it ends in .gguf, so a
    # repo cannot evade the scan by naming itself "*.gguf".
    try:
        from utils.paths import is_local_path
        if is_local_path(model_name):
            return FileSecurityDecision(model_name, False, reason = "local path; no Hub scan")
    except Exception:
        # If we cannot even classify the path, do not block on that account.
        return FileSecurityDecision(model_name, False, reason = "path check failed; not blocked")

    status = _fetch_security_status(model_name, hf_token)
    if not isinstance(status, dict):
        return FileSecurityDecision(
            model_name, False, reason = "scan unavailable; allowed (fail-open)"
        )

    # Block on a file the Hub has flagged with a non-``safe`` level, scoped to the
    # actual load-path RCE vector (root-level, code-executing format). We do NOT
    # gate on ``scansDone`` (frequently false even for clean repos; an already-
    # flagged file is flagged regardless). Unknown levels fail closed (block);
    # in-progress/clean levels do not. A flag on a subdirectory pickle or an inert
    # format (safetensors/gguf) is not loaded by ``from_pretrained`` and does not
    # block. An unavailable status (above) is the sole fail-open path.
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
        # Classify relative to the load root: a file under a from_pretrained load
        # subdir (e.g. LLM/) is root-level there, not a nested artifact.
        load_rel = _load_relative_path(norm, load_subdirs)
        if not norm or suffix in _INERT_SUFFIXES or suffix in _SOURCE_SUFFIXES:
            # Inert formats cannot execute on load; source code is the consent gate's
            # domain (auto_map under trust_remote_code), not a deserialization vector.
            skipped.append({"path": path, "level": level})
        elif "/" not in load_rel:
            unsafe.append({"path": path, "level": level})  # root pickle -> load vector
        else:
            # Subdir pickle: deserialized only if a weight index references it.
            maybe_shard.append({"path": path, "level": level, "norm": norm})

    if maybe_shard:
        indexed = _indexed_shard_paths(model_name, hf_token, load_subdirs)
        for m in maybe_shard:
            # Block if a root index lists this shard, or if the index lookup was
            # inconclusive (a transient error -- treat a flagged subdir pickle that
            # COULD be a loadable shard conservatively). A definitive "no index /
            # not listed" keeps it non-blocking (e.g. NeMo nemo/*.distcp).
            if indexed is None or m["norm"] in indexed:
                unsafe.append({"path": m["path"], "level": m["level"]})
            else:
                skipped.append({"path": m["path"], "level": m["level"]})

    if not unsafe:
        if skipped:
            # The repo has flagged files, but only ones the model load never
            # deserializes (a subdirectory pickle, or an inert format like a
            # safetensors). Not an RCE vector for a from_pretrained load -> allow,
            # but log the flagged-but-not-loaded files so they remain visible.
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
