# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Seed inspect endpoints for data recipe."""

from __future__ import annotations

from core.training.account_jobs import account_path, managed_account
import base64
import binascii
import json
import os
import re
import shutil
from itertools import islice
from pathlib import Path, PurePosixPath
from typing import Any
from uuid import uuid4

from anyio import CapacityLimiter, to_process
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File as FastAPIFile, Form

from auth.authentication import allow_ambient_hf_token
from core.data_recipe.jsonable import to_preview_jsonable
from hub.services.datasets.local_options import (
    _sharded_splits,
    _IGNORED_DATA_FILENAMES,
    _MAX_MODULE_INFERENCE_FILES,
    _METADATA_FILENAMES,
    _SEP,
    _SPLIT_KEYWORDS,
)
from hub.utils.dataset_cache import refuse_unauthorized_dataset_preview
from hub.utils.hf_tokens import HfTokenArg, hf_token_arg
from loggers import get_logger
from utils.paths.lazy import LazyPath
from utils.paths import ensure_dir, seed_uploads_root, unstructured_uploads_root
from utils.utils import log_and_http_error
from utils.upload_limits import (
    LOCAL_SEED_UPLOAD_MAX_BYTES,
    LOCAL_SEED_UPLOAD_MAX_LABEL,
    UNSTRUCTURED_RECIPE_UPLOAD_MAX_BYTES,
    UNSTRUCTURED_RECIPE_UPLOAD_MAX_LABEL,
    UNSTRUCTURED_RECIPE_UPLOAD_TOTAL_MAX_BYTES,
    UNSTRUCTURED_RECIPE_UPLOAD_TOTAL_MAX_LABEL,
)

from models.data_recipe import (
    SeedInspectRequest,
    SeedInspectResponse,
    SeedInspectUploadRequest,
    UnstructuredFileUploadResponse,
)
from utils.paths.path_utils import is_appledouble_metadata

logger = get_logger(__name__)
router = APIRouter()

# Resolved on first use, not at module scope: the plugin package pulls the data designer engine, pandas and
# pyarrow, delaying uvicorn binding the port. False means "probed once, not installed", so callers still just see
# None.
_CHUNKING: Any = None


def _chunking() -> Any:
    global _CHUNKING
    if _CHUNKING is None:
        try:
            from data_designer_unstructured_seed import chunking
        except ImportError:
            _CHUNKING = False
        else:
            _CHUNKING = chunking
    return _CHUNKING or None


DATA_EXTS = (".parquet", ".jsonl", ".json", ".csv")
DEFAULT_SPLIT = "train"
DEFAULT_CONFIG = "default"
LOCAL_UPLOAD_EXTS = {".csv", ".json", ".jsonl"}
UNSTRUCTURED_ALLOWED_EXTS = {".pdf", ".docx", ".txt", ".md"}
SEED_UPLOAD_DIR = LazyPath(seed_uploads_root)
UNSTRUCTURED_UPLOAD_ROOT = LazyPath(unstructured_uploads_root)
_SAFE_ID_RE = re.compile(r"^[a-zA-Z0-9_-]+$")
# Frontend-generated upload namespace (UUID4 hex); legacy node ids (n1, ...) never match,
# since those directories can be shared by several recipes.
_UPLOAD_UID_RE = re.compile(r"^[0-9a-f]{32}$")


def _validate_safe_id(value: str, label: str) -> str:
    if not value or not _SAFE_ID_RE.match(value):
        raise HTTPException(400, f"Invalid {label}: must be alphanumeric/dash/underscore only")
    return value


def _serialize_preview_value(value: Any) -> Any:
    return to_preview_jsonable(value)


def _serialize_preview_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {str(key): _serialize_preview_value(value) for key, value in row.items()} for row in rows
    ]


def _normalize_optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    trimmed = value.strip()
    return trimmed if trimmed else None


def _list_hf_repo_files(*, dataset_name: str, token: HfTokenArg) -> list[str]:
    """Every file in the repo, including the ones no builder reads.

    A glob is run by the reader over all of them, so a pattern has to be judged
    against the whole listing even where only the data files can be loaded.
    """
    try:
        from huggingface_hub import HfApi
        from huggingface_hub.utils import HfHubHTTPError
    except ImportError:
        return []
    try:
        api = HfApi(token = token)
        return api.list_repo_files(dataset_name, repo_type = "dataset", token = token)
    except (HfHubHTTPError, OSError, ValueError):
        return []


def _list_hf_data_files(*, dataset_name: str, token: HfTokenArg) -> list[str]:
    repo_files = _list_hf_repo_files(dataset_name = dataset_name, token = token)
    return [file for file in repo_files if file.lower().endswith(DATA_EXTS)]


def _list_hf_dataset_configs(*, dataset_name: str, token: HfTokenArg) -> list[dict[str, Any]]:
    """The `configs:` block of the dataset card, which maps a config to its own file globs.

    A config name is not required to be a folder name: fineweb-edu's `sample-10BT`
    lives under `sample/10BT/`, so a folder-name guess reads a different config.
    """
    try:
        from huggingface_hub import HfApi
        from huggingface_hub.utils import HfHubHTTPError
    except ImportError:
        return []
    try:
        api = HfApi(token = token)
        card_data = api.dataset_info(dataset_name, token = token).card_data
        configs = (card_data or {}).get("configs") if card_data is not None else None
    except (HfHubHTTPError, OSError, ValueError, AttributeError):
        return []
    return [config for config in (configs or []) if isinstance(config, dict)]


def _as_pattern_list(paths: Any) -> list[str]:
    if isinstance(paths, str):
        return [paths]
    if isinstance(paths, list):
        return [path for path in paths if isinstance(path, str)]
    return []


def _patterns_for_split(
    data_files: Any,
    split: str,
    *,
    exact: bool = True,
) -> list[str]:
    """Every shape a card may use for `data_files`, as hub/services/datasets does.

    A bare string or list of strings is the shorthand for the train split. The
    name is read exactly first: a card may declare Train beside train, and
    `load_dataset` is handed the name the caller asked for, not a folded one.
    """
    if isinstance(data_files, str):
        return [data_files] if _same_split(DEFAULT_SPLIT, split, exact) else []
    if isinstance(data_files, dict):
        for name, paths in data_files.items():
            if _same_split(str(name), split, exact):
                return _as_pattern_list(paths)
        return []
    if isinstance(data_files, list):
        # Entry by entry, as hub/services/datasets does: a list may mix the bare
        # train shorthand with explicit mappings, and a split may be declared
        # more than once. Stopping at the first match drops the rest of it.
        patterns: list[str] = []
        for entry in data_files:
            if isinstance(entry, str):
                if _same_split(DEFAULT_SPLIT, split, exact):
                    patterns.append(entry)
            elif isinstance(entry, dict) and _same_split(
                str(entry.get("split") or ""), split, exact
            ):
                patterns.extend(_as_pattern_list(entry.get("path")))
        return patterns
    return []


def _same_split(declared: str, split: str, exact: bool) -> bool:
    return declared == split if exact else declared.lower() == split.lower()


def _declared_split_patterns(
    configs: list[dict[str, Any]],
    split: str = DEFAULT_SPLIT,
    subset: str | None = None,
) -> list[str]:
    """The globs the card declares for this split, under the named config."""
    config = _pick_config(configs, subset)
    if config is None:
        return []
    declared = config.get("data_files")
    named = _patterns_for_split(declared, split) or _patterns_for_split(
        declared, split, exact = False
    )
    folder = _config_folder(config)
    # `data_dir` scopes the config and its `data_files` are read under it, with
    # no exception for a pattern that happens to start with the folder's name:
    # `_resolve_data_files` joins the two whatever they say (`builder.py`).
    return [_normalized_glob(f"{folder}/{p}" if folder else p) for p in named]


def _normalized_glob(pattern: str) -> str:
    """The pattern as the listing spells it: no leading ./ and no bare dot.

    `datasets` runs `os.path.normpath` over `data_dir` before it resolves
    anything (`builder.py`), while `list_repo_files` returns data/train.parquet
    with no dot at all, so ./data or staging/../data has to be spelled the
    listing's way here or match nothing.
    """
    parts: list[str] = []
    for part in pattern.strip().split("/"):
        if part in ("", "."):
            continue
        if part == ".." and parts and parts[-1] != "..":
            parts.pop()
            continue
        parts.append(part)
    return "/".join(parts)


def _config_folder(config: dict[str, Any]) -> str:
    folder = config.get("data_dir")
    return _normalized_glob(folder) if isinstance(folder, str) else ""


def _config_scope(configs: list[dict[str, Any]], subset: str | None) -> str:
    """The folder a config is confined to, for a card that names one and lists no files."""
    config = _pick_config(configs, subset)
    return _config_folder(config) if config else ""


def _pick_config(configs: list[dict[str, Any]], subset: str | None) -> dict[str, Any] | None:
    """The config a bare request lands on, the way the Hub resolves it.

    Without a subset the answer is not always the one literally called `default`:
    a card may flag another config as the default, and a card with a single
    config uses it whatever its name (imdb ships only `plain_text`).
    """
    if subset:
        # Config names are case sensitive and a card may carry both Foo and foo,
        # so the exact one wins; a differently cased request only falls back to a
        # loose match when nothing matches exactly. An omitted config_name is the
        # default one, the same normalization hub/services/datasets applies.
        names = [(c, str(c.get("config_name") or DEFAULT_CONFIG)) for c in configs]
        return next(
            (c for c, name in names if name == subset),
            next((c for c, name in names if name.lower() == subset.lower()), None),
        )
    flagged = next((c for c in configs if c.get("default") is True), None)
    if flagged is not None:
        return flagged
    if len(configs) == 1:
        return configs[0]
    return next(
        (c for c in configs if str(c.get("config_name") or "default").lower() == DEFAULT_CONFIG),
        None,
    )


def _glob_to_regex(pattern: str) -> re.Pattern[str]:
    """A card glob as a regex: `*` stops at a folder boundary, `**` crosses it.

    Matching on the literal prefix instead would let `data/train-*.parquet` claim
    `data/train-notes.json`, which the recipe path then never reads.
    """
    parts: list[str] = []
    i = 0
    while i < len(pattern):
        char = pattern[i]
        if char == "*" and pattern[i + 1 : i + 2] == "*":
            if pattern[i + 2 : i + 3] == "/":
                # Whole folders, or none: `data/**/train.*` is not `data/nottrain.x`.
                parts.append("(?:[^/]+/)*")
                i += 3
            else:
                parts.append(".*")
                i += 2
            continue
        if char == "*":
            parts.append("[^/]*")
        elif char == "?":
            parts.append("[^/]")
        elif char == "[":
            end = _class_end(pattern, i)
            if end < 0:
                parts.append(r"\[")
            else:
                body = pattern[i + 1 : end].replace("\\", "\\\\")
                # Only `!` negates a glob class. A leading `^` is just a
                # character, so it is escaped rather than read as a regex.
                if body[:1] == "!":
                    body = f"^{body[1:]}"
                elif body[:1] == "^":
                    body = f"\\^{body[1:]}"
                parts.append(f"[{body}]")
                i = end
        else:
            parts.append(re.escape(char))
        i += 1
    return re.compile("".join(parts) + r"\Z")


def _class_end(pattern: str, start: int) -> int:
    """Index of the `]` closing the class opened at `start`, or -1 if unclosed."""
    i = start + 1
    if pattern[i : i + 1] in ("!", "^"):
        i += 1
    if pattern[i : i + 1] == "]":
        i += 1
    closing = pattern.find("]", i)
    return closing if closing > start else -1


def _ignored_by_the_loader(path: str, pattern: str) -> bool:
    """Whether the loader drops this match by name, as dataset_info.json.

    `datasets` removes those basenames from a glob's matches unless the pattern
    is that file itself (`data_files.FILES_TO_IGNORE`), so a card's `*` covers
    the shards and none of the metadata sitting beside them.
    """
    name = PurePosixPath(path).name
    return name in _IGNORED_DATA_FILENAMES and PurePosixPath(pattern).name != name


def _hidden_to_the_loader(path: str, pattern: str) -> bool:
    """Whether a match sits in a part the loader skips unless it was asked for.

    `datasets` drops a match whose path carries a hidden or dunder component the
    pattern does not carry itself, so `**/*` never reaches .backup/ or
    __pycache__/ (`data_files._is_unrequested_hidden_file_or_is_inside_unrequested_hidden_dir`).
    """
    for prefix in (".", "__"):
        in_path = [
            part
            for part in PurePosixPath(path).parts
            if part.startswith(prefix) and set(part) != {"."}
        ]
        in_pattern = [
            part
            for part in PurePosixPath(pattern).parts
            if part.startswith(prefix) and set(part) != {"."}
        ]
        if len(in_path) != len(in_pattern):
            return True
    return False


def _files_under_patterns(
    patterns: list[str],
    data_files: list[str],
    *,
    as_the_loader_would: bool = False,
) -> list[str]:
    """The files a glob takes. Reading a card's own glob, the loader's rules apply.

    Only there: a pattern this module WRITES is handed to a reader that skips
    nothing, so it has to be judged on what it plainly matches.
    """
    matchers = [(pattern, _glob_to_regex(pattern)) for pattern in patterns]
    return [
        f
        for f in data_files
        if any(
            m.match(f)
            and not (
                as_the_loader_would
                and (_hidden_to_the_loader(f, p) or _ignored_by_the_loader(f, p))
            )
            for p, m in matchers
        )
    ]


def _common_parent(paths: list[str]) -> str:
    parts = [Path(path).parent.as_posix().split("/") for path in paths]
    shared: list[str] = []
    for segments in zip(*parts):
        if len(set(segments)) != 1 or segments[0] == ".":
            break
        shared.append(segments[0])
    return "/".join(shared)


def _label_in_name(name: str, label: str) -> bool:
    """The label standing on its own in a file name, separators and all.

    Tokenizing the name instead would lose a label that carries a separator, so
    validation_matched or sample-10BT could never match anything. The separators
    are the loader's own (`local_options._SEP`), digits included, so train1 is
    the train split while training is not.
    """
    return re.search(rf"(?:^|{_SEP}){re.escape(label)}(?:$|{_SEP})", name) is not None


def _split_folder(lowered_path: str, labels: tuple[str, ...]) -> bool:
    """A folder standing for this split, qualified or not: train, train_a, en-train.

    The loader reads a separator-qualified folder as the split too
    (`local_options._DIR_NAME_KEYWORD_PATTERNS`), so an exact component match
    would leave train_a ranking as nothing.
    """
    return any(
        _label_in_name(part, label) for part in lowered_path.split("/")[:-1] for label in labels
    )


def _split_labels(split_lower: str) -> tuple[str, ...]:
    """The names `datasets` accepts for this split: dev is validation, training is train."""
    for canonical, aliases in _SPLIT_KEYWORDS.items():
        if split_lower == canonical or split_lower in aliases:
            return aliases
    return (split_lower,)


def _carries_another_split(path: str, split_lower: str) -> bool:
    """The path is labelled with a split, and it is not the one being asked for."""
    wanted = set(_split_labels(split_lower))
    return any(
        canonical not in wanted and _split_rank(path, canonical) <= 1
        for canonical in _SPLIT_KEYWORDS
    )


def _split_rank(path: str, split_lower: str) -> int:
    """0 the split is a folder, 1 the file name carries it, 2 neither.

    The name is read as whole labels rather than by where the split sits in it,
    so questions_train_000.jsonl counts as train, and `datasets` own aliases
    count too, so dev.jsonl is validation.
    """
    lowered = path.lower()
    labels = _split_labels(split_lower)
    if _split_folder(lowered, labels):
        return 0
    # Only the final extension comes off: questions.train.parquet keeps its split.
    stem = Path(lowered).stem
    if any(_label_in_name(stem, label) for label in labels):
        return 1
    return 2


def _in_subset(data_files: list[str], subset: str | None, split_lower: str) -> list[str]:
    """The files carrying this config label, as a folder or in the file name.

    A repo may encode its configs in the names instead of the layout, as
    main-train.parquet beside socratic-train.parquet. A folder carrying the
    label is the config whatever its files are called; a name only carrying it
    while belonging to no part of the split is a coincidence, not the config,
    so it is dropped rather than allowed to decide the answer.
    """
    if not subset:
        return data_files
    subset_lower = subset.lower()
    in_folder = [f for f in data_files if subset_lower in f.lower().split("/")[:-1]]
    if in_folder:
        return in_folder
    named = [f for f in data_files if _label_in_name(Path(f.lower()).stem, subset_lower)]
    if any(_split_rank(f, split_lower) <= 1 for f in named):
        return named
    if any(_split_rank(f, split_lower) <= 1 for f in data_files):
        return data_files
    return named or data_files


def _select_best_file(
    data_files: list[str],
    split: str = DEFAULT_SPLIT,
    subset: str | None = None,
) -> str | None:
    if not data_files:
        return None
    split_lower = split.lower()
    return sorted(
        _in_subset(data_files, subset, split_lower),
        key = lambda p: (_split_rank(p, split_lower), len(p)),
    )[0]


def _pattern_fits_the_split(
    data_files: list[str], wanted: set[str], root: str, pattern: str
) -> bool:
    """The pattern must take every wanted file under the root, and no other file there.

    A folder may name one split several ways at once (train-part.parquet beside
    questions_train.parquet), so a glob built from whichever file was picked can
    drop the rest; a loose glob can just as easily swallow a neighbouring split.
    Read relative to the root the pattern is anchored at, so a split spread over
    sibling folders is only accepted by a pattern that reaches all of them.
    Judged against the WHOLE listing, since a candidate checked only against the
    subset slice can still match another subset's files.
    """
    matcher = _glob_to_regex(pattern)
    prefix = f"{root}/" if root and root != "." else ""
    for path in data_files:
        if not path.startswith(prefix):
            continue
        if bool(matcher.match(path[len(prefix) :])) != (path in wanted):
            return False
    return True


def _candidate_patterns(stem: str, suffix: str, split_lower: str) -> list[str]:
    """Globs for this split's files in one folder, narrowest first.

    Built from whichever of the split's names this file actually uses, since a
    validation shard may well be called dev.
    """
    stem_lower = stem.lower()
    candidates: list[str] = []
    for label in _split_labels(split_lower):
        if stem_lower.startswith((f"{label}-", f"{label}_", f"{label}.")):
            candidates.append(f"{stem[: len(label) + 1]}*{suffix}")
        if stem_lower == label or stem_lower.endswith((f"_{label}", f"-{label}")):
            # Keep the trailing star: the split may still be sharded as train.jsonl
            # beside train_2.jsonl, so an exact name would read only the first shard.
            candidates.append(f"{stem}*{suffix}")
        at = stem_lower.find(label)
        if at >= 0:
            candidates.append(f"*{stem[at : at + len(label)]}*{suffix}")
    # Last and narrowest: the whole name. A config written after the split, as
    # train-main beside train-socratic, is only kept by carrying both labels.
    candidates.append(f"{stem}*{suffix}")
    return list(dict.fromkeys(candidates))


def _with_data_extension(pattern: str, suffix: str) -> str:
    """A card glob ends where the file name does; the reader picks by extension."""
    current = Path(pattern).suffix
    if current.lower() in DATA_EXTS:
        return pattern
    if current and any(char in current for char in "*?["):
        pattern = pattern[: -len(current)]
    # `**` has to be a whole path component: fsspec refuses data/**.parquet
    # outright, so a card's data/** becomes data/**/* before the extension.
    if pattern.endswith("**"):
        pattern = f"{pattern}/*"
    return f"{pattern}{suffix}"


def _dominant_suffix(paths: list[str]) -> str:
    """The format the loader would build this split from: most files, parquet on ties.

    One broad card glob can cover several formats, and picking whichever name
    sorts first would point the recipe at a format the split does not use
    (`local_options._one_module`).
    """
    counts: dict[str, int] = {}
    # Folder-builder metadata never decides a builder for the loader either, so
    # a couple of metadata.csv cannot outvote the real shards.
    voting = [p for p in paths if Path(p).name.lower() not in _METADATA_FILENAMES] or paths
    # The same window the loader infers from, so a split whose formats change
    # past it is read as the loader reads it rather than as the whole listing.
    for path in sorted(voting)[:_MAX_MODULE_INFERENCE_FILES]:
        # One vote per extension, as `load.infer_module_for_data_files_list`
        # counts them: .json and .jsonl share a builder, but they are still two
        # extensions, so neither lends the other its votes before the tie-break.
        suffix = Path(path).suffix.lower()
        counts[suffix] = counts.get(suffix, 0) + 1
    if not counts:
        return ""
    best = max(
        counts,
        key = lambda s: (counts[s], -DATA_EXTS.index(s) if s in DATA_EXTS else -99),
    )
    # The builder only widens once its extension has won on its own votes.
    group = _builder_exts(best)
    return next(Path(p).suffix for p in paths if Path(p).suffix.lower() in group)


# Extensions one builder reads: `datasets` loads .json and .jsonl with the same one.
_BUILDER_EXTS = ((".json", ".jsonl"),)


def _builder_exts(suffix: str) -> tuple[str, ...]:
    lowered = suffix.lower()
    for group in _BUILDER_EXTS:
        if lowered in group:
            return group
    return (lowered,)


def _of_suffix(paths: list[str], suffix: str) -> list[str]:
    group = _builder_exts(suffix)
    return [p for p in paths if Path(p).suffix.lower() in group]


def _extension_glob(paths: list[str], suffix: str, repo_files: list[str]) -> str:
    """What the pattern should end in to keep every file this builder reads.

    A split declared over both .json and .jsonl cannot be named by one of them
    without dropping the other, so the shared start of the two ends the glob.
    That glob is open at the end, and the reader runs it over the whole repo
    rather than over the data files listed here, so a a.json.gz sitting beside
    them would be read as well: where the repo holds one, the split is named by
    the one extension that won instead.
    """
    group = set(_builder_exts(suffix))
    used = {Path(path).suffix.lower() for path in paths} & group
    if len(used) < 2:
        return suffix
    stem = os.path.commonprefix(sorted(used))
    matcher = re.compile(rf"{re.escape(stem)}[^/]*\Z", re.IGNORECASE)
    if any(
        Path(path).suffix.lower() not in group and matcher.search(Path(path).name)
        for path in repo_files
    ):
        return suffix
    return f"{stem}*"


def _common_name_prefix(paths: list[str]) -> str:
    """What the file names share before the extension, which the glob supplies."""
    return os.path.commonprefix([Path(path).stem for path in paths]) if paths else ""


def _name_initials(paths: list[str]) -> str:
    """The first characters of these file names, as a glob class body.

    Only letters and digits, so nothing in the class can be read as syntax.
    """
    initials = sorted({Path(path).name[:1] for path in paths if Path(path).name[:1].isalnum()})
    if not initials or len(initials) != len({Path(path).name[:1] for path in paths}):
        return ""
    return "".join(initials)


def _staged_split_files(
    data_files: list[str],
    labels: tuple[str, ...],
    scope: str = "",
) -> set[str]:
    """This split's shards, when the repo is laid out the way the loader looks first.

    Split inference is staged, and the sharded data names come first: once
    data/train-00000-of-00001.parquet is there, the loader stops and never
    considers a train-named file anywhere else (`local_options._grouped_splits`).
    Widening over one of those would read data the split, and the preview beside
    it, both leave out.
    """
    # The loader reads a config's files relative to its `data_dir`, and the
    # sharded names it looks for start at `data/`, so the scope comes off first.
    prefix = f"{scope}/" if scope else ""
    inside = [path[len(prefix) :] for path in data_files if path.startswith(prefix)]
    grouped = _sharded_splits([PurePosixPath(path) for path in inside])
    if not grouped:
        return set()
    staged = {
        f"{prefix}{path.as_posix()}"
        for split, paths in grouped.items()
        if split.lower() in labels
        for path in paths
    }
    return staged if staged < set(data_files) else set()


def _anchor(dataset_name: str, root: str) -> str:
    """The recipe path up to the folder a pattern is written relative to."""
    base = f"datasets/{dataset_name}"
    return f"{base}/{root}" if root and root != "." else base


def _name_finals(paths: list[str]) -> str:
    """The last characters of these names before the extension, as a class body."""
    finals = {Path(path).stem[-1:] for path in paths}
    return "".join(sorted(finals)) if finals and all(c.isalnum() for c in finals) else ""


def _widened_declared_pattern(
    declared_files: list[str],
    data_files: list[str],
    split_lower: str,
    suffix: str,
    *,
    fallback_glob: bool = False,
) -> str:
    """One glob covering several declared ones, keeping the split out of it if it can.

    The folder holding them all may hold the other splits too, which is what the
    split-named form avoids; it is only used when it takes the declared files
    and nothing else.
    """
    parent = _common_parent(declared_files)
    base = f"{parent}/**" if parent else "**"
    prefix = f"{parent}/" if parent else ""
    wanted = set(declared_files)
    # The split may be written into the file names or into the folders between
    # here and them, so try both before giving up on keeping it.
    # Whichever of the split's names the files use: a validation set may well be
    # declared as dev-*.parquet.
    candidates = [f"{base}/*{label}*{suffix}" for label in _split_labels(split_lower)]
    candidates += [f"{prefix}**/*{label}*/**/*{suffix}" for label in _split_labels(split_lower)]
    # Declared files that say nothing about the split may still share a name the
    # neighbouring splits do not, as data/part-a beside data/part-b, or
    # sets/a/part beside sets/b/part with a test file in sets/c.
    shared_name = _common_name_prefix(declared_files)
    if shared_name:
        candidates.append(f"{prefix}{shared_name}*{suffix}")
        candidates.append(f"{prefix}**/{shared_name}*{suffix}")
    # Unrelated names still start somewhere the neighbours do not. A class is the
    # only union the reader understands; it rejects `{a,b}` outright.
    initials = _name_initials(declared_files)
    finals = _name_finals(declared_files)
    if initials:
        candidates.append(f"{prefix}**/[{initials}]*{suffix}")
    # Names that start where a neighbour also starts may still end elsewhere:
    # apple and banana beside avocado share their first letter with it but not
    # their last, and one class at each end separates them.
    if finals:
        candidates.append(f"{prefix}**/*[{finals}]{suffix}")
    if initials and finals:
        candidates.append(f"{prefix}**/[{initials}]*[{finals}]{suffix}")
    # A hand-written mapping may name files no glob can gather. Each one still
    # names itself, so there is always something to fall back to that reads only
    # what the card declared.
    candidates += sorted(wanted)
    for candidate in candidates:
        if set(_files_under_patterns([candidate], data_files)) == wanted:
            return candidate
    # Nothing names these files and only these files: train aa and bb beside
    # test ab and ba cannot be told apart by any class a glob can carry. A card
    # says outright which files are in the split, so a pattern that reads a
    # neighbour contradicts it and one that drops a shard truncates it; the
    # caller refuses the dataset rather than quietly doing either. A guess from
    # the file names says no such thing, so there the folder is still covered.
    return f"{base}/*{suffix}" if fallback_glob else ""


def _resolve_seed_hf_path(
    dataset_name: str,
    data_files: list[str],
    split: str = DEFAULT_SPLIT,
    subset: str | None = None,
    configs: list[dict[str, Any]] | None = None,
    repo_files: list[str] | None = None,
) -> str | None:
    declared = _declared_split_patterns(configs or [], split, subset)
    declared_files = _files_under_patterns(declared, data_files, as_the_loader_would = True)
    # The card's own mapping beats any guess from the folder names, as long as it
    # resolves to files that are really there.
    if declared_files:
        suffix = _dominant_suffix(declared_files)
        declared_files = _of_suffix(declared_files, suffix)
        exact = suffix
        suffix = _extension_glob(declared_files, suffix, repo_files or data_files)
        if suffix == exact and len({Path(f).suffix.lower() for f in declared_files}) > 1:
            # The card put both of the builder's extensions in this split and no
            # glob can take the two without also taking the file that made the
            # combined form unsafe. Refused, rather than read short.
            return None
        pattern = ""
        if len(declared) == 1:
            pattern = _with_data_extension(declared[0], suffix)
        # A split spread over several declared globs cannot be written as one, and
        # neither can a rewritten pattern that takes anything other than what the
        # card declared: the reader skips no hidden folder the loader would, so
        # the rewrite has to exclude one itself.
        if not pattern or set(declared_files) != set(_files_under_patterns([pattern], data_files)):
            pattern = _widened_declared_pattern(declared_files, data_files, split.lower(), suffix)
        # An unrepresentable mapping is refused, not approximated: the endpoint
        # answers 422 and the recipe is never pointed at the wrong rows.
        return f"datasets/{dataset_name}/{pattern}" if pattern else None

    # Without a card mapping the subset is only a label on the files. Narrow to
    # the ones carrying it first, so the pattern is checked against those alone
    # and cannot be widened back over another config.
    folder = _config_scope(configs or [], subset)
    scoped = _in_subset(
        [f for f in data_files if f.startswith(f"{folder}/")] or data_files
        if folder
        else data_files,
        subset,
        split.lower(),
    )
    split_lower = split.lower()
    labels = _split_labels(split_lower)
    # The loader looks at the sharded names first and stops there, so when the
    # repo has them the request is for those files, whatever else carries the
    # split in its name.
    staged = _staged_split_files(scoped, labels, folder)
    selected = _select_best_file(sorted(staged) or scoped, split)
    if not selected:
        return None

    suffix = Path(selected).suffix
    ext = suffix.lower()
    if ext not in DATA_EXTS:
        return f"datasets/{dataset_name}/{selected}"

    parent = Path(selected).parent.as_posix()
    base = f"datasets/{dataset_name}"
    if parent and parent != ".":
        base = f"{base}/{parent}"

    if _split_folder(selected.lower(), labels):
        # The loader reads train_a and train_b as one train split, so a folder
        # pattern rooted at the one folder that was picked drops the other.
        folders = staged or {f for f in scoped if _split_rank(f, split_lower) == 0}
        root = _common_parent(sorted(folders | {selected}))
        if root != parent:
            for label in labels:
                # The four folder shapes the loader's grammar accepts
                # (`local_options._DIR_NAME_KEYWORD_PATTERNS`), the last for a
                # label sitting between separators, as a_train_x.
                for shape in (
                    f"**/{label}/**/*{ext}",
                    f"**/{label}{_SEP}*/**/*{ext}",
                    f"**/*{_SEP}{label}/**/*{ext}",
                    f"**/*{_SEP}{label}{_SEP}*/**/*{ext}",
                    # One split may use several of those shapes at once, as
                    # train/ beside sets/train_a/, and only a looser form covers
                    # both. Still only taken where the listing bears it out.
                    f"**/{label}*/**/*{ext}",
                    f"**/*{label}*/**/*{ext}",
                ):
                    if _pattern_fits_the_split(data_files, folders, root, shape):
                        return f"{_anchor(dataset_name, root)}/{shape}"
    else:
        # The files this request is for: in the chosen subset, and of this split.
        wanted = staged or {f for f in scoped if _split_rank(f, split_lower) <= 1}
        # A split sharded over sibling folders, as a/train-0.parquet beside
        # b/train-1.parquet, cannot be written under the one folder its chosen
        # file sits in, so the pattern is anchored at the folder they share and
        # walks down from there. Without a card the loader reads those as one
        # split whatever else the folders hold, and a requested subset has
        # already narrowed `scoped`, so a neighbour cannot pull the answer wide.
        spread = any(Path(f).parent.as_posix() != parent for f in wanted)
        root = _common_parent(sorted(wanted | {selected})) if spread else parent
        stem = Path(selected).name[: -len(suffix)]
        for candidate in _candidate_patterns(stem, suffix, split_lower):
            pattern = f"**/{candidate}" if spread else candidate
            if _pattern_fits_the_split(data_files, wanted, root, pattern):
                return f"{_anchor(dataset_name, root)}/{pattern}"
        # One split may be written under several of its own names, as dev-0
        # beside validation-0, and no pattern built from the one file that was
        # picked can gather the others. The declared-mapping widening knows how
        # to name a set of files without reaching past it, so it is reused here,
        # but only where every neighbour is labelled: an unlabelled file may
        # belong to this split too, and only the broad glob would keep it.
        prefix = f"{root}/" if root and root != "." else ""
        neighbours = [f for f in data_files if f.startswith(prefix) and f not in wanted]
        if len(wanted) > 1 and all(_carries_another_split(f, split_lower) for f in neighbours):
            widened = _widened_declared_pattern(
                sorted(wanted), data_files, split_lower, suffix, fallback_glob = True
            )
            return f"datasets/{dataset_name}/{widened}"
    return f"{base}/**/*{ext}"


def _build_stream_load_kwargs(
    *,
    dataset_name: str,
    split: str,
    subset: str | None,
    token: HfTokenArg,
    data_file: str | None = None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "path": dataset_name,
        "split": split,
        "streaming": True,
        "trust_remote_code": False,
        "token": token,
    }
    if data_file:
        kwargs["data_files"] = [data_file]
    if subset:
        kwargs["name"] = subset
    return kwargs


def _load_preview_rows(
    *, load_dataset_fn, load_kwargs: dict[str, Any], preview_size: int
) -> list[dict[str, Any]]:
    streamed_ds = load_dataset_fn(**load_kwargs)
    return [row for row in islice(streamed_ds, preview_size)]


def _extract_columns(rows: list[dict[str, Any]]) -> list[str]:
    columns_seen: dict[str, None] = {}
    for row in rows:
        for key in row.keys():
            columns_seen[str(key)] = None
    return list(columns_seen.keys())


def _sanitize_filename(filename: str) -> str:
    name = Path(filename).name.strip().replace("\x00", "")
    if not name:
        return "seed_upload"
    return name


def _decode_base64_payload(content_base64: str) -> bytes:
    raw = content_base64.strip()
    if "," in raw and raw.lower().startswith("data:"):
        raw = raw.split(",", 1)[1]
    try:
        return base64.b64decode(raw, validate = True)
    except binascii.Error as exc:
        raise HTTPException(status_code = 400, detail = "invalid base64 payload") from exc


def _read_preview_rows_from_local_file(path: Path, preview_size: int) -> list[dict[str, Any]]:
    account_path(path)
    try:
        import pandas as pd
    except ImportError as exc:
        raise log_and_http_error(
            exc,
            500,
            "seed inspect dependencies unavailable",
            event = "data_recipe.seed.dependencies_unavailable",
            log = logger,
        ) from exc

    ext = path.suffix.lower()
    try:
        if ext == ".csv":
            df = pd.read_csv(path, nrows = preview_size, encoding = "utf-8-sig")
            df.columns = df.columns.str.strip()
            unnamed = [c for c in df.columns if c == "" or c.startswith("Unnamed:")]
            if unnamed:
                df = df.drop(columns = unnamed)
                full_df = pd.read_csv(path, encoding = "utf-8-sig")
                full_df.columns = full_df.columns.str.strip()
                full_df = full_df.drop(columns = unnamed)
                tmp_csv = path.with_suffix(".tmp.csv")
                full_df.to_csv(tmp_csv, index = False, encoding = "utf-8")
                tmp_csv.replace(path)
        elif ext == ".jsonl":
            df = pd.read_json(path, lines = True).head(preview_size)
        elif ext == ".json":
            try:
                df = pd.read_json(path).head(preview_size)
            except ValueError:
                df = pd.read_json(path, lines = True).head(preview_size)
        else:
            raise HTTPException(status_code = 422, detail = f"unsupported file type: {ext}")
    except HTTPException:
        raise
    except (ValueError, OSError) as exc:
        raise log_and_http_error(
            exc,
            422,
            "seed inspect failed",
            event = "data_recipe.seed.local_preview_failed",
            log = logger,
        ) from exc

    rows = df.to_dict(orient = "records")
    return _serialize_preview_rows(rows)


def _read_preview_rows_from_unstructured_file(
    *, path: Path, preview_size: int, chunk_size: int | None, chunk_overlap: int | None
) -> list[dict[str, Any]]:
    account_path(path)
    chunking = _chunking()
    if chunking is None:
        raise HTTPException(
            500,
            "Unstructured seed support not available (missing data_designer_unstructured_seed)",
        )
    size, overlap = chunking.resolve_chunking(chunk_size, chunk_overlap)
    try:
        rows = chunking.build_unstructured_preview_rows(
            source_path = path,
            preview_size = preview_size,
            chunk_size = size,
            chunk_overlap = overlap,
        )
    except (FileNotFoundError, RuntimeError, ValueError, OSError) as exc:
        raise log_and_http_error(
            exc,
            422,
            "seed inspect failed",
            event = "data_recipe.seed.unstructured_preview_failed",
            log = logger,
        ) from exc
    return _serialize_preview_rows(rows)


def _read_preview_rows_from_multi_files(
    *,
    block_id: str,
    file_ids: list[str],
    file_names: list[str],
    preview_size: int,
    chunk_size: int | None,
    chunk_overlap: int | None,
) -> list[dict[str, str]]:
    chunking = _chunking()
    if chunking is None:
        raise HTTPException(
            500,
            "Unstructured seed support not available (missing data_designer_unstructured_seed)",
        )

    _validate_safe_id(block_id, "block_id")
    block_dir = UNSTRUCTURED_UPLOAD_ROOT / block_id
    file_entries: list[tuple[Path, str]] = []
    for fid, fname in zip(file_ids, file_names):
        extracted = block_dir / f"{fid}.extracted.txt"
        if not extracted.exists():
            raise HTTPException(404, f"Extracted text not found for file: {fname} (id: {fid})")
        file_entries.append((extracted, fname))

    return chunking.build_multi_file_preview_rows(
        file_entries = file_entries,
        preview_size = preview_size,
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap,
    )


@router.post("/seed/inspect", response_model = SeedInspectResponse)
def inspect_seed_dataset(
    payload: SeedInspectRequest, allow_ambient_token: bool = Depends(allow_ambient_hf_token)
) -> SeedInspectResponse:
    dataset_name = payload.dataset_name.strip()
    if not dataset_name or dataset_name.count("/") < 1:
        raise HTTPException(
            status_code = 400,
            detail = "dataset_name must be a Hugging Face repo id like org/repo",
        )

    split = _normalize_optional_text(payload.split) or DEFAULT_SPLIT
    subset = _normalize_optional_text(payload.subset)
    # From the caller: a hardcoded False takes the ambient fallback from UI sessions too.
    token = hf_token_arg(
        _normalize_optional_text(payload.hf_token),
        allow_ambient_token = allow_ambient_token and not managed_account(),
    )
    preview_size = int(payload.preview_size)
    refuse_unauthorized_dataset_preview(token, dataset_name)

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise log_and_http_error(
            exc,
            500,
            "seed inspect dependencies unavailable",
            event = "data_recipe.seed.dependencies_unavailable",
            log = logger,
        ) from exc

    preview_rows: list[dict[str, Any]] = []
    repo_files = _list_hf_repo_files(dataset_name = dataset_name, token = token)
    data_files = [file for file in repo_files if file.lower().endswith(DATA_EXTS)]
    configs = _list_hf_dataset_configs(dataset_name = dataset_name, token = token)

    # Preview the same files the recipe will read, so the rows on screen are not
    # from a config the resolved path excludes.
    declared_files = _files_under_patterns(
        _declared_split_patterns(configs, split, subset),
        data_files,
        as_the_loader_would = True,
    )
    # Same format the resolved path will read, or the rows on screen come from a
    # file the recipe never opens.
    declared_files = _of_suffix(declared_files, _dominant_suffix(declared_files))
    # A config that only names a folder declares no files, and its name need not
    # be that folder, so the fallback previews the folder rather than the repo.
    folder = _config_scope(configs, subset)
    in_folder = [f for f in data_files if f.startswith(f"{folder}/")] if folder else []
    selected_file = _select_best_file(declared_files or in_folder or data_files, split, subset)
    if selected_file:
        try:
            single_file_kwargs = _build_stream_load_kwargs(
                dataset_name = dataset_name,
                split = split,
                subset = subset,
                token = token,
                data_file = selected_file,
            )
            preview_rows = _load_preview_rows(
                load_dataset_fn = load_dataset,
                load_kwargs = single_file_kwargs,
                preview_size = preview_size,
            )
        except (ValueError, OSError, RuntimeError):
            preview_rows = []

    if not preview_rows:
        try:
            split_kwargs = _build_stream_load_kwargs(
                dataset_name = dataset_name,
                split = split,
                subset = subset,
                token = token,
            )
            preview_rows = _load_preview_rows(
                load_dataset_fn = load_dataset,
                load_kwargs = split_kwargs,
                preview_size = preview_size,
            )
        except (ValueError, OSError, RuntimeError) as exc:
            raise log_and_http_error(
                exc,
                422,
                "seed inspect failed",
                event = "data_recipe.seed.hf_preview_failed",
                log = logger,
            ) from exc

    if not preview_rows:
        raise HTTPException(status_code = 422, detail = "dataset appears empty or unreadable")
    preview_rows = _serialize_preview_rows(preview_rows)
    columns = _extract_columns(preview_rows)

    if not data_files:
        resolved_path = f"datasets/{dataset_name}/**/*.parquet"
    else:
        resolved_path = _resolve_seed_hf_path(
            dataset_name, data_files, split, subset, configs, repo_files
        )
        if not resolved_path:
            raise HTTPException(status_code = 422, detail = "unable to resolve seed dataset path")

    return SeedInspectResponse(
        dataset_name = dataset_name,
        resolved_path = resolved_path,
        columns = columns,
        preview_rows = preview_rows,
        split = split,
        subset = subset,
    )


def _extract_text_from_file(file_path: Path, ext: str) -> str:
    """Extract text from an uploaded file by extension, to markdown where possible."""
    if ext in {".txt", ".md"}:
        raw = file_path.read_text(encoding = "utf-8", errors = "ignore")
    elif ext == ".pdf":
        from core.rag import config, pdf_ocr
        raw = pdf_ocr.extract_text(str(file_path), config.OCR_SCANNED, config.OCR_MAX_PAGES)
    elif ext == ".docx":
        import mammoth
        with open(str(file_path), "rb") as f:
            result = mammoth.convert_to_markdown(f)
            raw = result.value
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    chunking = _chunking()
    if chunking is None:
        return raw
    return chunking.normalize_unstructured_text(raw)


_pdf_extraction_limiter = CapacityLimiter(2)


async def _extract_text_from_file_async(file_path: Path, ext: str) -> str:
    if ext != ".pdf":
        return _extract_text_from_file(file_path, ext)
    from core.rag import config, pdf_ocr

    # MuPDF is not thread-safe; each worker owns its document and OCR state.
    raw = await to_process.run_sync(
        pdf_ocr.extract_text,
        str(file_path),
        config.OCR_SCANNED,
        config.OCR_MAX_PAGES,
        cancellable = True,
        limiter = _pdf_extraction_limiter,
    )
    chunking = _chunking()
    return chunking.normalize_unstructured_text(raw) if chunking is not None else raw


def _get_block_total_size(block_dir: Path) -> int:
    """Sum raw upload sizes for the whole block from server-owned files."""
    if not block_dir.exists():
        return 0
    total = 0
    for f in block_dir.iterdir():
        if not f.is_file():
            continue
        if f.name.endswith(".extracted.txt") or f.name.endswith(".meta.json"):
            continue
        # remove_unstructured_file keys on the name up to the first dot.
        if is_appledouble_metadata(f):
            continue
        total += f.stat().st_size
    return total


def _require_within_budget(size_bytes: int, budget: int) -> None:
    """413 on the tighter of the per-file cap and what the block has left."""
    if size_bytes > UNSTRUCTURED_RECIPE_UPLOAD_MAX_BYTES:
        raise HTTPException(
            413,
            f"File too large ({size_bytes} bytes). Maximum is {UNSTRUCTURED_RECIPE_UPLOAD_MAX_LABEL}.",
        )
    if size_bytes > budget:
        raise HTTPException(
            413,
            f"Total upload limit ({UNSTRUCTURED_RECIPE_UPLOAD_TOTAL_MAX_LABEL}) exceeded",
        )


def _require_unstructured_ext(filename: str) -> str:
    """Reject an unsupported type before any bytes are read."""
    ext = Path(filename).suffix.lower()
    if ext not in UNSTRUCTURED_ALLOWED_EXTS:
        raise HTTPException(
            400,
            f"Unsupported file type: {ext}. Allowed: {', '.join(sorted(UNSTRUCTURED_ALLOWED_EXTS))}",
        )
    return ext


def _read_native_drop(lease: str, budget: int) -> tuple[str, bytes]:
    """Read a desktop drop; returns (filename, content). The webview never names a path directly: Rust
    signs what the OS handed it, and this re-verifies and re-stats that grant before reading a byte.
    Same contract as the RAG route's ``_save_native_path_upload``. ``budget`` is what is still
    allowed for this block. The path is a local file of any size, so it is refused on its stat
    rather than after a multi-gigabyte read, and the read itself stops one byte past the budget in
    case the file grew between the two."""
    from utils.native_path_leases import NativePathLeaseError, verify_native_path_lease

    try:
        grant = verify_native_path_lease(
            lease,
            operation = "attach",
            expected_kind = "attachment",
            expected_path_type = "file",
            allowed_suffixes = sorted(UNSTRUCTURED_ALLOWED_EXTS),
        )
    except NativePathLeaseError as exc:
        raise HTTPException(400, str(exc)) from exc

    account_path(grant.canonical_path)
    _require_unstructured_ext(grant.canonical_path.name)
    try:
        size_bytes = grant.canonical_path.stat().st_size
    except OSError as exc:
        raise HTTPException(400, "Dropped file could not be read.") from exc
    _require_within_budget(size_bytes, budget)

    try:
        with grant.canonical_path.open("rb") as source:
            content = source.read(budget + 1)
    except OSError as exc:
        raise HTTPException(400, "Dropped file could not be read.") from exc
    _require_within_budget(len(content), budget)
    return grant.canonical_path.name, content


@router.post("/seed/upload-unstructured-file")
async def upload_unstructured_file(
    file: UploadFile | None = FastAPIFile(None),
    block_id: str = Form(...),
    native_path_lease: str | None = Form(None, alias = "nativePathLease"),
) -> UnstructuredFileUploadResponse:
    _validate_safe_id(block_id, "block_id")

    block_dir = UNSTRUCTURED_UPLOAD_ROOT / block_id
    # Reads 0 for a block with no directory yet, so this does not create one for
    # an upload that is about to be refused.
    budget = UNSTRUCTURED_RECIPE_UPLOAD_TOTAL_MAX_BYTES - _get_block_total_size(block_dir)

    # Desktop drops arrive as a signed path: Tauri hands the webview a path, never a File (#9036); isinstance,
    # not a truth test, since an unfilled Form param is still truthy.
    lease = native_path_lease if isinstance(native_path_lease, str) else None
    if lease:
        original_filename, content = _read_native_drop(lease, budget)
    elif file is not None and hasattr(file, "read"):
        original_filename = file.filename or "upload"
        # Before the read: a rejected 500 MB upload must not be pulled into memory first.
        _require_unstructured_ext(original_filename)
        content = await file.read()
    else:
        raise HTTPException(400, "No file was provided.")

    ext = Path(original_filename).suffix.lower()
    size_bytes = len(content)

    if size_bytes == 0:
        raise HTTPException(400, "Empty file not allowed")

    _require_within_budget(size_bytes, budget)
    ensure_dir(block_dir)

    file_id = uuid4().hex
    raw_path = block_dir / f"{file_id}{ext}"
    raw_path.write_bytes(content)

    extracted_path = block_dir / f"{file_id}.extracted.txt"
    try:
        extracted_text = await _extract_text_from_file_async(raw_path, ext)
        if not extracted_text or not extracted_text.strip():
            raw_path.unlink(missing_ok = True)
            return UnstructuredFileUploadResponse(
                file_id = file_id,
                filename = original_filename,
                size_bytes = size_bytes,
                status = "error",
                error = "No extractable text found in file",
            )
        extracted_path.write_text(extracted_text, encoding = "utf-8")
    except ImportError as e:
        raw_path.unlink(missing_ok = True)
        extracted_path.unlink(missing_ok = True)
        missing = getattr(e, "name", None)
        expected_missing = {".pdf": "pymupdf4llm", ".docx": "mammoth"}.get(ext)
        if isinstance(e, ModuleNotFoundError) and missing == expected_missing:
            logger.error(
                "data_recipe.seed.text_extraction_dependency_missing",
                error = str(e),
                missing = missing,
                exc_info = True,
            )
            return UnstructuredFileUploadResponse(
                file_id = file_id,
                filename = original_filename,
                size_bytes = size_bytes,
                status = "error",
                error = f"Cannot read {ext} files: the '{missing}' package is not installed.",
            )
        logger.error(
            "data_recipe.seed.text_extraction_failed",
            error = str(e),
            exc_info = True,
        )
        return UnstructuredFileUploadResponse(
            file_id = file_id,
            filename = original_filename,
            size_bytes = size_bytes,
            status = "error",
            error = "Text extraction failed.",
        )
    except Exception as e:
        from core.rag.pdf_ocr import PDFOCRError

        raw_path.unlink(missing_ok = True)
        extracted_path.unlink(missing_ok = True)
        logger.error(
            "data_recipe.seed.text_extraction_failed",
            error = str(e),
            exc_info = True,
        )
        return UnstructuredFileUploadResponse(
            file_id = file_id,
            filename = original_filename,
            size_bytes = size_bytes,
            status = "error",
            error = str(e) if isinstance(e, PDFOCRError) else "Text extraction failed.",
        )

    except BaseException:
        # Cancellation kills the OCR worker before its input can be removed.
        raw_path.unlink(missing_ok = True)
        extracted_path.unlink(missing_ok = True)
        raise

    try:
        meta_path = block_dir / f"{file_id}.meta.json"
        meta_path.write_text(
            json.dumps({"original_filename": original_filename, "size_bytes": size_bytes}),
            encoding = "utf-8",
        )
    except OSError:
        raw_path.unlink(missing_ok = True)
        extracted_path.unlink(missing_ok = True)
        return UnstructuredFileUploadResponse(
            file_id = file_id,
            filename = original_filename,
            size_bytes = size_bytes,
            status = "error",
            error = "Failed to save file metadata",
        )

    return UnstructuredFileUploadResponse(
        file_id = file_id,
        filename = original_filename,
        size_bytes = size_bytes,
        status = "ok",
    )


@router.delete("/seed/unstructured-file/{block_id}/{file_id}")
async def remove_unstructured_file(block_id: str, file_id: str):
    _validate_safe_id(block_id, "block_id")
    _validate_safe_id(file_id, "file_id")

    block_dir = UNSTRUCTURED_UPLOAD_ROOT / block_id
    if not block_dir.exists():
        raise HTTPException(404, "Block not found")

    deleted = False
    for f in block_dir.iterdir():
        stem = f.name.split(".")[0]
        if stem == file_id:
            f.unlink(missing_ok = True)
            deleted = True

    if not deleted:
        raise HTTPException(404, "File not found")
    try:
        if not any(block_dir.iterdir()):
            block_dir.rmdir()
    except OSError:
        pass

    return {"status": "ok"}


@router.delete("/seed/unstructured-block/{block_id}")
async def remove_unstructured_block(block_id: str):
    """Delete a block's upload directory; files on disk still count toward its quota.

    Only uid-namespaced directories may be bulk-deleted: they have exactly one
    owning block. Legacy node-id directories (n1, ...) can be shared by other
    recipes, so they are managed file-by-file instead.
    """
    _validate_safe_id(block_id, "block_id")
    if not _UPLOAD_UID_RE.match(block_id):
        raise HTTPException(400, "Invalid block_id: only uid-namespaced blocks can be deleted")

    block_dir = (UNSTRUCTURED_UPLOAD_ROOT / block_id).resolve()
    if not block_dir.is_relative_to(UNSTRUCTURED_UPLOAD_ROOT.resolve()):
        raise HTTPException(400, "Invalid block_id: outside upload root")
    if not block_dir.exists():
        return {"status": "ok", "deleted": False}

    try:
        shutil.rmtree(block_dir)
    except OSError as exc:
        raise log_and_http_error(
            exc,
            500,
            "failed to delete uploaded files",
            event = "data_recipe.seed.unstructured_block_delete_failed",
            log = logger,
        ) from exc
    if block_dir.exists():
        raise HTTPException(500, "failed to delete uploaded files")
    return {"status": "ok", "deleted": True}


@router.post("/seed/inspect-upload", response_model = SeedInspectResponse)
def inspect_seed_upload(payload: SeedInspectUploadRequest) -> SeedInspectResponse:
    if payload.file_ids is not None:
        if len(payload.file_ids) == 0:
            raise HTTPException(400, "file_ids must not be empty")
        _validate_safe_id(payload.block_id, "block_id")
        for fid in payload.file_ids:
            _validate_safe_id(fid, "file_id")
        preview_rows = _read_preview_rows_from_multi_files(
            block_id = payload.block_id,
            file_ids = payload.file_ids,
            file_names = payload.file_names,
            preview_size = payload.preview_size,
            chunk_size = payload.unstructured_chunk_size,
            chunk_overlap = payload.unstructured_chunk_overlap,
        )
        columns = ["chunk_text", "source_file"] if preview_rows else []
        resolved_paths = [
            str(UNSTRUCTURED_UPLOAD_ROOT / payload.block_id / f"{fid}.extracted.txt")
            for fid in payload.file_ids
        ]
        return SeedInspectResponse(
            dataset_name = "unstructured_seed",
            resolved_path = resolved_paths[0] if resolved_paths else "",
            resolved_paths = resolved_paths,
            columns = columns,
            preview_rows = _serialize_preview_rows(preview_rows),
        )

    seed_source_type = _normalize_optional_text(payload.seed_source_type) or "local"
    filename = _sanitize_filename(payload.filename)
    ext = Path(filename).suffix.lower()
    # Legacy single-file path is .txt/.md only; PDF/DOCX use multi-file upload
    _LEGACY_UNSTRUCTURED_EXTS = {".txt", ".md"}
    if seed_source_type == "unstructured":
        if ext not in _LEGACY_UNSTRUCTURED_EXTS:
            allowed = ", ".join(sorted(_LEGACY_UNSTRUCTURED_EXTS))
            raise HTTPException(
                status_code = 400,
                detail = f"unsupported file type: {ext}. allowed: {allowed}",
            )
    else:
        if ext not in LOCAL_UPLOAD_EXTS:
            allowed = ", ".join(sorted(LOCAL_UPLOAD_EXTS))
            raise HTTPException(
                status_code = 400,
                detail = f"unsupported file type: {ext}. allowed: {allowed}",
            )

    file_bytes = _decode_base64_payload(payload.content_base64)
    if not file_bytes:
        raise HTTPException(status_code = 400, detail = "empty upload payload")
    if len(file_bytes) > LOCAL_SEED_UPLOAD_MAX_BYTES:
        raise HTTPException(
            status_code = 413,
            detail = f"file too large (max {LOCAL_SEED_UPLOAD_MAX_LABEL})",
        )

    ensure_dir(SEED_UPLOAD_DIR)
    stored_name = f"{uuid4().hex}_{filename}"
    stored_path = SEED_UPLOAD_DIR / stored_name
    stored_path.write_bytes(file_bytes)

    if seed_source_type == "unstructured":
        preview_rows = _read_preview_rows_from_unstructured_file(
            path = stored_path,
            preview_size = int(payload.preview_size),
            chunk_size = payload.unstructured_chunk_size,
            chunk_overlap = payload.unstructured_chunk_overlap,
        )
    else:
        preview_rows = _read_preview_rows_from_local_file(
            stored_path,
            int(payload.preview_size),
        )
    if not preview_rows:
        raise HTTPException(status_code = 422, detail = "dataset appears empty or unreadable")
    columns = _extract_columns(preview_rows)

    return SeedInspectResponse(
        dataset_name = filename,
        resolved_path = str(stored_path),
        columns = columns,
        preview_rows = preview_rows,
        split = None,
        subset = None,
    )


@router.get("/seed/github/env-token")
def get_github_env_token_status() -> dict:
    """Report whether the server has a GH_TOKEN / GITHUB_TOKEN env var.

    The value is never returned; the UI uses this to tell the user they
    can leave the token field blank.
    """
    if managed_account():
        return {"has_token": False}
    has_token = bool(os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN"))
    return {"has_token": has_token}
