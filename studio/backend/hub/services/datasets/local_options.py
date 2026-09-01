# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import json
import os
import re
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Optional

from hub.schemas.datasets import (
    DatasetSplitOption,
    LocalDatasetOptionsRequest,
    LocalDatasetOptionsResponse,
)
from hub.utils.dataset_cache import (
    TRAINING_DATA_EXTS,
    dataset_cache_path_from_cache_path,
    dataset_snapshot_from_cache_path,
    latest_cached_dataset_path,
    processed_dataset_cache_path,
    resolved_dataset_snapshot_file,
)
from hub.utils.paths import is_valid_repo_id
from utils.hf_dataset_options import (
    HF_DATASET_SPLIT_NAME_PATTERN,
    has_unsafe_hf_dataset_option_characters,
)
from utils.paths.path_utils import is_appledouble_metadata


_MAX_METADATA_BYTES = 2 * 1024 * 1024
_MAX_PROCESSED_METADATA_FILES = 256
_MAX_PROCESSED_WALK_DEPTH = 4
_MAX_OPTIONS = 2048
_MAX_OPTION_LENGTH = 128
_CONFIG_RE = re.compile(r"[^<>:/\\|?*\x00-\x1f\x7f]+")
_SPLIT_RE = HF_DATASET_SPLIT_NAME_PATTERN


def _valid_option(
    value: Any,
    pattern: re.Pattern[str],
    *,
    reject_dotdot: bool = False,
) -> Optional[str]:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    if not normalized or len(normalized) > _MAX_OPTION_LENGTH:
        return None
    if has_unsafe_hf_dataset_option_characters(normalized):
        return None
    if "/" in normalized or "\\" in normalized:
        return None
    if normalized in {".", ".."} or (reject_dotdot and ".." in normalized):
        return None
    if pattern.fullmatch(normalized) is None:
        return None
    return normalized


def _split_names(value: Any) -> list[str]:
    if isinstance(value, dict):
        candidates: Iterable[Any] = value.keys()
    elif isinstance(value, list):
        candidates = (item.get("name") if isinstance(item, dict) else item for item in value)
    else:
        return []
    return [
        name
        for item in candidates
        if (name := _valid_option(item, _SPLIT_RE, reject_dotdot = True)) is not None
    ]


def _config_name(value: Any, fallback: Any = None) -> Optional[str]:
    candidate = value if value is not None else fallback
    return "default" if candidate is None else _valid_option(candidate, _CONFIG_RE)


def _add_info_options(
    options: set[tuple[str, str]],
    payload: Any,
    *,
    fallback_config: Optional[str] = None,
) -> None:
    if not isinstance(payload, dict) or len(options) >= _MAX_OPTIONS:
        return
    config = _config_name(payload.get("config_name"), fallback_config)
    if config is None:
        return
    for split in _split_names(payload.get("splits")):
        options.add((config, split))
        if len(options) >= _MAX_OPTIONS:
            return


def _add_dataset_info_options(options: set[tuple[str, str]], payload: Any) -> None:
    if isinstance(payload, list):
        for item in payload:
            _add_info_options(options, item)
        return
    if not isinstance(payload, dict):
        return
    if "splits" in payload or "config_name" in payload:
        _add_info_options(options, payload)
        return
    for config, info in payload.items():
        _add_info_options(options, info, fallback_config = config)


def _add_config_options(options: set[tuple[str, str]], payload: Any) -> None:
    if not isinstance(payload, list):
        return
    for item in payload[:_MAX_OPTIONS]:
        if not isinstance(item, dict):
            continue
        config = _config_name(item.get("config_name"))
        if config is None:
            continue
        data_files = item.get("data_files")
        if isinstance(data_files, dict):
            splits: Iterable[Any] = data_files.keys()
        elif isinstance(data_files, str):
            splits = ("train",)
        elif isinstance(data_files, list):
            splits = (
                entry.get("split")
                if isinstance(entry, dict)
                else "train"
                if isinstance(entry, str)
                else None
                for entry in data_files
            )
        else:
            splits = ()
        for split_value in splits:
            split = _valid_option(split_value, _SPLIT_RE, reject_dotdot = True)
            if split is None:
                continue
            options.add((config, split))
            if len(options) >= _MAX_OPTIONS:
                return


def _safe_json_file(
    path: Path,
    root: Path,
    *,
    allow_snapshot_symlink: bool = False,
) -> Any:
    try:
        if not path.is_file() or (path.is_symlink() and not allow_snapshot_symlink):
            return None
        resolved = path.resolve(strict = True)
        if not allow_snapshot_symlink:
            resolved.relative_to(root)
        size = resolved.stat().st_size
        if size <= 0 or size > _MAX_METADATA_BYTES:
            return None
        return json.loads(resolved.read_text(encoding = "utf-8"))
    except (OSError, RuntimeError, UnicodeError, ValueError):
        return None


def _processed_options(path: Path) -> set[tuple[str, str]]:
    options: set[tuple[str, str]] = set()
    visited = 0
    try:
        root = path.resolve(strict = True)
    except (OSError, RuntimeError):
        return options

    for directory, dirnames, filenames in os.walk(root, followlinks = False):
        base = Path(directory)
        try:
            relative = base.relative_to(root)
        except ValueError:
            dirnames[:] = []
            continue
        if len(relative.parts) >= _MAX_PROCESSED_WALK_DEPTH:
            dirnames[:] = []
        else:
            dirnames[:] = [
                name
                for name in dirnames
                if not (base / name).is_symlink() and not name.endswith(".incomplete")
            ]
        if "dataset_info.json" not in filenames:
            continue
        visited += 1
        if visited > _MAX_PROCESSED_METADATA_FILES:
            break
        if not any(
            name.lower().endswith(".arrow")
            and not (base / name).is_symlink()
            and (base / name).is_file()
            and not is_appledouble_metadata(base / name)
            for name in filenames
        ):
            continue
        payload = _safe_json_file(base / "dataset_info.json", root)
        fallback = relative.parts[0] if relative.parts else None
        _add_info_options(options, payload, fallback_config = fallback)
        if len(options) >= _MAX_OPTIONS:
            break
    return options


def _snapshot_metadata_file(snapshot: Path, name: str) -> Optional[Path]:
    path = resolved_dataset_snapshot_file(snapshot, name)
    if path is None:
        return None
    try:
        size = path.stat().st_size
    except OSError:
        return None
    return path if 0 < size <= _MAX_METADATA_BYTES else None


# DatasetCard.load raises on front matter datasets cannot parse, so nothing in the snapshot is loadable.
_UNPARSABLE_METADATA = object()


def _read_card_metadata(path: Path) -> Any:
    try:
        lines = path.read_text(encoding = "utf-8").splitlines()
        if not lines or lines[0].strip() != "---":
            return None
        end = next(index for index, line in enumerate(lines[1:], start = 1) if line.strip() == "---")
        from yaml import YAMLError, safe_load

        try:
            payload = safe_load("\n".join(lines[1:end]))
        except YAMLError:
            return _UNPARSABLE_METADATA
        if payload is None:
            # Empty, comment-only or null front matter, which RepoCard reads as an empty card rather than refusing.
            return {}
    except StopIteration:
        # Front matter that never closes is not front matter at all.
        return None
    except UnicodeError:
        # DatasetCard.load reads the card as utf-8 and raises, so nothing here loads.
        return _UNPARSABLE_METADATA
    except (ImportError, OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else _UNPARSABLE_METADATA


# Mirrors datasets' get_data_patterns (sharded data/{split}-NNNNN files, then directory keywords,
# then filename keywords, then one train split) without importing datasets, which
# dataset_cache.py forbids on cache paths.
# An offered split has to be trainable, and this is deliberately tighter in two places: a file
# whose symlink leaves the repository is refused, and only trainable extensions are offered.

# Past this the result would depend on traversal order, so nothing is offered.
_MAX_SNAPSHOT_DATA_FILES = 200_000
# datasets drops these by basename before it infers anything, so a metadata-only cache is empty
# rather than a bogus train split.
_IGNORED_DATA_FILENAMES = frozenset(
    {
        "README.md",
        "config.json",
        "dataset_info.json",
        "dataset_infos.json",
        "dataset_dict.json",
        "dummy_data.zip",
    }
)
# What fsspec can decompress in an Unsloth install, as suffixes that sit after the real one.
_COMPRESSION_EXTENSIONS = frozenset({".gz", ".gzip", ".bz2", ".xz", ".zip"})
# Named by datasets but needing codecs an Unsloth install does not ship, so they raise
# "Compression type not supported" and would put a dead split in the picker; .lzma is the legacy
# alone-format, for whose filter datasets registers .xz.
_UNREADABLE_COMPRESSION = frozenset({".zst", ".zstd", ".lz4", ".lzma"})
# datasets picks one builder for the whole dataset, and splits that disagree make it raise, so a
# snapshot mixing formats is not offerable at all.
_MODULE_EXTENSIONS = {
    "arrow": ".arrow",
    "csv": ".csv",
    # datasets compares the whole builder result, and tsv carries sep="\t", so it is not csv.
    "csv+tab": ".tsv",
    "json": ".json .jsonl .ndjson",
    "parquet": ".parquet .geoparquet .gpq",
    "text": ".txt",
    "webdataset": ".tar",
    "xml": ".xml",
    "hdf5": ".h5 .hdf5",
}
# datasets registers only the folder builders in both letter cases, so only their extensions match case-insensitively.
_FOLDER_EXTENSIONS = frozenset(
    ".apng .blp .bmp .bufr .bw .cur .dcx .dds .dib .emf .eps .fit .fits .flc .fli .ftc .ftu "
    ".gbr .gif .grib .icb .icns .ico .iim .im .j2c .j2k .jfif .jp2 .jpc .jpe .jpeg .jpf "
    ".jpg .jpx .msp .pbm .pcd .pcx .pgm .png .pnm .ppm .ps .psd .pxr .ras .rgb .rgba .sgi "
    ".tga .tif .tiff .vda .vst .webp .wmf .xbm .xpm "
    ".3g2 .3gp .aiff .asf .au .avr .caf .f4v .flac .flv .htk .ircam .m4v .mat4 .mat5 .mp3 "
    ".mpc2k .mpg .mxf .nist .nut .ogg .ogm .opus .paf .pvf .raw .rf64 .sd2 .sds .svx .voc "
    ".w64 .wav .wavex .webm .wma .wmv .wve .xi "
    ".avi .mkv .mov .mp4 .mpeg .pdf".split()
)
_EXTENSION_MODULES = {
    extension: module for module, names in _MODULE_EXTENSIONS.items() for extension in names.split()
}
# datasets infers a split's module from its first 200 files, in resolved (sorted) order.
_MAX_MODULE_INFERENCE_FILES = 200
# datasets' tie-break once the counts are level, then the extension string itself.
# TRAINING_DATA_EXTS only ever resolves to these, so any other builder means the snapshot holds
# nothing trainable.
_TRAINABLE_MODULES = frozenset({"csv", "json", "parquet"})
_ROW_PROBE_BYTES = 8192
_EXTENSION_PRIORITY = (".parquet", ".jsonl", ".json", ".csv")
# Folder-builder metadata loses every tie-break, so it never decides a split's builder.
_METADATA_FILENAMES = frozenset({"metadata.csv", "metadata.jsonl", "metadata.parquet"})
# datasets' split keywords, in the order it resolves them.
_SPLIT_KEYWORDS = {
    "train": ("train", "training"),
    "validation": ("validation", "valid", "dev", "val"),
    "test": ("test", "testing", "eval", "evaluation"),
}
# Its keyword globs as regexes: "sep" is NON_WORDS_CHARS and * never crosses a directory.
_SEP = "[-._ 0-9]"
_FILENAME_KEYWORD_PATTERNS = (
    r"(?:.*/)?{keyword}%s[^/]*" % _SEP,
    r"(?:.*/)?[^/]*%s{keyword}%s[^/]*" % (_SEP, _SEP),
)
_DIR_NAME_KEYWORD_PATTERNS = (
    r"(?:.*/)?{keyword}/.*",
    r"(?:.*/)?{keyword}%s[^/]*/.*" % _SEP,
    r"(?:.*/)?[^/]*%s{keyword}/.*" % _SEP,
    r"(?:.*/)?[^/]*%s{keyword}%s[^/]*/.*" % (_SEP, _SEP),
)
# data/{split}-NNNNN-of-NNNNN*.*, where a * matches nothing as happily as something.
_SHARDED_DATA_RE = re.compile(r"^data/(?P<split>[^/]*)-[0-9]{5}-of-[0-9]{5}[^/]*\.[^/]*$")
# datasets' own split grammar is stricter than the one the picker accepts: a shard named outside it
# makes the whole snapshot unloadable rather than falling through.
_SHARD_SPLIT_RE = re.compile(r"^\w+(\.\w+)*$")


def _keyword_patterns(bases: tuple[str, ...]) -> dict[str, list["re.Pattern[str]"]]:
    return {
        split: [
            re.compile(base.format(keyword = re.escape(keyword)) + r"\Z")
            for keyword in keywords
            for base in bases
        ]
        for split, keywords in _SPLIT_KEYWORDS.items()
    }


_DIR_NAME_SPLITS = _keyword_patterns(_DIR_NAME_KEYWORD_PATTERNS)
_FILENAME_SPLITS = _keyword_patterns(_FILENAME_KEYWORD_PATTERNS)


def _data_suffix(name: str) -> Optional[str]:
    """The suffix that decides this file's builder, or None when it has none.

    datasets reads the whole suffix chain, so records.parquet.backup is still parquet, and
    a trailing compression suffix is stripped before the rest is considered.
    """
    suffixes = PurePosixPath(name).suffixes
    if suffixes and suffixes[-1].lower() in _COMPRESSION_EXTENSIONS | _UNREADABLE_COMPRESSION:
        suffixes = suffixes[:-1]
    for suffix in reversed(suffixes):
        if suffix in _EXTENSION_MODULES or suffix.lower() in _FOLDER_EXTENSIONS:
            return suffix
    return suffixes[-1] if suffixes else None


def _file_module(name: str) -> Optional[str]:
    """The builder datasets would pick for this filename.

    Its globs are case-sensitive, so TRAIN.JSONL is a file it never resolves. The folder
    builders are the exception: those are registered in both cases.
    """
    suffix = _data_suffix(name)
    if suffix is None:
        return None
    if suffix.lower() in _FOLDER_EXTENSIONS:
        return "folder"
    return _EXTENSION_MODULES.get(suffix)


def _trainable_name(name: str) -> bool:
    """Whether Unsloth can train on this file. Narrower than what datasets would read."""
    suffix = _data_suffix(name)
    return suffix is not None and suffix in TRAINING_DATA_EXTS


def _snapshot_data_files(snapshot: Path) -> Optional[list[PurePosixPath]]:
    """Every file datasets would consider, relative to the snapshot and sorted the way
    fsspec returns them, or None when the snapshot is too large to compare with."""
    found: list[PurePosixPath] = []
    try:
        root = snapshot.resolve(strict = True)
    except (OSError, RuntimeError, ValueError):
        return found
    for directory, dirnames, filenames in os.walk(root):
        base = Path(directory)
        try:
            relative = base.relative_to(root)
        except ValueError:
            dirnames[:] = []
            continue
        # datasets hides dot and __ directories from its own patterns.
        dirnames[:] = [name for name in dirnames if not name.startswith((".", "__"))]
        for filename in filenames:
            if filename in _IGNORED_DATA_FILENAMES or filename.startswith("."):
                continue
            # resolve_pattern keeps a link only when its target is a file, so a dangling one is not a file
            # datasets sees.
            if not (base / filename).is_file():
                continue
            # Files with no builder of their own are kept: they cannot win the vote, but a split holding nothing
            # else is one datasets refuses to build.
            if len(found) >= _MAX_SNAPSHOT_DATA_FILES:
                # Past the cap this is a traversal-order prefix, which cannot be compared with what the loader would
                # resolve.
                return None
            found.append(PurePosixPath((relative / filename).as_posix()))
    found.sort(key = lambda path: path.as_posix())
    return found


def _sharded_splits(files: list[PurePosixPath]) -> Optional[dict[str, list[PurePosixPath]]]:
    """The sharded stage, or None when a shard name is one datasets refuses outright."""
    grouped: dict[str, list[PurePosixPath]] = {}
    for path in files:
        matched = _SHARDED_DATA_RE.match(path.as_posix())
        if matched is None:
            continue
        split = matched.group("split")
        if _SHARD_SPLIT_RE.fullmatch(split) is None:
            # datasets raises on the name rather than moving on to the next stage.
            return None
        grouped.setdefault(split, []).append(path)
    return grouped


def _keyword_splits(
    files: list[PurePosixPath], patterns: dict[str, list["re.Pattern[str]"]]
) -> dict[str, list[PurePosixPath]]:
    grouped: dict[str, list[PurePosixPath]] = {}
    for split, expressions in patterns.items():
        matched = [
            path
            for path in files
            if any(expression.match(path.as_posix()) for expression in expressions)
        ]
        if matched:
            grouped[split] = matched
    return grouped


def _grouped_splits(files: list[PurePosixPath]) -> Optional[dict[str, list[PurePosixPath]]]:
    """The splits datasets would resolve, by the first stage that names any."""
    grouped = _sharded_splits(files)
    if grouped is None:
        return None
    for stage in (_DIR_NAME_SPLITS, _FILENAME_SPLITS):
        if grouped:
            break
        grouped = _keyword_splits(files, stage)
    return grouped or {"train": files}


def _one_module(grouped: dict[str, list[PurePosixPath]]) -> Optional[str]:
    """The single builder every split agrees on, or None when datasets would refuse them.

    It counts a split's files by extension and takes the winner, so a split with no clear
    winner, or two splits wanting different builders, is a dataset it cannot build.
    """
    modules = set()
    for entries in grouped.values():
        counts: dict[tuple[str, str], int] = {}
        for path in entries[:_MAX_MODULE_INFERENCE_FILES]:
            suffix = _data_suffix(path.name)
            module = _file_module(path.name)
            # Folder metadata is counted last whatever it is, so it never wins on its own.
            if module is not None and path.name not in _METADATA_FILENAMES:
                counts[(module, suffix or "")] = counts.get((module, suffix or ""), 0) + 1
        if not counts:
            return None
        best = max(
            counts,
            key = lambda key: (
                counts[key],
                -_EXTENSION_PRIORITY.index(key[1]) if key[1] in _EXTENSION_PRIORITY else -99,
                key[1],
            ),
        )
        modules.add(best[0])
    return modules.pop() if len(modules) == 1 else None


def _blocked_by_compression(name: str, module: str) -> bool:
    """Whether datasets keeps this file for the chosen builder but cannot decompress it.
    Its extension filter reads the whole basename, so train.jsonl.zst still looks like
    json to it, and the build then dies on the missing codec."""
    stem, _, suffix = name.rpartition(".")
    if not stem or f".{suffix}".lower() not in _UNREADABLE_COMPRESSION:
        return False
    return _file_module(stem) == module


def _offerable(entries: list[PurePosixPath], snapshot: Path, module: str) -> Optional[bool]:
    """True when the split holds data Unsloth can train on, False when it holds none, None
    when the config is unusable. datasets reads every file in the split, so one file that
    escapes the cache, or one the builder chokes on, condemns the config rather than
    the single file."""
    trainable = False
    empty = False
    for path in entries:
        resolved = resolved_dataset_snapshot_file(snapshot, path.as_posix())
        if resolved is None or _blocked_by_compression(path.name, module):
            return None
        # Once a builder is chosen datasets reads only what that builder claims, so a training file left
        # behind by a folder builder is not data this split offers.
        if _file_module(path.name) != module or not _trainable_name(path.name):
            continue
        if _empty_payload(resolved):
            empty = True
            continue
        if _rowless(resolved, path.name, module):
            continue
        trainable = True
    # Every builder but json fails outright on a file with no rows, and datasets prepares every split
    # before handing one back, so such a file condemns its siblings too.
    if empty and module != "json":
        return None
    return trainable


def _rowless(path: Path, name: str, module: str) -> bool:
    """Whether the builder would read this file and find no row in it, as it does for a
    csv holding only its header. That split is dropped, but unlike a file it cannot read
    at all, datasets still builds the rest of the dataset around it."""
    suffixes = PurePosixPath(name).suffixes
    if module not in {"csv", "json"} or (
        suffixes and suffixes[-1].lower() in _COMPRESSION_EXTENSIONS | _UNREADABLE_COMPRESSION
    ):
        # Compressed bytes say nothing about the rows inside without decompressing them.
        return False
    try:
        with path.open("rb") as handle:
            head = handle.read(_ROW_PROBE_BYTES)
    except OSError:
        return True
    if module == "json":
        # A canonical empty container parses fine and yields no row.
        return head.strip() in (b"", b"[]", b"{}")
    # A header with no row under it. Anything longer than the probe has one.
    return len(head) < _ROW_PROBE_BYTES and len([x for x in head.splitlines() if x.strip()]) < 2


def _empty_payload(path: Path) -> bool:
    """Whether the builder would read no bytes out of this file."""
    try:
        return path.stat().st_size == 0
    except OSError:
        return True


def _inferred_snapshot_options(snapshot: Path) -> set[tuple[str, str]]:
    """What the picker can offer for a snapshot whose metadata named nothing."""
    files = _snapshot_data_files(snapshot)
    if not files:
        return set()
    grouped = _grouped_splits(files)
    module = _one_module(grouped) if grouped is not None else None
    if grouped is None or module is None or module not in _TRAINABLE_MODULES:
        return set()
    offerable = {split: _offerable(entries, snapshot, module) for split, entries in grouped.items()}
    if any(state is None for state in offerable.values()):
        return set()
    return {
        ("default", name)
        for split, state in offerable.items()
        if state and (name := _valid_option(split, _SPLIT_RE, reject_dotdot = True)) is not None
    }


def _metadata_present(snapshot: Path, name: str) -> bool:
    """Whether the snapshot holds this metadata file at all, readable or not."""
    try:
        return (snapshot / name).is_file()
    except OSError:
        return False


def _nonempty(snapshot: Path, name: str) -> bool:
    """Whether the file holds any bytes. One that holds none declares nothing."""
    try:
        return (snapshot / name).stat().st_size > 0
    except OSError:
        return True


def _unreadable_metadata(snapshot: Path, name: str) -> bool:
    """Whether the file is there but _snapshot_metadata_file refused it, for being too big
    or for pointing out of the cache. The loader still reads it, so we cannot ignore it.
    An empty file is refused too, but it declares nothing and so blocks nothing."""
    return (
        _metadata_present(snapshot, name)
        and _snapshot_metadata_file(snapshot, name) is None
        and _nonempty(snapshot, name)
    )


def _declares_configs(snapshot: Path, name: str) -> bool:
    """Whether this standalone yaml file names configs the loader would build. Unreadable
    counts as declaring, since datasets reads it as a card and we cannot see what it says."""
    if not _metadata_present(snapshot, name):
        return False
    path = _snapshot_metadata_file(snapshot, name)
    if path is None:
        return _nonempty(snapshot, name)
    try:
        from yaml import YAMLError, safe_load
        try:
            payload = safe_load(path.read_text(encoding = "utf-8"))
        except YAMLError:
            return True
    except (ImportError, OSError, UnicodeError, ValueError):
        return True
    if not isinstance(payload, dict):
        # DatasetCardData updates a dict from it, which raises on anything else.
        return bool(payload)
    # Only configs count: 4.3.0 builds no config from dataset_info declared here, so a feature schema
    # leaves the loader inferring the files by pattern.
    return bool(payload.get("configs"))


def _declares_splits(payload: Any) -> bool:
    """Whether dataset_info states its splits. An empty statement is still authoritative:
    datasets exposes no config at all rather than falling back to the files."""
    entries = payload if isinstance(payload, list) else [payload]
    return any(isinstance(item, dict) and "splits" in item for item in entries)


def _malformed_info(payload: Any) -> bool:
    """A dataset_info list holding anything but mappings, which datasets calls .get on."""
    return isinstance(payload, list) and any(not isinstance(item, dict) for item in payload)


def _snapshot_options(snapshot: Path) -> set[tuple[str, str]]:
    options: set[tuple[str, str]] = set()

    # Metadata we cannot read still names the loader's configs, so inference must not step in beside it:
    # a card too large or unsafe to open, or the standalone yaml file.
    declared = _unreadable_metadata(snapshot, "README.md") or _declares_configs(
        snapshot, ".huggingface.yaml"
    )
    readme = _snapshot_metadata_file(snapshot, "README.md")
    if readme is not None:
        card_data = _read_card_metadata(readme)
        if card_data is _UNPARSABLE_METADATA:
            # datasets raises out of DatasetCard.load, so no option here would ever start.
            return options
        if isinstance(card_data, dict):
            # datasets merges the standalone YAML into the card, so a README that declares nothing does not undo
            # a declaration made there.
            declared = declared or bool(card_data.get("configs"))
            _add_config_options(options, card_data.get("configs"))
            named = len(options)
            info = card_data.get("dataset_info")
            _add_dataset_info_options(options, info)
            # dataset_info carrying only a feature schema names no config, so datasets still resolves the files
            # by pattern and inference has to run.
            declared = (
                declared or len(options) > named or _malformed_info(info) or _declares_splits(info)
            )

    for filename in ("dataset_infos.json", "dataset_info.json"):
        metadata = _snapshot_metadata_file(snapshot, filename)
        if metadata is None:
            # datasets json.loads dataset_infos.json unconditionally while resolving configs, so one it cannot
            # read raises before any split exists, inferred or not.
            declared = declared or (
                filename == "dataset_infos.json" and _metadata_present(snapshot, filename)
            )
            continue
        payload = _safe_json_file(metadata, snapshot, allow_snapshot_symlink = True)
        if filename == "dataset_infos.json":
            # datasets json.loads this one while resolving configs, so a file it cannot parse raises before any
            # split exists, inferred or not.
            declared = declared or payload is None
            _add_dataset_info_options(options, payload)
        else:
            _add_info_options(options, payload)

    if not options and not declared:
        # Nothing was declared, the case #8140 reports: the loader still resolves the files by pattern, so
        # the picker infers the same splits.
        options.update(_inferred_snapshot_options(snapshot))
    return options


def _sorted_options(options: set[tuple[str, str]], dataset: str = "") -> list[DatasetSplitOption]:
    ordered = sorted(
        options,
        key = lambda item: (
            item[0].casefold() != "default",
            item[0].casefold(),
            item[1].casefold() != "train",
            item[1].casefold(),
            item,
        ),
    )
    return [
        DatasetSplitOption(dataset = dataset, config = config, split = split)
        for config, split in ordered[:_MAX_OPTIONS]
    ]


def local_dataset_options(request: LocalDatasetOptionsRequest) -> LocalDatasetOptionsResponse:
    repo_id = request.dataset_name.strip()
    if not is_valid_repo_id(repo_id):
        return LocalDatasetOptionsResponse(cache_available = False, splits = [])

    selected = (
        dataset_cache_path_from_cache_path(request.local_path, repo_id)
        if request.local_path
        else latest_cached_dataset_path(repo_id)
    )
    if selected is None:
        return LocalDatasetOptionsResponse(cache_available = False, splits = [])

    processed = processed_dataset_cache_path(str(selected), repo_id)
    if processed is not None:
        options = _processed_options(processed)
    else:
        snapshot = dataset_snapshot_from_cache_path(str(selected), repo_id)
        options = _snapshot_options(snapshot) if snapshot is not None else set()

    splits = _sorted_options(options, repo_id)
    return LocalDatasetOptionsResponse(cache_available = True, splits = splits)
