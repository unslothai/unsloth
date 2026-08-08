# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import json
import os
import re
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable, Optional

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


_MAX_METADATA_BYTES = 2 * 1024 * 1024
_MAX_PROCESSED_METADATA_FILES = 256
_MAX_PROCESSED_WALK_DEPTH = 4
_MAX_OPTIONS = 2048
_MAX_OPTION_LENGTH = 128
# Only names are read during the scan, so this can sit well above any real snapshot. Past it
# the scan is truncated and cannot be compared with the loader, so nothing is offered.
_MAX_SNAPSHOT_DATA_FILES = 200_000
_COMPRESSION_EXTENSIONS = ("", ".gz", ".bz2", ".xz", ".zst", ".zip")
_SPLIT_KEYWORDS = {
    "train": frozenset({"train", "training"}),
    "validation": frozenset({"validation", "valid", "dev", "val"}),
    "test": frozenset({"test", "testing", "eval", "evaluation"}),
}
_SHARDED_DATA_RE = re.compile(r"^data/(?P<split>[^/]+)-[0-9]{5}-of-[0-9]{5}[^/]*\.[^/]+$")
# datasets drops these by basename (FILES_TO_IGNORE) before it infers anything, so a
# metadata-only cache is empty rather than a bogus train split.
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
_INDETERMINATE_MODULE = "?"


def _exts(names: str) -> frozenset:
    return frozenset(names.split())


# datasets' own extension table, by builder. The folder builders are the only ones it
# registers in both cases, so those match case-insensitively and the rest stay case-exact.
_MODULE_EXTENSIONS = {
    "arrow": ".arrow",
    "csv": ".csv",
    # datasets compares the whole builder result and tsv carries sep="\t", so it is not csv.
    "csv+tab": ".tsv",
    "hdf5": ".h5 .hdf5",
    "json": ".json .jsonl .ndjson",
    "parquet": ".parquet .geoparquet .gpq",
    "text": ".txt",
    "webdataset": ".tar",
    "xml": ".xml",
    "imagefolder": (
        ".apng .blp .bmp .bufr .bw .cur .dcx .dds .dib .emf .eps .fit .fits .flc .fli .ftc "
        ".ftu .gbr .gif .grib .icb .icns .ico .iim .im .j2c .j2k .jfif .jp2 .jpc .jpe .jpeg "
        ".jpf .jpg .jpx .msp .pbm .pcd .pcx .pgm .png .pnm .ppm .ps .psd .pxr .ras .rgb "
        ".rgba .sgi .tga .tif .tiff .vda .vst .webp .wmf .xbm .xpm"
    ),
    "audiofolder": (
        ".3g2 .3gp .aiff .asf .au .avr .caf .f4v .flac .flv .htk .ircam .m4v .mat4 .mat5 "
        ".mp3 .mpc2k .mpg .mxf .nist .nut .ogg .ogm .opus .paf .pvf .raw .rf64 .sd2 .sds "
        ".svx .voc .w64 .wav .wavex .webm .wma .wmv .wve .xi"
    ),
    "videofolder": ".avi .mkv .mov .mp4 .mpeg",
    "pdffolder": ".pdf",
    # datasets reads a zip to pick its module. We do not open archives, so a split a zip
    # would decide is unknowable and the snapshot is left alone.
    _INDETERMINATE_MODULE: ".zip",
}
_EXTENSION_MODULES = {
    extension: module for module, names in _MODULE_EXTENSIONS.items() for extension in names.split()
}
_CASE_INSENSITIVE_MODULES = _exts("imagefolder audiofolder videofolder pdffolder")
# datasets' tie-break after the count, before falling back to the extension string itself.
_EXTENSION_PRIORITY = (".parquet", ".jsonl", ".json", ".csv")
# Folder-builder metadata loses every tie-break in datasets, so it never picks a split's
# module. This is the pinned 4.3 list; 3.4 flagged only the csv and jsonl names.
_METADATA_FILENAMES = frozenset({"metadata.csv", "metadata.jsonl", "metadata.parquet"})
# datasets infers a split's module from its first 200 files, in resolved (sorted) order.
_MAX_MODULE_INFERENCE_FILES = 200
# _read_card_metadata returns this when a card exists but its YAML does not parse. datasets
# lets that error out of DatasetCard.load, so nothing in the snapshot is loadable.
_UNPARSABLE_METADATA = object()
_STANDALONE_YAML = ".huggingface.yaml"
# A snapshot file and the module datasets would build it with, None when it is not data.
_DataFile = tuple[PurePosixPath, Optional[str]]
_CONFIG_RE = re.compile(r"[^<>:/\\|?*\x00-\x1f\x7f]+")
# Also datasets' own _split_re, so a sharded name it would reject never reaches the picker.
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


def _declared_configs(payload: Any) -> Any:
    """Declared config -> data_dir, or _UNPARSABLE_METADATA when datasets would refuse the card.

    datasets keys its metadata configs by name, so a repeated name is last-wins, and it infers
    each config under its own data_dir.
    """
    if payload is None:
        return {}
    if not isinstance(payload, list):
        return _UNPARSABLE_METADATA
    declared: dict[str, str] = {}
    for item in payload[:_MAX_OPTIONS]:
        if not isinstance(item, dict) or not isinstance(item.get("config_name"), str):
            return _UNPARSABLE_METADATA
        name = _valid_option(item.get("config_name"), _CONFIG_RE)
        if name is None:
            continue
        data_dir = item.get("data_dir")
        if data_dir is None:
            declared[name] = ""
        elif isinstance(data_dir, str) and "\\" not in data_dir and ".." not in data_dir:
            declared[name] = data_dir
        else:
            # Rewriting the scope would silently widen it, so drop the config instead.
            declared.pop(name, None)
    return declared


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


def _read_card_metadata(path: Path) -> Optional[dict[str, Any]]:
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
    except (ImportError, OSError, UnicodeError, ValueError, StopIteration):
        return None
    return payload if isinstance(payload, dict) else None


def _snapshot_card_data(snapshot: Path) -> Any:
    """The card datasets would build: README front matter, then .huggingface.yaml over it."""
    card: dict[str, Any] = {}
    readme = _snapshot_metadata_file(snapshot, "README.md")
    if readme is not None:
        payload = _read_card_metadata(readme)
        if payload is _UNPARSABLE_METADATA:
            return _UNPARSABLE_METADATA
        if isinstance(payload, dict):
            card.update(payload)

    standalone = _snapshot_metadata_file(snapshot, _STANDALONE_YAML)
    if standalone is not None:
        try:
            from yaml import YAMLError, safe_load
            try:
                payload = safe_load(standalone.read_text(encoding = "utf-8"))
            except YAMLError:
                return _UNPARSABLE_METADATA
        except (ImportError, OSError, UnicodeError, ValueError):
            payload = None
        if isinstance(payload, dict):
            card.update(payload)
    return card


def _has_snapshot_data_extension(filename: str) -> bool:
    # Case-sensitive on every platform: datasets globs through fsspec, whose matcher is a
    # plain regex with no normcase, so a .JSONL file is unsupported even on Windows.
    return any(
        filename.endswith(extension + compression)
        for extension in TRAINING_DATA_EXTS
        for compression in _COMPRESSION_EXTENSIONS
    )


def _extension_module(extension: str) -> Optional[str]:
    """The builder datasets registers for one suffix, honouring its case rules."""
    module = _EXTENSION_MODULES.get(extension)
    if module is not None:
        return module
    module = _EXTENSION_MODULES.get(extension.lower())
    return module if module in _CASE_INSENSITIVE_MODULES else None


def _file_module(filename: str) -> Optional[str]:
    """The builder datasets would reach for, or None when the file is not data to it."""
    for suffix in filename.split(".")[1:]:
        module = _extension_module("." + suffix)
        if module is not None:
            return module
    return None


def _snapshot_data_files(snapshot: Path) -> Optional[list[_DataFile]]:
    """Every file datasets would see, or None when the snapshot is too large to judge.

    Nothing is resolved here. datasets picks a split pattern over all non-ignored files and
    only then drops the unsupported ones, so a `test/notes.bin` decides which stage wins even
    though it can never be trained on. Resolution is deferred to the files actually offered.
    """
    files: list[_DataFile] = []
    try:
        root = snapshot.resolve(strict = True)
    except (OSError, RuntimeError):
        return files

    for directory, dirnames, filenames in os.walk(root, followlinks = False):
        base = Path(directory)
        try:
            relative = base.relative_to(root)
        except ValueError:
            dirnames[:] = []
            continue
        dirnames[:] = [
            name
            for name in dirnames
            if not name.startswith((".", "__")) and not (base / name).is_symlink()
        ]
        for filename in filenames:
            # datasets hides dot files and `__` directories, but not `__` filenames.
            if filename.startswith(".") or filename in _IGNORED_DATA_FILENAMES:
                continue
            if len(files) >= _MAX_SNAPSHOT_DATA_FILES:
                return None
            files.append((PurePosixPath((relative / filename).as_posix()), _file_module(filename)))
    return files


def _keyword_splits(parts: Iterable[str]) -> dict[str, int]:
    """Splits named by these path parts, and how many of the split's keyword patterns hit.

    datasets resolves every keyword pattern separately and concatenates, so a name carrying
    two synonyms of one split contributes that file twice to its extension count.
    """
    tokens = {token for part in parts for token in re.split(r"[-._ 0-9]+", part) if token}
    matched = {split: len(tokens & keywords) for split, keywords in _SPLIT_KEYWORDS.items()}
    return {split: hits for split, hits in matched.items() if hits}


def _sharded_split_files(files: Iterable[_DataFile]) -> Optional[dict[str, list[_DataFile]]]:
    """Sharded splits, or None when a name datasets rejects makes the snapshot unloadable."""
    grouped: dict[str, list[_DataFile]] = {}
    for entry in files:
        match = _SHARDED_DATA_RE.fullmatch(entry[0].as_posix())
        if match is None:
            continue
        split = _valid_option(match.group("split"), _SPLIT_RE, reject_dotdot = True)
        if split is None:
            return None
        grouped.setdefault(split, []).append(entry)
    return grouped


def _keyword_split_files(
    files: Iterable[_DataFile], naming: Callable[[PurePosixPath], Iterable[str]]
) -> dict[str, list[_DataFile]]:
    grouped: dict[str, list[_DataFile]] = {}
    for entry in files:
        for split, hits in _keyword_splits(naming(entry[0])).items():
            grouped.setdefault(split, []).extend([entry] * hits)
    return grouped


def _split_module(files: Iterable[_DataFile]) -> Optional[str]:
    """The one module datasets would build this split with, counting and ranking as it does."""
    counts: dict[tuple[bool, str], int] = {}
    for path, module in sorted(files)[:_MAX_MODULE_INFERENCE_FILES]:
        if module is None:
            continue
        is_metadata = path.name in _METADATA_FILENAMES
        # Every suffix counts, not just the first that resolves, and the counter is folded to
        # lower case even though what may reach it is not.
        for suffix in path.name.split(".")[1:]:
            if _extension_module("." + suffix) is None:
                continue
            key = (is_metadata, "." + suffix.lower())
            counts[key] = counts.get(key, 0) + 1
    if not counts:
        return None
    best = max(
        counts,
        key = lambda key: (
            not key[0],
            counts[key],
            *(key[1] == extension for extension in _EXTENSION_PRIORITY),
            key[1],
        ),
    )
    return _EXTENSION_MODULES[best[1]]


def _offerable_split(entries: Iterable[_DataFile], snapshot: Path, root: str, module: str) -> bool:
    """A split is offerable when it holds trainable data and every file the builder would
    read stays inside the cache. One safe file is not enough: datasets loads them all."""
    trainable = False
    for path, file_module in entries:
        if file_module != module:
            continue
        if resolved_dataset_snapshot_file(snapshot, root + path.as_posix()) is None:
            return False
        trainable = trainable or _has_snapshot_data_extension(path.name)
    return trainable


def _inferred_snapshot_options(
    snapshot: Path, configs: Iterable[tuple[str, str]] = (("default", ""),)
) -> set[tuple[str, str]]:
    """Mirror datasets' default local-file split inference without importing it.

    Known gaps, all of which hide an option rather than offer a dead one: names longer than
    _MAX_OPTION_LENGTH are dropped because the picker cannot start them, splits made only of
    files outside TRAINING_DATA_EXTS are not offered, a zip's module is left unknown rather
    than read out of the archive, and external symlinks stay rejected for cache safety.
    """
    options: set[tuple[str, str]] = set()
    for config, data_dir in configs:
        root = data_dir.strip("/")
        files = _snapshot_data_files(snapshot / root if root else snapshot)
        if not files:
            continue

        grouped = _sharded_split_files(files)
        if grouped is None:
            continue
        if not grouped:
            grouped = _keyword_split_files(files, lambda path: path.parent.parts)
        if not grouped:
            grouped = _keyword_split_files(files, lambda path: (path.name,))
        if not grouped:
            grouped = {"train": files}

        # A split with no data at all makes datasets raise, and one whose module we cannot
        # pin down could disagree with the others, so neither leaves anything to offer.
        modules = {_split_module(entries) for entries in grouped.values()}
        if len(modules) != 1 or modules & {None, _INDETERMINATE_MODULE}:
            continue
        module = modules.pop()
        prefix = root + "/" if root else ""
        options.update(
            (config, split)
            for split, entries in grouped.items()
            if _offerable_split(entries, snapshot, prefix, module)
        )
    return options


def _snapshot_options(snapshot: Path) -> set[tuple[str, str]]:
    options: set[tuple[str, str]] = set()

    card_data = _snapshot_card_data(snapshot)
    if card_data is _UNPARSABLE_METADATA:
        # datasets raises out of DatasetCard.load, so no option here would ever start.
        return options
    declared_configs = _declared_configs(card_data.get("configs"))
    if declared_configs is _UNPARSABLE_METADATA:
        # datasets raises building MetadataConfigs, well before it ever looks at a file.
        return options
    _add_config_options(options, card_data.get("configs"))
    _add_dataset_info_options(options, card_data.get("dataset_info"))

    for filename in ("dataset_infos.json", "dataset_info.json"):
        metadata = _snapshot_metadata_file(snapshot, filename)
        if metadata is None:
            continue
        payload = _safe_json_file(metadata, snapshot, allow_snapshot_symlink = True)
        if filename == "dataset_infos.json":
            _add_dataset_info_options(options, payload)
        else:
            _add_info_options(options, payload)
    # datasets infers patterns per config, so a config with no data_files still gets them even
    # when a sibling config declared its own, and it builds under that config's name.
    pending = [
        (config, data_dir)
        for config, data_dir in declared_configs.items()
        if not any(existing == config for existing, _split in options)
    ]
    if declared_configs and pending:
        options.update(_inferred_snapshot_options(snapshot, pending))
    elif not options and not isinstance(card_data.get("configs"), list):
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
