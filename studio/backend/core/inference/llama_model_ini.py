# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Optional ``unsloth.ini`` shipped beside a GGUF, compiled to llama-server pass-through args.

The file uses llama.cpp's preset INI format (``common/preset.cpp``, ``docs/preset.md``), but it is
NOT named ``preset.ini``: llama.cpp's ``-hf`` fetches a repo-root ``preset.ini`` instead of the
GGUFs (``common/download.cpp``), so that name would change what plain llama.cpp users get.
The file can come from any Hugging Face repo, so only an allowlist of performance and sampling
options is applied; everything else is reported in ``ignored`` and never reaches the command.
The tokens go through the same path as typed extra arguments (after Studio's flags, last wins)."""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

MODEL_INI_FILENAME = "unsloth.ini"
MAX_MODEL_INI_BYTES = 64 * 1024

_CACHE_TYPES = frozenset({"f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1"})
_SPEC_TYPES = frozenset(
    {
        "none",
        "draft-simple",
        "draft-eagle3",
        "draft-mtp",
        "draft-dflash",
        "draft-dspark",
        "ngram-simple",
        "ngram-map-k",
        "ngram-map-k4v",
        "ngram-mod",
        "ngram-cache",
    }
)
_TRUE = frozenset({"on", "enabled", "true", "1"})
_FALSE = frozenset({"off", "disabled", "false", "0"})

# Router / preset-only keys llama.cpp itself skips for a single model (common/arg.cpp
# common_params_add_preset_options); ignored without a report, as there.
_ROUTER_ONLY = frozenset({"load-on-startup", "stop-timeout", "dedup-cache-models"})

# Studio picks the model and projector from the model picker; an INI naming them (as a
# hand-written router preset does) is normal, so these are reported, not refused.
_STUDIO_SUPPLIED = frozenset(
    {"m", "model", "mm", "mmproj", "hf", "hfr", "hf-repo", "hff", "hf-file", "a", "alias"}
)


@dataclass(frozen = True)
class _Option:
    flag: str  # canonical long flag emitted
    names: tuple[str, ...]  # INI spellings without leading dashes
    env: Optional[str]
    kind: str
    lo: float = -math.inf
    hi: float = math.inf
    choices: frozenset[str] = frozenset()
    negative: Optional[str] = None  # flag emitted for a false switch
    negative_names: tuple[str, ...] = ()  # INI spellings whose truth value is inverted


def _opt(flag, names, env, kind, **kw) -> _Option:
    return _Option(flag, tuple(names), env, kind, **kw)


# Spellings and env names from `llama-server --help` at llama.cpp 10a60cf (common/arg.cpp).
_OPTIONS: tuple[_Option, ...] = (
    _opt("--ctx-size", ("c", "ctx-size"), "LLAMA_ARG_CTX_SIZE", "int", lo = 0, hi = 2**31 - 1),
    _opt("--batch-size", ("b", "batch-size"), "LLAMA_ARG_BATCH", "int", lo = 1, hi = 65536),
    _opt("--ubatch-size", ("ub", "ubatch-size"), "LLAMA_ARG_UBATCH", "int", lo = 1, hi = 65536),
    _opt("parallel", ("np", "parallel", "n-parallel"), "LLAMA_ARG_N_PARALLEL", "parallel"),
    _opt("--threads", ("t", "threads"), "LLAMA_ARG_THREADS", "int", lo = -1, hi = 4096),
    _opt("--threads-batch", ("tb", "threads-batch"), None, "int", lo = -1, hi = 4096),
    _opt(
        "--gpu-layers",
        ("ngl", "gpu-layers", "n-gpu-layers"),
        "LLAMA_ARG_N_GPU_LAYERS",
        "layers",
    ),
    _opt("--fit", ("fit",), "LLAMA_ARG_FIT", "choice", choices = frozenset({"on", "off"})),
    _opt("--fit-ctx", ("fitc", "fit-ctx"), "LLAMA_ARG_FIT_CTX", "int", lo = 0, hi = 2**31 - 1),
    _opt("--fit-target", ("fitt", "fit-target"), "LLAMA_ARG_FIT_TARGET", "int_list"),
    _opt("--n-cpu-moe", ("ncmoe", "n-cpu-moe"), "LLAMA_ARG_N_CPU_MOE", "int", lo = 0, hi = 4096),
    _opt("--cpu-moe", ("cmoe", "cpu-moe"), "LLAMA_ARG_CPU_MOE", "switch"),
    _opt("--override-tensor", ("ot", "override-tensor"), "LLAMA_ARG_OVERRIDE_TENSOR", "pattern"),
    _opt(
        "--split-mode",
        ("sm", "split-mode"),
        "LLAMA_ARG_SPLIT_MODE",
        "choice",
        choices = frozenset({"none", "layer", "row", "tensor"}),
    ),
    _opt("--tensor-split", ("ts", "tensor-split"), "LLAMA_ARG_TENSOR_SPLIT", "float_list"),
    _opt("--main-gpu", ("mg", "main-gpu"), "LLAMA_ARG_MAIN_GPU", "int", lo = 0, hi = 256),
    _opt(
        "--flash-attn",
        ("fa", "flash-attn"),
        "LLAMA_ARG_FLASH_ATTN",
        "choice",
        choices = frozenset({"on", "off", "auto"}),
    ),
    _opt(
        "--cache-type-k",
        ("ctk", "cache-type-k"),
        "LLAMA_ARG_CACHE_TYPE_K",
        "choice",
        choices = _CACHE_TYPES,
    ),
    _opt(
        "--cache-type-v",
        ("ctv", "cache-type-v"),
        "LLAMA_ARG_CACHE_TYPE_V",
        "choice",
        choices = _CACHE_TYPES,
    ),
    _opt(
        "--cache-type-k-draft",
        ("ctkd", "cache-type-k-draft", "spec-draft-type-k"),
        "LLAMA_ARG_SPEC_DRAFT_CACHE_TYPE_K",
        "choice",
        choices = _CACHE_TYPES,
    ),
    _opt(
        "--cache-type-v-draft",
        ("ctvd", "cache-type-v-draft", "spec-draft-type-v"),
        "LLAMA_ARG_SPEC_DRAFT_CACHE_TYPE_V",
        "choice",
        choices = _CACHE_TYPES,
    ),
    _opt(
        "--kv-unified",
        ("kvu", "kv-unified"),
        "LLAMA_ARG_KV_UNIFIED",
        "switch",
        negative = "--no-kv-unified",
        negative_names = ("no-kvu", "no-kv-unified"),
    ),
    _opt(
        "--kv-offload",
        ("kvo", "kv-offload"),
        "LLAMA_ARG_KV_OFFLOAD",
        "switch",
        negative = "--no-kv-offload",
        negative_names = ("nkvo", "no-kv-offload"),
    ),
    _opt("--cache-reuse", ("cache-reuse",), "LLAMA_ARG_CACHE_REUSE", "int", lo = 0, hi = 2**31 - 1),
    _opt("--keep", ("keep",), None, "int", lo = -1, hi = 2**31 - 1),
    _opt("--swa-full", ("swa-full",), "LLAMA_ARG_SWA_FULL", "switch"),
    _opt(
        "--warmup",
        ("warmup",),
        None,
        "switch",
        negative = "--no-warmup",
        negative_names = ("no-warmup",),
    ),
    _opt(
        "--context-shift",
        ("context-shift",),
        "LLAMA_ARG_CONTEXT_SHIFT",
        "switch",
        negative = "--no-context-shift",
        negative_names = ("no-context-shift",),
    ),
    _opt(
        "--mmproj-offload",
        ("mmproj-offload",),
        "LLAMA_ARG_MMPROJ_OFFLOAD",
        "switch",
        negative = "--no-mmproj-offload",
        negative_names = ("no-mmproj-offload",),
    ),
    _opt(
        "--image-min-tokens",
        ("image-min-tokens",),
        "LLAMA_ARG_IMAGE_MIN_TOKENS",
        "int",
        lo = 1,
        hi = 1 << 20,
    ),
    _opt(
        "--image-max-tokens",
        ("image-max-tokens",),
        "LLAMA_ARG_IMAGE_MAX_TOKENS",
        "int",
        lo = 1,
        hi = 1 << 20,
    ),
    _opt(
        "--jinja",
        ("jinja",),
        "LLAMA_ARG_JINJA",
        "switch",
        negative = "--no-jinja",
        negative_names = ("no-jinja",),
    ),
    _opt(
        "--reasoning",
        ("rea", "reasoning"),
        "LLAMA_ARG_REASONING",
        "choice",
        choices = frozenset({"on", "off", "auto"}),
    ),
    _opt("--reasoning-effort", ("reasoning-effort",), "LLAMA_ARG_REASONING_EFFORT", "word"),
    _opt(
        "--reasoning-preserve",
        ("reasoning-preserve",),
        "LLAMA_ARG_REASONING_PRESERVE",
        "switch",
        negative = "--no-reasoning-preserve",
        negative_names = ("no-reasoning-preserve",),
    ),
    _opt(
        "--chat-template-kwargs",
        ("chat-template-kwargs",),
        "LLAMA_ARG_CHAT_TEMPLATE_KWARGS",
        "json_object",
    ),
    _opt("--temp", ("temp", "temperature"), "LLAMA_ARG_TEMPERATURE", "float", lo = 0, hi = 100),
    _opt("--top-k", ("top-k",), "LLAMA_ARG_TOP_K", "int", lo = 0, hi = 1 << 20),
    _opt("--top-p", ("top-p",), "LLAMA_ARG_TOP_P", "float", lo = 0, hi = 1),
    _opt("--min-p", ("min-p",), "LLAMA_ARG_MIN_P", "float", lo = 0, hi = 1),
    _opt("--typical", ("typical", "typical-p"), None, "float", lo = 0, hi = 1),
    _opt("--top-n-sigma", ("top-n-sigma", "top-nsigma"), None, "float", lo = -1, hi = 100),
    _opt("--repeat-last-n", ("repeat-last-n",), None, "int", lo = -1, hi = 2**31 - 1),
    _opt("--repeat-penalty", ("repeat-penalty",), "LLAMA_ARG_REPEAT_PENALTY", "float", lo = 0, hi = 10),
    _opt(
        "--presence-penalty",
        ("presence-penalty",),
        "LLAMA_ARG_PRESENCE_PENALTY",
        "float",
        lo = -2,
        hi = 2,
    ),
    _opt(
        "--frequency-penalty",
        ("frequency-penalty",),
        "LLAMA_ARG_FREQUENCY_PENALTY",
        "float",
        lo = -2,
        hi = 2,
    ),
    _opt("--dry-multiplier", ("dry-multiplier",), None, "float", lo = 0, hi = 100),
    _opt("--dry-base", ("dry-base",), None, "float", lo = 0, hi = 100),
    _opt("--dry-allowed-length", ("dry-allowed-length",), None, "int", lo = 0, hi = 1 << 20),
    _opt("--dry-penalty-last-n", ("dry-penalty-last-n",), None, "int", lo = -1, hi = 2**31 - 1),
    _opt("--xtc-probability", ("xtc-probability",), None, "float", lo = 0, hi = 1),
    _opt("--xtc-threshold", ("xtc-threshold",), None, "float", lo = 0, hi = 1),
    _opt("--mirostat", ("mirostat",), None, "int", lo = 0, hi = 2),
    _opt("--mirostat-lr", ("mirostat-lr",), None, "float", lo = 0, hi = 100),
    _opt("--mirostat-ent", ("mirostat-ent",), None, "float", lo = 0, hi = 100),
    _opt("--seed", ("s", "seed"), None, "int", lo = -1, hi = 2**32 - 1),
    _opt("--spec-type", ("spec-type",), "LLAMA_ARG_SPEC_TYPE", "spec_types"),
    _opt("--spec-default", ("spec-default",), None, "switch"),
    _opt(
        "--spec-draft-n-max",
        ("spec-draft-n-max",),
        "LLAMA_ARG_SPEC_DRAFT_N_MAX",
        "int",
        lo = 0,
        hi = 1024,
    ),
    _opt(
        "--spec-draft-n-min",
        ("spec-draft-n-min",),
        "LLAMA_ARG_SPEC_DRAFT_N_MIN",
        "int",
        lo = 0,
        hi = 1024,
    ),
    _opt(
        "--spec-draft-p-min",
        ("spec-draft-p-min", "draft-p-min"),
        "LLAMA_ARG_SPEC_DRAFT_P_MIN",
        "float",
        lo = 0,
        hi = 1,
    ),
    _opt("--spec-ngram-mod-n-min", ("spec-ngram-mod-n-min",), None, "int", lo = 0, hi = 1024),
    _opt("--spec-ngram-mod-n-max", ("spec-ngram-mod-n-max",), None, "int", lo = 0, hi = 1024),
    _opt("--spec-ngram-mod-n-match", ("spec-ngram-mod-n-match",), None, "int", lo = 0, hi = 1024),
)


def _build_lookup() -> dict[str, tuple[_Option, bool]]:
    table: dict[str, tuple[_Option, bool]] = {}
    for opt in _OPTIONS:
        for name in opt.names:
            table[name] = (opt, False)
        for name in opt.negative_names:
            table[name] = (opt, True)
        if opt.env:
            table[opt.env.lower()] = (opt, False)
    return table


_LOOKUP = _build_lookup()


@dataclass
class ModelIni:
    args: list[str] = field(default_factory = list)
    n_parallel: Optional[int] = None
    sections: list[str] = field(default_factory = list)
    applied_sections: list[str] = field(default_factory = list)
    ignored: list[dict] = field(default_factory = list)


def _key_name(raw: str) -> str:
    key = raw.strip()
    if key.upper().startswith("LLAMA_ARG_"):
        return key.lower()
    return key.lstrip("-").lower()


def _parse_sections(text: str) -> list[tuple[str, list[tuple[str, Optional[str]]]]]:
    """``[(section, [(key, value), ...]), ...]`` in file order; keys above the first header
    go in ``default``. Mirrors common/preset.cpp parse_ini_from_file: ``;``/``#`` ends a value
    anywhere, quotes are kept, and a repeated header starts that section over."""
    order: list[str] = ["default"]
    entries: dict[str, list[tuple[str, Optional[str]]]] = {"default": []}
    current = "default"
    for raw in text.splitlines():
        line = re.split(r"[;#]", raw, maxsplit = 1)[0].strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            current = line[1:-1].strip()
            if current not in entries:
                order.append(current)
            entries[current] = []
            continue
        key, eq, value = line.partition("=")
        entries[current].append((key.strip(), value.strip() if eq else None))
    return [(name, entries[name]) for name in order if entries[name] or name != "default"]


def _gguf_stem(filename: Optional[str]) -> Optional[str]:
    if not filename:
        return None
    stem = Path(filename).name
    if stem.lower().endswith(".gguf"):
        stem = stem[:-5]
    return re.sub(r"-\d{5}-of-\d{5}$", "", stem)


def _number(value: str, integer: bool) -> float:
    if integer:
        if not re.fullmatch(r"[+-]?\d+", value):
            raise ValueError("expected an integer")
        return int(value)
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("expected a finite number")
    return number


def _compile_value(opt: _Option, value: str, inverted: bool) -> list[str]:
    """Tokens for one allowlisted key, or ValueError with a short reason."""
    v = value.strip()
    if opt.kind == "switch":
        low = v.lower()
        if low in _TRUE:
            on = True
        elif low in _FALSE:
            on = False
        else:
            raise ValueError("expected true or false")
        if inverted:
            on = not on
        if on:
            return [opt.flag]
        return [opt.negative] if opt.negative else []
    if not v:
        raise ValueError("missing value")
    if opt.kind in ("int", "float"):
        number = _number(v, opt.kind == "int")
        if not opt.lo <= number <= opt.hi:
            raise ValueError(f"must be between {opt.lo:g} and {opt.hi:g}")
        return [opt.flag, str(number) if opt.kind == "int" else v]
    if opt.kind == "choice":
        if v.lower() not in opt.choices:
            raise ValueError("expected one of " + ", ".join(sorted(opt.choices)))
        return [opt.flag, v.lower()]
    if opt.kind == "layers":
        if v.lower() in ("auto", "all"):
            return [opt.flag, v.lower()]
        number = _number(v, True)
        if number < -1:
            raise ValueError("expected a layer count, auto or all")
        return [opt.flag, str(number)]
    if opt.kind == "int_list":
        parts = [p.strip() for p in v.split(",")]
        for part in parts:
            if _number(part, True) < 0:
                raise ValueError("expected non-negative integers")
        return [opt.flag, ",".join(parts)]
    if opt.kind == "float_list":
        parts = [p.strip() for p in v.split(",")]
        for part in parts:
            if _number(part, False) < 0:
                raise ValueError("expected non-negative numbers")
        return [opt.flag, ",".join(parts)]
    if opt.kind == "pattern":
        if len(v) > 1024 or re.search(r"\s", v) or "=" not in v:
            raise ValueError("expected <tensor pattern>=<buffer type>")
        return [opt.flag, v]
    if opt.kind == "word":
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,32}", v):
            raise ValueError("expected a single word")
        return [opt.flag, v]
    if opt.kind == "spec_types":
        names = [p.strip().lower() for p in v.split(",")]
        unknown = [n for n in names if n not in _SPEC_TYPES]
        if unknown:
            raise ValueError("unknown speculative type " + ", ".join(unknown))
        return [opt.flag, ",".join(names)]
    if opt.kind == "json_object":
        try:
            parsed = json.loads(v)
        except ValueError:
            raise ValueError("expected a JSON object") from None
        if not isinstance(parsed, dict):
            raise ValueError("expected a JSON object")
        return [opt.flag, json.dumps(parsed, separators = (",", ":"))]
    raise ValueError("unsupported option")


def parse_model_ini(
    text: str,
    *,
    quant: Optional[str] = None,
    gguf_filename: Optional[str] = None,
) -> ModelIni:
    """Compile an ``unsloth.ini`` for one GGUF variant.

    Applied in order, later wins: keys above the first header and ``[*]`` (every variant), then
    the section named after the quant (``[UD-Q4_K_XL]``) or the GGUF file stem, case-insensitive.
    Raises ValueError only for a file too large to be an INI; bad keys land in ``ignored``."""
    if len(text.encode("utf-8", "surrogatepass")) > MAX_MODEL_INI_BYTES:
        raise ValueError(f"{MODEL_INI_FILENAME} is larger than {MAX_MODEL_INI_BYTES // 1024} KiB")
    parsed = _parse_sections(text)
    result = ModelIni(sections = [name for name, _ in parsed])
    wanted = {n.lower() for n in (quant, _gguf_stem(gguf_filename)) if n}
    chosen = [name for name, _ in parsed if name in ("default", "*")]
    chosen += [name for name, _ in parsed if name.lower() in wanted and name not in chosen]
    result.applied_sections = chosen
    by_name = dict(parsed)

    # One value per option: a later section replaces an earlier value (and moves it last).
    values: dict[str, list[str]] = {}
    for section in chosen:
        for raw_key, value in by_name[section]:
            key = _key_name(raw_key)
            if key in _ROUTER_ONLY:
                continue
            if value is None:
                result.ignored.append(
                    {"key": raw_key, "section": section, "reason": "expected key = value"}
                )
                continue
            if key in _STUDIO_SUPPLIED:
                result.ignored.append(
                    {"key": raw_key, "section": section, "reason": "Studio supplies the model"}
                )
                continue
            entry = _LOOKUP.get(key)
            if entry is None:
                result.ignored.append(
                    {"key": raw_key, "section": section, "reason": "not an allowed setting"}
                )
                continue
            opt, inverted = entry
            if opt.kind == "parallel":
                try:
                    number = int(_number(value, True))
                except ValueError:
                    number = 0
                if not 1 <= number <= 64:
                    result.ignored.append(
                        {"key": raw_key, "section": section, "reason": "must be between 1 and 64"}
                    )
                    continue
                result.n_parallel = number
                continue
            try:
                tokens = _compile_value(opt, value, inverted)
            except ValueError as exc:
                result.ignored.append({"key": raw_key, "section": section, "reason": str(exc)})
                continue
            values.pop(opt.flag, None)
            values[opt.flag] = tokens
    result.args = [token for tokens in values.values() for token in tokens]
    return result


@dataclass(frozen = True)
class LocatedModelIni:
    text: str
    location: str  # variant_folder | repo_root | local_dir
    quant: Optional[str]
    gguf_filename: Optional[str]


class NotGgufModel(Exception):
    """The identifier names no GGUF model, so there is nothing to attach an INI to."""


def _read_capped(path: Path) -> str:
    if path.stat().st_size > MAX_MODEL_INI_BYTES:
        raise ValueError(f"{MODEL_INI_FILENAME} is larger than {MAX_MODEL_INI_BYTES // 1024} KiB")
    return path.read_text(encoding = "utf-8-sig")


def _locate_local(model_path: str, gguf_variant: Optional[str]) -> Optional[LocatedModelIni]:
    from utils.models.model_config import (
        _find_local_gguf_by_variant,
        _gguf_variant_token,
        _is_gguf_filename,
    )

    path = Path(model_path).expanduser()
    gguf_file: Optional[Path] = None
    if path.is_file():
        if not _is_gguf_filename(path.name):
            raise NotGgufModel(model_path)
        gguf_file, root = path, path.parent
    elif path.is_dir():
        root = path
        if gguf_variant:
            found = _find_local_gguf_by_variant(str(path), gguf_variant)
            gguf_file = Path(found) if found else None
        if gguf_file is None:
            ggufs = sorted(
                p
                for pattern in ("*.gguf", "*/*.gguf")
                for p in path.glob(pattern)
                if "mmproj" not in p.name.lower()
            )
            if not ggufs:
                raise NotGgufModel(model_path)
            gguf_file = ggufs[0] if not gguf_variant else None
    else:
        raise NotGgufModel(model_path)
    folders = [gguf_file.parent] if gguf_file is not None else []
    folders.append(root)
    for folder in dict.fromkeys(folders):
        candidate = folder / MODEL_INI_FILENAME
        if candidate.is_file():
            # A local load rarely names its variant; the quant in the file name picks [Q4_K_M].
            quant = gguf_variant or (
                _gguf_variant_token(gguf_file.name) if gguf_file is not None else None
            )
            return LocatedModelIni(
                _read_capped(candidate),
                "local_dir",
                quant,
                gguf_file.name if gguf_file is not None else None,
            )
    return None


def _locate_hf(
    repo_id: str,
    gguf_variant: Optional[str],
    hf_token,
    offline: bool,
    list_variants: Optional[Callable] = None,
    download: Optional[Callable] = None,
) -> Optional[LocatedModelIni]:
    if list_variants is None:
        from utils.models.model_config import list_gguf_variants as list_variants
    try:
        variants, _ = list_variants(repo_id, hf_token = hf_token)
    except Exception:
        variants = None
    if variants is not None and not variants:
        raise NotGgufModel(repo_id)
    chosen = None
    if variants and gguf_variant:
        want = gguf_variant.lower()
        chosen = next((v for v in variants if (v.quant or "").lower() == want), None) or next(
            (v for v in variants if (_gguf_stem(v.filename) or "").lower().endswith(want)), None
        )
    folder = os.path.dirname(chosen.filename).strip("/") if chosen is not None else ""
    candidates = ([(f"{folder}/{MODEL_INI_FILENAME}", "variant_folder")] if folder else []) + [
        (MODEL_INI_FILENAME, "repo_root")
    ]
    fetch = download or _hf_download
    for filename, location in candidates:
        local = fetch(repo_id, filename, hf_token, offline)
        if local is None:
            continue
        return LocatedModelIni(
            _read_capped(Path(local)),
            location,
            chosen.quant if chosen is not None else gguf_variant,
            chosen.filename if chosen is not None else None,
        )
    return None


def _hf_download(repo_id: str, filename: str, hf_token, offline: bool) -> Optional[str]:
    """Local path of one repo file, or None when the repo has no such file."""
    from huggingface_hub import hf_hub_download, try_to_load_from_cache

    from hub.utils.hf_tokens import cached_read_refused, call_with_anonymous_retry
    from utils.hf_cache_settings import active_hf_hub_cache
    from utils.models.model_config import _env_offline

    cache_dir = active_hf_hub_cache()
    offline = offline or _env_offline()
    # A private repo's cached copy must not answer a caller that cannot read the repo.
    if cached_read_refused(
        hf_token,
        repo_id = repo_id,
        is_cached = lambda: isinstance(
            try_to_load_from_cache(repo_id, filename, cache_dir = cache_dir), str
        ),
        offline = offline,
    ):
        return None
    try:
        return call_with_anonymous_retry(
            lambda token: hf_hub_download(
                repo_id = repo_id,
                filename = filename,
                token = token,
                cache_dir = cache_dir,
                local_files_only = offline,
            ),
            hf_token,
        )
    except Exception as exc:
        # EntryNotFound (no such file), LocalEntryNotFound (offline, not cached) and an
        # unreachable Hub all mean "no INI here"; the load goes on without it.
        if type(exc).__name__ in (
            "EntryNotFoundError",
            "RemoteEntryNotFoundError",
            "LocalEntryNotFoundError",
        ):
            return None
        raise


def locate_model_ini(
    model_path: str,
    gguf_variant: Optional[str] = None,
    *,
    hf_token = None,
    offline: bool = False,
) -> Optional[LocatedModelIni]:
    """The ``unsloth.ini`` for a GGUF model: beside the selected variant, then at the model
    root. None when the GGUF model has none; NotGgufModel when it is not a GGUF model."""
    from utils.paths.path_utils import is_local_path

    if is_local_path(model_path):
        return _locate_local(model_path, gguf_variant)
    return _locate_hf(model_path, gguf_variant, hf_token, offline)


def describe(located: Optional[LocatedModelIni], compiled: Optional[ModelIni]) -> dict:
    if located is None or compiled is None:
        return {
            "found": False,
            "filename": MODEL_INI_FILENAME,
            "location": None,
            "sections": [],
            "applied_sections": [],
            "args": [],
            "n_parallel": None,
            "ignored": [],
        }
    return {
        "found": True,
        "filename": MODEL_INI_FILENAME,
        "location": located.location,
        "sections": compiled.sections,
        "applied_sections": compiled.applied_sections,
        "args": compiled.args,
        "n_parallel": compiled.n_parallel,
        "ignored": compiled.ignored,
    }
