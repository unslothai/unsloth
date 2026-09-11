# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Compile one native INI preset into an immutable Studio launch configuration.

The grammar follows common/preset.cpp: comments terminate even quoted values,
strings are not unquoted, duplicate keys replace, and repeated sections reset.
The caller must obtain a complete successful help probe from the selected binary.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import ntpath
import os
import re
import sys

from . import llama_server_args as policy

MAX_INI_BYTES = 64 * 1024
MAX_SECTIONS = 128
MAX_ENTRIES = 2048
NORMALIZATION_VERSION = 1
_TRUE = frozenset({"on", "enabled", "true", "1"})
_FALSE = frozenset({"off", "disabled", "false", "0"})
_NAME = r"--?[A-Za-z][A-Za-z0-9_-]*"
_DECL = re.compile(rf"^({_NAME}(?:,\s*{_NAME})*)(.*)$")


class CustomConfigError(ValueError):
    """A configuration error safe to expose without logging its source values."""


@dataclass(frozen = True)
class CustomConfigSource:
    version: int = 1
    mode: str = "managed"
    ini: str | None = None
    section: str | None = None

    def to_wire(self) -> dict:
        result = {"version": self.version, "mode": self.mode}
        if self.mode == "custom":
            result.update(ini = self.ini, section = self.section)
        return result


def parse_config_source(value: Mapping | CustomConfigSource | None) -> CustomConfigSource | None:
    if value is None:
        return None
    if isinstance(value, CustomConfigSource):
        # Validate constructed instances too; the dataclass is not a trust boundary.
        value = {
            "version": value.version,
            "mode": value.mode,
            "ini": value.ini,
            "section": value.section,
        }
        if value["mode"] == "managed" and value["ini"] is None and value["section"] is None:
            value = {"version": value["version"], "mode": value["mode"]}
    if not isinstance(value, Mapping):
        raise CustomConfigError("llama_cpp_config must be an object")
    if type(value.get("version")) is not int or value["version"] != 1:
        raise CustomConfigError("llama_cpp_config version must be 1")
    mode = value.get("mode")
    if not isinstance(mode, str) or mode not in {"managed", "custom"}:
        raise CustomConfigError("llama_cpp_config mode must be managed or custom")
    allowed = {"version", "mode"} if mode == "managed" else {"version", "mode", "ini", "section"}
    if set(value) - allowed:
        raise CustomConfigError("llama_cpp_config contains unsupported fields")
    if mode == "managed":
        return CustomConfigSource()
    ini, section = value.get("ini"), value.get("section")
    if not isinstance(ini, str) or not ini.strip():
        raise CustomConfigError("Custom configuration requires INI text")
    try:
        size = len(ini.encode("utf-8"))
    except UnicodeEncodeError:
        raise CustomConfigError("INI text contains invalid Unicode") from None
    if size > MAX_INI_BYTES:
        raise CustomConfigError("INI text exceeds the 64 KiB limit")
    if any((ord(c) < 32 and c not in "\r\n\t") or ord(c) == 127 for c in ini):
        raise CustomConfigError("INI text contains control characters")
    if section is not None and (
        not isinstance(section, str)
        or not section
        or len(section) > 512
        or any(c in section for c in "\r\n[]")
    ):
        raise CustomConfigError("Select a valid INI section name")
    return CustomConfigSource(1, "custom", ini, section)


def parse_option_catalog(help_text: str) -> tuple[dict, ...]:
    """Read native help declaration groups, never flags mentioned in prose.

    Unsupported/removed declarations remain unavailable. Unknown arity is marked
    -1 so a named option cannot accidentally become a valueless switch.
    """
    groups: list[tuple[str, list[str]]] = []
    for line in help_text.splitlines():
        if _DECL.match(line):
            groups.append((line, []))
        elif groups and line[:1].isspace():
            groups[-1][1].append(line.strip())
    result = []
    for line, continuation in groups:
        match = _DECL.match(line)
        names = re.findall(_NAME, match[1])
        rest = match[2]
        if rest.startswith("  ") or not rest.strip():
            hint, description = "", rest.strip()
        else:
            parts = re.split(r"\s{2,}", rest.strip(), maxsplit = 1)
            hint = parts[0]
            description = parts[1] if len(parts) > 1 else ""
        description = " ".join([description, *continuation]).strip()
        if re.search(r"\b(?:removed|no longer supported|no longer available)\b", description, re.I):
            continue
        hints = re.findall(r"\[[^]]*\]|\{[^}]*\}|<[^>]*>|\S+", hint)
        arity = len(hints) if len(hints) <= 2 else -1
        # Native emits every positive alias before its negative aliases.
        negatives = []
        for index, name in enumerate(names):
            if name.startswith("--no-") and "--" + name[5:] in names:
                start = index
                if index > 0 and not names[index - 1].startswith("--"):
                    start -= 1
                negatives = names[start:]
                break
        env = re.findall(r"\(env:\s*([A-Za-z_][A-Za-z0-9_]*)\)", description)
        if negatives:
            env += [
                e.replace("LLAMA_ARG_", "LLAMA_ARG_NO_", 1)
                for e in list(env)
                if e.startswith("LLAMA_ARG_")
            ]
        default = re.search(r"\bdefault:\s*([^)]*)", description)
        default_value = default[1].strip() if default else None
        if default_value:
            default_value = default_value.split(",", 1)[0].strip().strip("'\"")
        descriptor = {
            "names": names,
            "env": env,
            "arity": arity,
            "default": default_value,
            "negative_names": negatives,
        }
        # Bounded enum hints are stronger evidence than a universal enum registry.
        if len(hints) == 1 and hints[0][:1] in "[{<" and hints[0][-1:] in "]}>":
            choices = re.split(r"[|,]", hints[0][1:-1])
            if (
                len(choices) > 1
                and all(re.fullmatch(r"[\w+.-]+", c) for c in choices)
                and not any(
                    ".." in c or re.fullmatch(r"[A-Z][A-Z0-9_]*|dev[0-9]+", c) for c in choices
                )
            ):
                descriptor["choices"] = choices
        # These native convenience presets explicitly change model/port identity.
        if re.search(r"(?:download weights|model.*from the internet)", description, re.I):
            descriptor["resource_preset"] = True
        result.append(descriptor)
    return tuple(result)


@dataclass(frozen = True)
class _JsonValue:
    encoded: str


def _wire(value):
    return json.loads(value.encoded) if isinstance(value, _JsonValue) else value


@dataclass(frozen = True)
class CompiledCustomConfig:
    source: CustomConfigSource
    argv: tuple[str, ...]
    digest: str
    source_digest: str
    n_parallel: int
    tuning: tuple[tuple[str, object], ...]
    request_defaults: tuple[tuple[str, object], ...]
    diagnostics: tuple[str, ...]

    def summary(self) -> dict:
        return {
            "mode": "custom",
            "section": self.source.section,
            "digest": self.digest,
            "tuning": {k: _wire(v) for k, v in self.tuning},
            "request_defaults": {k: _wire(v) for k, v in self.request_defaults},
            "diagnostics": list(self.diagnostics),
        }


def _error(section: str, key: str, message: str) -> CustomConfigError:
    # Keys follow the ASCII grammar; bounded section labels cannot expose values.
    return CustomConfigError(f"INI section [{section[:80]}], key '{key[:80]}': {message}")


def _sections(ini: str) -> dict[str, dict[str, tuple[str, int]]]:
    sections: dict[str, dict[str, tuple[str, int]]] = {}
    section = "default"
    headers = entries = 0
    for number, raw in enumerate(re.split(r"\r\n|\r|\n", ini), 1):
        header = re.fullmatch(r"\[[ \t]*([^]]+)\][ \t]*(?:[;#].*)?", raw)
        if header:
            # The native section-name capture greedily includes trailing spaces.
            section = header[1]
            if len(section) > 512:
                raise CustomConfigError(f"INI section name is too long at line {number}")
            headers += 1
            if headers > MAX_SECTIONS:
                raise CustomConfigError("INI text exceeds the 128 section limit")
            sections[section] = {}
            continue
        line = re.split(r"[;#]", raw, maxsplit = 1)[0].rstrip(" \t")
        if not line.strip():
            continue
        entry = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_.-]*)[ \t]*=[ \t]*(.*)", line)
        if not entry:
            raise CustomConfigError(f"Invalid INI syntax at line {number}")
        entries += 1
        if entries > MAX_ENTRIES:
            raise CustomConfigError("INI text exceeds the 2048 entry limit")
        sections.setdefault(section, {})[entry[1]] = (entry[2], number)
    return sections


# Canonical names here carry Studio accounting/request semantics, not support.
# Availability and aliases always come from the selected executable's catalog.
_INT_FIELDS = {
    "ctx-size": ("n_ctx", 0),
    "batch-size": ("n_batch", 1),
    "ubatch-size": ("n_ubatch", 1),
    "parallel": ("n_parallel", 1),
    "n-parallel": ("n_parallel", 1),
    "threads": ("n_threads", -1),
    "n-cpu-moe": ("n_cpu_moe", 0),
    "gpu-layers": ("gpu_layers", -1),
    "n-gpu-layers": ("gpu_layers", -1),
    "top-k": ("top_k", -(2**31)),
    "seed": ("seed", -1),
    "repeat-last-n": ("repeat_last_n", -1),
    "predict": ("n_predict", -2),
    "n-predict": ("n_predict", -2),
}
_FLOAT_FIELDS = {
    "temp": "temperature",
    "temperature": "temperature",
    "top-p": "top_p",
    "min-p": "min_p",
    "repeat-penalty": "repeat_penalty",
    "presence-penalty": "presence_penalty",
    "frequency-penalty": "frequency_penalty",
    "typical": "typical_p",
    "typical-p": "typical_p",
}
_TUNING_FIELDS = {
    "cache-type-k": "cache_type_k",
    "cache-type-v": "cache_type_v",
    "load-mode": "load_mode",
    "spec-type": "spec_type",
    "split-mode": "split_mode",
}
_REQUEST = frozenset(
    {
        "top_k",
        "seed",
        "repeat_last_n",
        "n_predict",
        *_FLOAT_FIELDS.values(),
        "chat_template_kwargs",
        "reasoning_effort",
        "chat_template",
    }
)
_RESOURCE_NAMES = frozenset({"--model", "-m", "--mmproj", "-mm"})
_PARALLEL_NAMES = frozenset({"--parallel", "--n-parallel", "-np"})
# Alternate model selectors and opaque configuration would introduce a second source.
_OPAQUE = frozenset(
    {
        "--config",
        "--config-file",
        "--preset",
        "--preset-file",
        "--in-file",
        "--model-draft",
        "--spec-draft-model",
        "-md",
        "--model-url-draft",
        "--spec-draft-model-url",
        "--hf-repo-draft",
        "--spec-draft-hf-repo",
        "--hf-file-draft",
        "--spec-draft-hf-file",
    }
)


def _typed(name: str, value: str, descriptor: Mapping, section: str, key: str):
    if name in _INT_FIELDS:
        field, minimum = _INT_FIELDS[name]
        if name in {"gpu-layers", "n-gpu-layers"} and value in {"auto", "all"}:
            return field, value
        if not re.fullmatch(r"[+-]?[0-9]+", value):
            raise _error(section, key, "requires a native integer value")
        number = int(value)
        maximum = 2**32 - 1 if name == "seed" else 2**31 - 1
        if number < minimum or number > maximum:
            raise _error(section, key, "integer is outside the native domain")
        return field, number
    if name in _FLOAT_FIELDS:
        try:
            number = float(value)
        except ValueError:
            raise _error(section, key, "requires a finite numeric value") from None
        if not math.isfinite(number):
            raise _error(section, key, "requires a finite numeric value")
        return _FLOAT_FIELDS[name], number
    if name == "chat-template-kwargs":
        try:
            obj = json.loads(value, parse_constant = lambda _: (_ for _ in ()).throw(ValueError()))
            if not isinstance(obj, dict):
                raise ValueError()
            encoded = json.dumps(
                obj, sort_keys = True, separators = (",", ":"), ensure_ascii = True, allow_nan = False
            )
        except (ValueError, RecursionError):
            raise _error(section, key, "requires a valid JSON object") from None
        return "chat_template_kwargs", _JsonValue(encoded)
    if descriptor.get("choices") and value not in descriptor["choices"]:
        raise _error(section, key, "value is not supported by the selected executable")
    return _TUNING_FIELDS.get(name, name.replace("-", "_")), value


def _path_identity(path: str, platform: str) -> str:
    if platform == "win32":
        # realpath handles junctions when running on Windows; ntpath also supports
        # platform-specific comparison in host-independent tests.
        path = os.path.realpath(path) if os.name == "nt" else ntpath.abspath(path)
        return ntpath.normcase(ntpath.normpath(path))
    return os.path.realpath(path)


def compile_custom_config(
    source,
    catalog: Sequence[Mapping],
    *,
    model_path: str | None = None,
    mmproj_path: str | None = None,
    platform: str | None = None,
    validate_resources: bool = True,
) -> CompiledCustomConfig:
    source = parse_config_source(source)
    if source is None or source.mode != "custom":
        raise CustomConfigError("Compilation requires custom mode")
    platform = platform or sys.platform
    if not catalog:
        raise CustomConfigError(
            "Custom configuration requires a complete successful executable help probe"
        )
    lookup: dict[str, tuple[str, Mapping]] = {}
    descriptors: dict[str, Mapping] = {}
    for descriptor in catalog:
        names = descriptor.get("names", [])
        negative = descriptor.get("negative_names", [])
        positives = [n for n in names if n not in negative]
        if not positives or any(not re.fullmatch(_NAME, n) for n in names):
            raise CustomConfigError("Executable option catalog is malformed; repeat the help probe")
        canonical = positives[-1]
        if canonical in descriptors:
            raise CustomConfigError(
                "Executable option declarations are ambiguous; repeat the help probe"
            )
        descriptors[canonical] = descriptor
        for spelling in [*(n.lstrip("-") for n in names), *descriptor.get("env", [])]:
            if spelling in lookup and lookup[spelling][0] != canonical:
                raise CustomConfigError(
                    "Executable option aliases are ambiguous; repeat the help probe"
                )
            lookup[spelling] = (canonical, descriptor)
    if not any(_PARALLEL_NAMES.intersection(d["names"]) for d in catalog) or not any(
        "--model" in d["names"] or "-m" in d["names"] for d in catalog
    ):
        raise CustomConfigError(
            "Executable help probe is incomplete: model and parallel declarations are required"
        )
    sections = _sections(source.ini)
    named = set(sections) - {"*"}
    if source.section is None and named:
        raise CustomConfigError("Select an explicit named INI section")
    if source.section is not None and (source.section == "*" or source.section not in named):
        raise CustomConfigError("Selected INI section does not exist")
    selected = ["*"] + ([source.section] if source.section is not None else [])
    effective = {}
    diagnostics = []
    for section in selected:
        identities = set()
        for key, (value, line) in sections.get(section, {}).items():
            if key == "version":
                if value != "1":
                    raise _error(section, key, "only preset metadata version 1 is supported")
                continue
            if key in {"load-on-startup", "__PRESET_LOAD_ON_STARTUP"}:
                if value not in _TRUE | _FALSE:
                    raise _error(section, key, "requires a boolean value")
                diagnostic = "load-on-startup is router metadata; Studio's explicit Load action controls startup."
                if diagnostic not in diagnostics:
                    diagnostics.append(diagnostic)
                continue
            if key not in lookup:
                raise _error(section, key, "unknown or unavailable in the selected executable")
            canonical, descriptor = lookup[key]
            if canonical in identities:
                raise _error(
                    section, key, "multiple aliases declare the same option in this section"
                )
            identities.add(canonical)
            effective[canonical] = (descriptor, value, section, key)
    argv: list[str] = []
    tuning, defaults, normalized = {}, {}, {}
    resources = {
        "model": _path_identity(model_path, platform) if model_path else None,
        "mmproj": _path_identity(mmproj_path, platform) if mmproj_path else None,
    }
    n_parallel = None
    for canonical, (descriptor, value, section, key) in sorted(effective.items()):
        names = set(descriptor["names"])
        if names & _RESOURCE_NAMES:
            resource = "mmproj" if names & {"-mm", "--mmproj"} else "model"
            if validate_resources:
                if (
                    not resources[resource]
                    or _path_identity(value, platform) != resources[resource]
                ):
                    raise _error(
                        section, key, "must match the model/projector already selected in Studio"
                    )
                actual = mmproj_path if resource == "mmproj" else model_path
                if not os.path.isfile(actual):
                    raise _error(
                        section, key, "the selected Studio resource must be an existing file"
                    )
            else:
                diagnostics.append(
                    f"{resource} identity must be checked against the resolved Studio resource before launch."
                )
            continue
        if names & _PARALLEL_NAMES:
            _, number = _typed("parallel", value, descriptor, section, key)
            if not 1 <= number <= 64:
                raise _error(section, key, "parallel slots must be between 1 and 64")
            n_parallel = number
            continue
        if (
            any(policy.is_managed_flag(n) for n in names)
            or names & _OPAQUE
            or descriptor.get("resource_preset")
        ):
            raise _error(
                section,
                key,
                "this option is managed by Studio and cannot be set in custom configuration",
            )
        arity = descriptor.get("arity", -1)
        if arity not in {0, 1}:
            raise _error(
                section, key, "option arity is ambiguous or requires unsupported multiple values"
            )
        name = canonical.lstrip("-")
        if arity == 0:
            if value not in _TRUE | _FALSE:
                raise _error(
                    section,
                    key,
                    "requires a native boolean value (true/false, on/off, enabled/disabled, 1/0)",
                )
            enabled = value in _TRUE
            negative_names = descriptor.get("negative_names", [])
            if key in {n.lstrip("-") for n in negative_names}:
                enabled = not enabled
            # Native INI negative environment aliases are not negated by parse_bool_arg.
            if enabled:
                argv.append(canonical)
            elif negative_names:
                argv.append(negative_names[-1])
            normalized[name] = enabled
            tuning[name.replace("-", "_")] = enabled
        else:
            if not value:
                raise _error(section, key, "requires a nonempty value")
            field, typed = _typed(name, value, descriptor, section, key)
            normalized[name] = _wire(typed)
            (defaults if field in _REQUEST else tuning)[field] = typed
            emitted = (
                typed.encoded
                if isinstance(typed, _JsonValue)
                else str(typed)
                if isinstance(typed, (int, float))
                else value
            )
            argv.extend([canonical, emitted])
    if n_parallel is None:
        descriptor = next(d for d in catalog if _PARALLEL_NAMES.intersection(d["names"]))
        value = descriptor.get("default")
        if (
            not isinstance(value, str)
            or not re.fullmatch(r"[0-9]+", value)
            or not 1 <= int(value) <= 64
        ):
            raise CustomConfigError(
                "Native parallel default is automatic or unavailable; set np explicitly between 1 and 64"
            )
        n_parallel = int(value)
    tuning["n_parallel"] = n_parallel
    # Resolve concrete native accounting defaults without putting competing flags
    # on argv. Model-dependent defaults such as context 0 retain their meaning.
    for canonical, descriptor in descriptors.items():
        name = canonical.lstrip("-")
        field = _INT_FIELDS.get(name, (None,))[0] or _TUNING_FIELDS.get(name)
        default = descriptor.get("default")
        if field and field not in tuning and field not in defaults and default is not None:
            try:
                resolved_field, typed = _typed(name, default, descriptor, "*", name)
            except CustomConfigError:
                continue
            if resolved_field not in _REQUEST:
                tuning[resolved_field] = typed
    if "split_mode" in tuning:
        tuning["tensor_parallel"] = tuning["split_mode"] == "tensor"
    # The shared policy's shape/Windows limits apply after alias reconciliation.
    # Avoid its legacy GPU parser for the native 'auto'/'all' spellings.
    total = sum(len(token.encode("utf-8")) for token in argv)
    limit = (
        policy.MAX_EXTRA_ARGS_BYTES_WINDOWS if platform == "win32" else policy.MAX_EXTRA_ARGS_BYTES
    )
    if len(argv) > policy.MAX_EXTRA_ARG_TOKENS or total > limit:
        raise CustomConfigError(
            "Compiled custom arguments exceed the supported argument size limit"
        )
    if (
        platform == "win32"
        and policy.windows_command_length(argv)
        > policy.WINDOWS_COMMAND_LIMIT - policy.WINDOWS_COMMAND_RESERVE
    ):
        raise CustomConfigError(
            "Compiled custom arguments exceed the Windows command line limit after quoting"
        )
    identity = {
        "normalization_version": NORMALIZATION_VERSION,
        "section": source.section,
        "options": normalized,
        "resources": resources,
        "n_parallel": n_parallel,
        "tuning": {k: _wire(v) for k, v in tuning.items()},
        "request_defaults": {k: _wire(v) for k, v in defaults.items()},
    }
    digest = hashlib.sha256(
        json.dumps(
            identity, sort_keys = True, separators = (",", ":"), ensure_ascii = True, allow_nan = False
        ).encode()
    ).hexdigest()
    return CompiledCustomConfig(
        source,
        tuple(argv),
        digest,
        hashlib.sha256(source.ini.encode("utf-8")).hexdigest(),
        n_parallel,
        tuple(sorted(tuning.items())),
        tuple(sorted(defaults.items())),
        tuple(diagnostics),
    )
