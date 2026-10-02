# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Compile a per-model llama.cpp preset INI into llama-server argv.

Grammar follows llama.cpp common/preset.cpp: keys before any header form the ``default``
section, only ``[*]`` applies to every section, ``;``/``#`` start comments and the last
duplicate key wins.
Option names and arity come from the selected binary's ``--help`` catalog.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import re

from . import llama_server_args as policy

MAX_INI_BYTES = 64 * 1024
_TRUE = frozenset({"on", "enabled", "true", "1"})
_FALSE = frozenset({"off", "disabled", "false", "0"})
# Studio selects the model and projector; preset files commonly carry both.
_IGNORED = frozenset({"-m", "--model", "-mm", "--mmproj"})
_PARALLEL = frozenset({"-np", "--parallel", "--n-parallel"})
# Router metadata in upstream preset files; llama-server itself has no such options.
_ROUTER_ONLY = frozenset({"load-on-startup", "stop-timeout", "dedup-cache-models", "version"})
# Flag -> (request field, Studio chat API bounds in models/inference.py).
_SAMPLING = {
    "--temp": ("temperature", 0.0, 2.0),
    "--top-p": ("top_p", 0.0, 1.0),
    "--top-k": ("top_k", -1, 100),
    "--min-p": ("min_p", 0.0, 1.0),
    "--repeat-penalty": ("repetition_penalty", 1.0, 2.0),
    "--presence-penalty": ("presence_penalty", 0.0, 2.0),
}


class CustomConfigError(ValueError):
    """A configuration error whose message never echoes INI values."""


@dataclass(frozen = True)
class CustomConfigSource:
    version: int = 1
    mode: str = "managed"
    ini: str | None = None
    section: str | None = None

    def to_wire(self) -> dict:
        if self.mode != "custom":
            return {"version": 1, "mode": "managed"}
        return {"version": 1, "mode": "custom", "ini": self.ini, "section": self.section}


def parse_config_source(value) -> CustomConfigSource | None:
    if value is None:
        return None
    if isinstance(value, CustomConfigSource):
        value = value.to_wire()
    if not isinstance(value, Mapping) or value.get("version") != 1:
        raise CustomConfigError("llama_cpp_config must be an object with version 1")
    mode = value.get("mode")
    if mode == "managed" and set(value) <= {"version", "mode"}:
        return CustomConfigSource()
    if mode != "custom" or set(value) - {"version", "mode", "ini", "section"}:
        raise CustomConfigError("llama_cpp_config mode must be managed or custom")
    ini, section = value.get("ini"), value.get("section")
    if not isinstance(ini, str) or not ini.strip():
        raise CustomConfigError("Custom configuration requires INI text")
    if len(ini.encode("utf-8", "surrogatepass")) > MAX_INI_BYTES:
        raise CustomConfigError("INI text exceeds the 64 KiB limit")
    if any((ord(c) < 32 and c not in "\r\n\t") or ord(c) == 127 for c in ini):
        raise CustomConfigError("INI text contains control characters")
    if section is not None and (
        not isinstance(section, str) or not section or len(section) > 512 or "]" in section
    ):
        raise CustomConfigError("Select a valid INI section name")
    return CustomConfigSource(1, "custom", ini, section)


@dataclass(frozen = True)
class CompiledCustomConfig:
    source: CustomConfigSource
    argv: tuple[str, ...]
    n_parallel: int | None
    options: dict = field(default_factory = dict)
    request_defaults: dict = field(default_factory = dict)
    diagnostics: tuple[str, ...] = ()
    # Every help-block spelling of a valued option, so lookups match whichever alias the INI used.
    aliases: dict = field(default_factory = dict)

    def option(self, *names: str):
        return next(
            (table[n] for table in (self.options, self.aliases) for n in names if n in table),
            None,
        )

    @property
    def cpu_only(self) -> bool:
        # -ngl 0 still offloads ops and the projector; --device none also moves the projector
        # (common/arg.cpp), but -mmdev and a drafter pick their own devices.
        return (
            self.option("-dev", "--device") == "none"
            and self.option("-mmdev", "--mmproj-device") in (None, "none")
            and (
                self.option("-md", "-hfd") is None
                or self.option("-devd", "--device-draft") == "none"
            )
        )

    def summary(self) -> dict:
        return {
            "mode": "custom",
            "section": self.source.section,
            "options": dict(self.options),
            "request_defaults": dict(self.request_defaults),
            "diagnostics": list(self.diagnostics),
        }


def _sections(ini: str) -> dict[str, dict[str, str]]:
    sections: dict[str, dict[str, str]] = {"default": {}}
    current = "default"
    for number, raw in enumerate(re.split(r"\r\n|\r|\n", ini), 1):
        header = re.fullmatch(r"\[[ \t]*([^\]]+)\][ \t]*(?:[;#].*)?", raw)
        if header:
            current = header[1].strip()
            sections[current] = {}
            continue
        line = re.split(r"[;#]", raw, maxsplit = 1)[0].strip()
        if not line:
            continue
        entry = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_.-]*)\s*=\s*(.*)", line)
        if not entry:
            raise CustomConfigError(f"Invalid INI syntax at line {number}")
        sections.setdefault(current, {})[entry[1]] = entry[2]
    return sections


def _polarity(block: str) -> tuple[list[str], list[str]]:
    """Positive and negative spellings from a help block's declaration line.

    llama.cpp prints args then args_neg, each run ending in its long form:
    ``-kvo, --kv-offload, -nkvo, --no-kv-offload``."""
    head = re.match(r"(-[\w.-]+(?:, -[\w.-]+)*)", block or "")
    names = head[1].split(", ") if head else []
    cut = next((i + 1 for i, n in enumerate(names) if n.startswith("--")), len(names))
    return names[:cut], names[cut:]


def _spelling(key: str, flags: Mapping) -> str | None:
    found = next((f for f in (f"--{key}", f"-{key}") if f in flags), None)
    if found is None and re.fullmatch(r"[A-Z][A-Z0-9_]*", key):
        # Presets may name an option by its env variable, which maps to the option itself.
        block = next((d for d in flags.values() if d and f"(env: {key})" in d), None)
        if block is not None:
            positive = [f for f in _polarity(block)[0] if f in flags]
            found = max(positive, key = len, default = None)
    return found


def compile_custom_config(source, flags: Mapping, switch_flags) -> CompiledCustomConfig:
    """``flags``/``switch_flags`` are the selected binary's probed help catalog."""
    source = parse_config_source(source)
    if source is None or source.mode != "custom":
        raise CustomConfigError("Compilation requires custom mode")
    if "--port" not in flags:
        raise CustomConfigError(
            "The selected llama-server did not report its options; check the install"
        )
    switches = set(switch_flags)
    sections = _sections(source.ini)
    named = [s for s in sections if s != "*" and (s != "default" or sections[s])]
    section = source.section
    if section is None and named:
        if named != ["default"]:
            raise CustomConfigError("Select which INI section to use")
        section = "default"
    if section is not None and section not in named:
        raise CustomConfigError("The selected INI section does not exist")
    # llama.cpp resolves keys to options before cascading: the section overrides [*] per option.
    entries: dict = {}
    for name in ("*", section):
        owners: dict = {}
        for key, value in sections.get(name or "", {}).items():
            if key in _ROUTER_ONLY:
                continue
            flag = _spelling(key, flags)
            if flag is None:
                raise CustomConfigError(
                    f"'{key[:80]}' is not an option of the selected llama-server"
                )
            # Two aliases in one section: llama.cpp keeps one arbitrarily, so refuse both.
            identity = flags[flag] or flag
            other = owners.setdefault(identity, key)
            if other != key:
                raise CustomConfigError(f"'{other[:80]}' and '{key[:80]}' set the same option")
            entries[identity] = (key, flag, value)

    argv: list[str] = []
    options: dict = {}
    aliases: dict = {}
    defaults: dict = {}
    diagnostics: list[str] = []
    n_parallel = None
    for identity, (key, flag, value) in entries.items():
        if flag in _IGNORED:
            note = "Studio selects the model and projector; m and mm entries are ignored."
            if note not in diagnostics:
                diagnostics.append(note)
            continue
        if flag in _PARALLEL:
            if (
                not re.fullmatch(r"[0-9]+", value)
                or not policy.PARALLEL_MIN <= int(value) <= policy.PARALLEL_MAX
            ):
                raise CustomConfigError(
                    f"'{key}' must be between {policy.PARALLEL_MIN} and {policy.PARALLEL_MAX}"
                )
            n_parallel = int(value)
            continue
        if policy.is_managed_flag(flag):
            raise CustomConfigError(f"'{key[:80]}' is managed by Studio and cannot be set here")
        if flag in switches:
            if value.lower() not in _TRUE | _FALSE:
                raise CustomConfigError(f"'{key[:80]}' needs true or false")
            # Like common/preset.cpp: a negative alias (-nkvo, --no-x) inverts the value, and
            # false on a switch with no negative form is dropped.
            positive, negative = _polarity(identity)
            if flag not in positive + negative:
                positive, negative = [flag], []
            on = (value.lower() in _TRUE) != (flag in negative)
            pick = [f for f in (positive if on else negative) if f in flags]
            if not pick:
                continue
            flag = flag if flag in pick else max(pick, key = len)
            argv.append(flag)
            options[flag] = True
            continue
        if not value:
            raise CustomConfigError(f"'{key[:80]}' needs a value")
        argv += [flag, value]
        options[flag] = value
        aliases.update((f, value) for f, d in flags.items() if d and d == identity)
        sampling = next((s for s in _SAMPLING if s in flags and flags[s] == flags[flag]), None)
        if sampling is not None:
            name, low, high = _SAMPLING[sampling]
            try:
                number = int(value) if sampling == "--top-k" else float(value)
            except ValueError:
                raise CustomConfigError(f"'{key[:80]}' needs a number") from None
            # It becomes the chat default, so it must pass the chat request schema.
            if not low <= number <= high:
                raise CustomConfigError(f"'{key[:80]}' must be between {low} and {high}")
            defaults[name] = number
    if "--no-jinja" in options:
        raise CustomConfigError("jinja = false would break Studio's tool calling")
    try:
        policy.validate_extra_args(argv)
    except ValueError as exc:
        raise CustomConfigError(str(exc)) from None
    return CompiledCustomConfig(
        source, tuple(argv), n_parallel, options, defaults, tuple(diagnostics), aliases
    )
