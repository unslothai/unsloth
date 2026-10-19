# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Reads distribution metadata, not the module, since vllm can report a version yet raise on import."""

from __future__ import annotations

GOAL_PACKAGES = (
    "torch",
    "transformers",
    "trl",
    "peft",
    "accelerate",
    "bitsandbytes",
    "vllm",
    "triton",
    "xformers",
    "datasets",
    # Transitive packages the canary and frontier legs let pip move (e.g. tokenizers, safetensors).
    "tokenizers",
    "safetensors",
    "huggingface_hub",
    "unsloth",
    "unsloth_zoo",
)

# Only unsloth_zoo has a distribution name differing from its import name.
_DISTRIBUTION = {"unsloth_zoo": "unsloth-zoo"}


def distribution_version(module: str):
    """The installed distribution's version, or None if it is not installed."""
    import importlib.metadata as md

    for name in (_DISTRIBUTION.get(module, module), module):
        try:
            return md.version(name)
        except Exception:  # noqa: BLE001
            continue
    return None


def import_version(module: str):
    """Imports rather than reads metadata, since a package can install cleanly yet raise on import."""
    import importlib
    try:
        return getattr(importlib.import_module(module), "__version__", "unknown")
    except BaseException as exc:  # noqa: BLE001
        return f"IMPORT FAILED: {type(exc).__name__}: {str(exc)[:200]}"


def resolved_versions(packages = GOAL_PACKAGES, *, import_check = ()) -> dict:
    """Only packages named in import_check are imported, since importing vllm everywhere adds a minute."""
    out: dict = {}
    for name in packages:
        installed = distribution_version(name)
        entry: dict = {"installed": installed}
        if name in import_check and installed is not None:
            entry["imported"] = import_version(name)
        out[name] = entry
    return out


def flatten_versions(resolved: dict) -> dict:
    """Installed version leads, since a bisect acts on it; an import failure is shown beside it."""
    flat = {}
    for name, entry in resolved.items():
        imported = entry.get("imported")
        if isinstance(imported, str) and imported.startswith("IMPORT FAILED"):
            flat[name] = f"{entry.get('installed')} ({imported})"
        else:
            flat[name] = entry.get("installed")
    return flat


def load_pins(path) -> dict:
    """A ``package==version`` pin file, parsed. Blank and ``#`` lines ignored."""
    from pathlib import Path

    pins: dict = {}
    text = Path(path).read_text(encoding = "utf-8")
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        name, sep, version = line.partition("==")
        if not sep:
            raise ValueError(f"pin file line is not name==version: {line!r}")
        pins[name.strip().replace("-", "_")] = version.strip()
    return pins


def pin_failures(pins: dict, resolved: dict) -> list[str]:
    """A pin outside the probe list means not probed, which is not the same as not installed."""
    failures = []
    for name, wanted in sorted(pins.items()):
        if name not in resolved:
            failures.append(
                f"pinned {name}=={wanted} but no version of it was recorded, so "
                f"whether the pin held is unknown"
            )
            continue
        got = (resolved.get(name) or {}).get("installed")
        if got is None:
            failures.append(f"pinned {name}=={wanted} but it is not installed")
        elif got != wanted:
            failures.append(f"pinned {name}=={wanted} but {got} was resolved")
    return failures


def versions_for_pins(
    pins: dict,
    packages = GOAL_PACKAGES,
    *,
    import_check = (),
) -> dict:
    """Probes the pinned packages too, so a pin outside the goal list is not reported as not installed."""
    ordered = list(packages) + [name for name in pins if name not in packages]
    return resolved_versions(tuple(ordered), import_check = import_check)
