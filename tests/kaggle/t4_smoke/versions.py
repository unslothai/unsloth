# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""What was actually installed, recorded so a regression can be attributed.

The canary leg installs the LATEST release of every library Unsloth sits on,
and runs the pinned control leg's payload; canary red plus control green means
a version bump, but only if the report names which versions moved.

Two rules. **Read the metadata, not the module**: ``importlib.metadata.version``
reads the installed distribution, ``torch.__version__`` is whatever a package
chose to expose. They disagree where it matters -- ``vllm`` on a card its wheel
has no kernels for has a metadata version and raises on import -- so both are
recorded, "installed 0.11.2, would not import" being a different finding from
"not installed". **Never let recording a version fail the run**: every probe is
wrapped, since dying while collecting diagnostics reports nothing at all.
"""

from __future__ import annotations

# Libraries whose version bumps this CI exists to detect. Order is report
# order: runtime stack first, Unsloth packages last.
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
    "unsloth",
    "unsloth_zoo",
)

# Exactly one of the above has a distribution name differing from its import
# name; getting it wrong records "not installed" for a package that is.
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
    """``__version__`` after a real import, or an error string.

    Importing is the point: ``vllm`` on a compute capability its wheel was not
    built for installs cleanly and raises on import, and that is the finding.
    """
    import importlib
    try:
        return getattr(importlib.import_module(module), "__version__", "unknown")
    except BaseException as exc:  # noqa: BLE001
        return f"IMPORT FAILED: {type(exc).__name__}: {str(exc)[:200]}"


def resolved_versions(packages = GOAL_PACKAGES, *, import_check = ()) -> dict:
    """``{package: {"installed": ..., "imported": ...}}`` for the goal list.

    ``import_check`` names the subset worth paying an import for: importing
    everything would pull ``vllm`` into payloads with no use for it and add a
    minute for a number the metadata already gave, so callers opt in per
    package.
    """
    out: dict = {}
    for name in packages:
        installed = distribution_version(name)
        entry: dict = {"installed": installed}
        if name in import_check and installed is not None:
            entry["imported"] = import_version(name)
        out[name] = entry
    return out


def flatten_versions(resolved: dict) -> dict:
    """``{package: version-or-None}``, for a one-line summary.

    The installed version leads: it is what a bisect over releases acts on. An
    import failure is surfaced alongside it, so a package that is present and
    unusable does not read as present and fine.
    """
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
    """Pins that did not hold.

    A control leg whose pins a transitive dependency silently overrode is not a
    control, and every canary-vs-control conclusion drawn from it would be
    wrong. A pin naming a package that is not installed is the same defect from
    the other side, so it is reported too.
    """
    failures = []
    for name, wanted in sorted(pins.items()):
        entry = resolved.get(name)
        got = entry.get("installed") if entry else None
        if got is None:
            failures.append(f"pinned {name}=={wanted} but it is not installed")
        elif got != wanted:
            failures.append(f"pinned {name}=={wanted} but {got} was resolved")
    return failures
