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

# Libraries whose version bumps this CI exists to detect.
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
    # The transitive runtime packages the canary and frontier resolutions are allowed to move, which are not optional
    # extras here: the frontier leg installs transformers and trl WITH their dependencies precisely so pip repairs them,
    # and legs.py records the resolution doing it
    # "Would install datasets-5.0.1 huggingface_hub-1.27.0 transformers-5.15.0 trl-1.9.2", and before that the two
    # errors that forced the change, "tokenizers<=0.23.0,>=0.22.0 is required, but found tokenizers==0.23.1" and
    # "safetensors>=0.8.0 is required, but found safetensors==0.7.0".
    "tokenizers",
    "safetensors",
    "huggingface_hub",
    "unsloth",
    "unsloth_zoo",
)

# name; getting it wrong records "not installed" for a package that is.
# Exactly one of the above has a distribution name differing from its import name;
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

    "Not probed" is a THIRD outcome and is kept apart from "not installed".
    ``resolved`` is whatever the caller asked ``resolved_versions`` about, so a
    pin outside that list has no entry at all, and folding it in with "it is not
    installed" is a failure invented about a package that may be installed and
    correct. Callers derive the probe list from the pin file (see
    ``versions_for_pins``) so this cannot normally happen; it is reported rather
    than assumed away because the invented failure is indistinguishable from a
    real one in a report.
    """
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
    """``resolved_versions`` over the goal list AND everything ``pins`` names.

    The probe list is derived from the pin file rather than assumed to cover
    it. Pinning a package the goal list does not carry used to make
    ``pin_failures`` report it as not installed, since the lookup answered from
    a table that was never asked about it: a control leg failing on a pin that
    held perfectly.
    """
    ordered = list(packages) + [name for name in pins if name not in packages]
    return resolved_versions(tuple(ordered), import_check = import_check)
