# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Speculative decoding and reasoning budget flags."""

import json

from typing import (
    Mapping,
    Optional,
)

from .log import logger


def _build_ngram_mod_flags(
    caps: Optional[dict],
    n_match: int = 24,
    n_min: int = 48,
    n_max: int = 64,
    *,
    chain_with_mtp: bool = False,
) -> list[str]:
    """Emit the right ngram-mod knob flags for the running llama-server.

    Post-rename builds expose ``--spec-ngram-mod-n-{match,min,max}``;
    pre-rename builds expose legacy ``--spec-ngram-size-n`` /
    ``--draft-min`` / ``--draft-max``. ``caps`` comes from
    ``probe_server_capabilities``; ``ngram_mod_flavor`` says which set is
    real (vs a removal-stub). Returns ``[]`` when neither is available so
    the caller can drop ngram-mod entirely.

    ``chain_with_mtp`` omits the legacy ``--draft-min``/``--draft-max`` pair,
    which a chained MTP invocation owns and would otherwise invert.
    """
    flavor = caps.get("ngram_mod_flavor") if caps else None
    if flavor == "new":
        return [
            "--spec-ngram-mod-n-match",
            str(n_match),
            "--spec-ngram-mod-n-min",
            str(n_min),
            "--spec-ngram-mod-n-max",
            str(n_max),
        ]
    if flavor == "legacy":
        # Pre-rename llama.cpp: same knobs lived under --spec-ngram-size-n
        # (lookup length) and generic --draft-min / --draft-max (N range).
        flags = ["--spec-ngram-size-n", str(n_match)]
        if not chain_with_mtp:
            flags.extend(["--draft-min", str(n_min), "--draft-max", str(n_max)])
        return flags
    return []


def _build_launch_reasoning_args(
    caps: Mapping[str, object], reasoning_kwargs: Mapping[str, object]
) -> list[str]:
    """The launch flags that carry ``reasoning_kwargs``, per key.

    llama-server's --chat-template-kwargs deprecation is PER KEY (#7526), and only
    ``enable_thinking`` has a flag of its own, so only that key moves; every other
    template variable stays in --chat-template-kwargs, still the only channel for one.

    Gated CLOSED on the probe: a build predating the flag exits with "error: invalid
    argument: --reasoning" rather than starting without it, so an unknown answer keeps
    main's argv byte for byte.

    The two spellings are equivalent, not identical. common/arg.cpp's --reasoning
    handler writes the same ``default_template_kwargs["enable_thinking"]`` entry AND
    sets ``params.enable_reasoning``, whose only consumer (tools/server/server-context.cpp)
    folds it into the per-server ``opt.enable_thinking`` default -- which
    tools/server/server-common.cpp then overrides from the merged kwarg anyway. If
    llama.cpp stops writing that kwarg, or stops letting it override, this becomes a
    behaviour change rather than a substitution.
    """
    remaining = dict(reasoning_kwargs)
    args: list[str] = []
    if caps.get("supports_reasoning_flag") and isinstance(remaining.get("enable_thinking"), bool):
        args.extend(["--reasoning", "on" if remaining.pop("enable_thinking") else "off"])
    # `not args`: a caller passing `{}` would otherwise lose main's `-ctk {}` on an old build.
    if remaining or not args:
        args.extend(["--chat-template-kwargs", json.dumps(remaining)])
    return args


def _build_reasoning_budget_flags(
    caps: Mapping[str, object], reasoning_budget: int, reasoning_budget_message: str
) -> list[str]:
    """Build only the reasoning flags advertised by this llama-server."""
    flags: list[str] = []
    if reasoning_budget != -1 and caps.get("supports_reasoning_budget"):
        flags.extend(["--reasoning-budget", str(reasoning_budget)])
    if reasoning_budget_message and caps.get("supports_reasoning_budget_message"):
        flags.extend(["--reasoning-budget-message", reasoning_budget_message])
    return flags


# Canonical Speculative Decoding modes exposed by the Unsloth chat UI.
# Dropdown renders seven (auto, mtp, dspark, dflash, ngram, mtp+ngram, off); the
# load API also accepts legacy values the original Switch and external callers
# emit (default, draft-mtp, ngram-mod, ngram-simple).
_CANONICAL_SPEC_MODES = {
    "auto",
    "mtp",
    "dspark",
    "eagle3",
    "dflash",
    "ngram",
    "mtp+ngram",
    "off",
    "ngram-simple",
}
_LEGACY_SPEC_MODE_MAP = {
    "default": "auto",
    "draft-mtp": "mtp",
    "draft-dspark": "dspark",
    "draft-dflash": "dflash",
    "draft-eagle3": "eagle3",
    "ngram-mod": "ngram",
    "none": "off",
    "disable": "off",
    "disabled": "off",
}


def _trained_drafter_sidecar(model_path):
    """(eagle3 drafter path, MTP head trained, draft steps) from the drafter.json a Studio export
    wrote beside the GGUF."""
    if not model_path:
        return None, False, None
    try:
        from core.export.drafter_export import eagle3_sidecar_path, gguf_drafter_sidecar

        info = gguf_drafter_sidecar(model_path) or {}
        try:
            steps = max(1, int(info["draft_steps"])) if info.get("draft_steps") else None
        except (TypeError, ValueError):
            steps = None
        return eagle3_sidecar_path(model_path), bool(info.get("mtp_trained")), steps
    except Exception as exc:
        logger.debug("Could not read a drafter manifest beside %s: %s", model_path, exc)
        return None, False, None


def _canonicalize_spec_mode(value):
    """Map any accepted ``speculative_type`` input onto a canonical mode.

    Returns ``auto``, ``mtp``, ``dspark``, ``dflash``, ``ngram``, ``mtp+ngram``,
    ``off``, ``ngram-simple``, or ``None`` (callers treat ``None`` as ``auto``).
    Unknown strings collapse to ``auto`` so a stale UI value or typo falls
    back to the safe platform-aware path.
    """
    if value is None:
        return None
    if not isinstance(value, str):
        return None
    stripped = value.strip().lower()
    if not stripped:
        return None
    if stripped in _CANONICAL_SPEC_MODES:
        return stripped
    if stripped in _LEGACY_SPEC_MODE_MAP:
        return _LEGACY_SPEC_MODE_MAP[stripped]
    # Old persisted state emits llama.cpp comma-chains e.g.
    # "ngram-mod,draft-mtp"; collapse the most common one explicitly.
    pieces = [p.strip() for p in stripped.split(",") if p.strip()]
    has_mtp = any(p in ("mtp", "draft-mtp") for p in pieces)
    has_ngram = any(p in ("ngram", "ngram-mod") for p in pieces)
    if has_mtp and has_ngram:
        return "mtp+ngram"
    if has_mtp:
        return "mtp"
    if has_ngram:
        return "ngram"
    return "auto"
