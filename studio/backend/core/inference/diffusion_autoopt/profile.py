# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Static introspection of a denoiser class, the input the optimizer plans from.

A :class:`ModelProfile` answers, without running the model: which blocks repeat (declared ``_repeated_blocks``,
Studio's verified list, or discovered via ``_no_split_modules`` + a ModuleList holding >= 2), whether the attention
goes through diffusers' dispatcher, whether a repeated block merges the text and image streams (static compile on old
inductors), and a static graph-break screen of each repeated block's ``forward`` source (host syncs, data-dependent
shapes). Profiles are cached by (class qualname, forward source digest, diffusers version), so a diffusers upgrade or
a patched forward re-profiles.

The profile only DESCRIBES. Registered families keep the decisions they always had; the plan uses the profile to
explain a decline and, for a family with no data, to decline a step up front with the reason instead of failing.
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field
from typing import Any, Optional

from .patches import source_digest

# Host-sync / data-dependent constructs that break a dynamo graph (zoo compiler.py screens the same set).
GRAPH_BREAK_PATTERNS: tuple[tuple[str, str], ...] = (
    (".item()", r"\.item\(\s*\)"),
    (".tolist()", r"\.tolist\(\s*\)"),
    (".nonzero()", r"\.nonzero\(|torch\.nonzero\("),
    (".cpu()", r"\.cpu\(\s*\)"),
    (".numpy()", r"\.numpy\(\s*\)"),
    ("torch.unique", r"torch\.unique\(|\.unique\(\s*\)"),
    ("bincount", r"\.bincount\(|torch\.bincount\("),
    ("masked_select", r"masked_select\("),
    ("print", r"(?<![\w.])print\("),
)
_COMPILED_PATTERNS = tuple((label, re.compile(rx)) for label, rx in GRAPH_BREAK_PATTERNS)
_DISPATCH_RX = re.compile(r"dispatch_attention_fn\(")
_SDPA_RX = re.compile(r"scaled_dot_product_attention\(")
_STREAM_MERGE_RX = re.compile(
    r"torch\.cat\(\s*\[\s*(?:encoder_hidden_states\s*,\s*hidden_states"
    r"|hidden_states\s*,\s*encoder_hidden_states)\s*\]"
)


@dataclass(frozen = True)
class BlockProfile:
    name: str
    count: int
    digest: Optional[str]
    graph_break_suspects: tuple[str, ...] = ()
    merges_streams: bool = False


@dataclass(frozen = True)
class ModelProfile:
    class_name: str
    qualname: str
    digest: Optional[str]
    diffusers_version: str
    declared_blocks: tuple[str, ...] = ()
    verified_blocks: tuple[str, ...] = ()
    discovered_blocks: tuple[str, ...] = ()
    blocks: tuple[BlockProfile, ...] = field(default_factory = tuple)
    uses_dispatcher: Optional[bool] = None
    calls_sdpa_directly: bool = False
    has_compile_repeated_blocks: bool = False

    @property
    def repeated_blocks(self) -> tuple[str, ...]:
        """What the regional compile will use: declared, else verified-and-present (Studio's ensure_repeated_blocks)."""
        if self.declared_blocks:
            return self.declared_blocks
        return tuple(n for n in self.verified_blocks if n in self.discovered_blocks)

    @property
    def graph_break_suspects(self) -> dict[str, tuple[str, ...]]:
        return {b.name: b.graph_break_suspects for b in self.blocks if b.graph_break_suspects}

    def compile_decline(self) -> Optional[str]:
        """Why a regional compile cannot engage on this class, or None."""
        if not self.has_compile_repeated_blocks:
            return "denoiser has no compile_repeated_blocks (not a diffusers ModelMixin)"
        if self.repeated_blocks:
            return None
        if self.discovered_blocks:
            return (
                "declares no _repeated_blocks; discovered "
                + ", ".join(self.discovered_blocks)
                + " but they are not verified for regional compile"
            )
        return "declares no _repeated_blocks and none were discovered"

    def as_dict(self) -> dict:
        return {
            "class": self.class_name,
            "digest": self.digest,
            "diffusers": self.diffusers_version,
            "repeated_blocks": list(self.repeated_blocks),
            "declared_blocks": list(self.declared_blocks),
            "discovered_blocks": list(self.discovered_blocks),
            "uses_dispatcher": self.uses_dispatcher,
            "graph_break_suspects": {k: list(v) for k, v in self.graph_break_suspects.items()},
            "merges_streams": [b.name for b in self.blocks if b.merges_streams],
        }


_CACHE: dict[tuple, ModelProfile] = {}
_LOCK = threading.Lock()


def _diffusers_version() -> str:
    try:
        import diffusers  # noqa: PLC0415
        return str(getattr(diffusers, "__version__", "unknown"))
    except Exception:  # noqa: BLE001
        return "unavailable"


def _source(fn: Any) -> Optional[str]:
    try:
        import inspect  # noqa: PLC0415
        return inspect.getsource(fn)
    except Exception:  # noqa: BLE001
        return None


def screen_source(source: Optional[str]) -> tuple[str, ...]:
    """Graph-break suspects in ``source`` (labels of :data:`GRAPH_BREAK_PATTERNS`), comment lines ignored."""
    if not source:
        return ()
    code = "\n".join(line.split("#", 1)[0] for line in source.splitlines())
    return tuple(label for label, rx in _COMPILED_PATTERNS if rx.search(code))


def _block_classes(model: Any, names: tuple[str, ...]) -> dict[str, tuple[type, int]]:
    out: dict[str, tuple[type, int]] = {}
    modules = getattr(model, "modules", None)
    if not callable(modules) or not names:
        return out
    try:
        for sub in modules():
            cls = type(sub)
            if cls.__name__ in names:
                prev = out.get(cls.__name__)
                out[cls.__name__] = (cls, (prev[1] if prev else 0) + 1)
    except Exception:  # noqa: BLE001 - a probe, never a failed load
        return {}
    return out


def profile_model(model: Any) -> Optional[ModelProfile]:
    """Profile ``model``'s class (cached). None for a non-module (stubs, views)."""
    if model is None:
        return None
    cls = type(model)
    forward = getattr(cls, "forward", None)
    key = (
        f"{cls.__module__}.{cls.__qualname__}",
        source_digest(forward) if forward else None,
        _diffusers_version(),
    )
    with _LOCK:
        cached = _CACHE.get(key)
    if cached is not None:
        return cached
    declared = tuple(getattr(cls, "_repeated_blocks", None) or ())
    try:
        from ..diffusion_regional_compile import discover_repeated_blocks, verified_repeated_blocks  # noqa: PLC0415
        verified = tuple(verified_repeated_blocks(cls.__name__))
        discovered = tuple(discover_repeated_blocks(model))
    except Exception:  # noqa: BLE001
        verified, discovered = (), ()
    names = declared or verified or discovered
    blocks = []
    for name, (bcls, count) in _block_classes(model, names).items():
        src = _source(getattr(bcls, "forward", None))
        blocks.append(
            BlockProfile(
                name = name,
                count = count,
                digest = source_digest(getattr(bcls, "forward", None)),
                graph_break_suspects = screen_source(src),
                merges_streams = bool(src and _STREAM_MERGE_RX.search(src)),
            )
        )
    module_src = None
    try:
        import sys  # noqa: PLC0415
        mod = sys.modules.get(cls.__module__)
        module_src = _source(mod) if mod is not None else None
    except Exception:  # noqa: BLE001
        module_src = None
    profile = ModelProfile(
        class_name = cls.__name__,
        qualname = key[0],
        digest = key[1],
        diffusers_version = key[2],
        declared_blocks = declared,
        verified_blocks = verified,
        discovered_blocks = discovered,
        blocks = tuple(sorted(blocks, key = lambda b: b.name)),
        uses_dispatcher = bool(_DISPATCH_RX.search(module_src)) if module_src else None,
        calls_sdpa_directly = bool(module_src and _SDPA_RX.search(module_src)),
        has_compile_repeated_blocks = callable(getattr(model, "compile_repeated_blocks", None)),
    )
    with _LOCK:
        _CACHE[key] = profile
    return profile


def clear_cache() -> None:
    with _LOCK:
        _CACHE.clear()
