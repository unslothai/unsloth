# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Bundled chat-template selection for GGUF inference.

Some shipped GGUF quants embed an older chat template. Rather than re-cutting and
asking users to re-download every quant, Unsloth can override the embedded template
at llama-server launch time with a bundled, up-to-date Jinja template for known
model families. The override is wired through the existing ``chat_template_override``
-> ``--chat-template-file`` path in ``LlamaCppBackend.load_model``.

Currently this covers ``unsloth/gemma-4-*-GGUF``, which gains the upstream PR #118
``preserve_thinking`` flag (defaulted OFF here) so the Unsloth "Preserve thinking"
toggle appears while staying disabled by default.
"""

import re
from functools import lru_cache
from pathlib import Path
from typing import Optional

# assets live at <backend>/assets/chat_templates/. This module is at
# <backend>/core/inference/chat_templates.py, so walk up three parents to
# <backend> (mirrors utils/inference/inference_config.py).
_ASSETS_DIR = Path(__file__).parent.parent.parent / "assets" / "chat_templates"

# unsloth/gemma-4-<variant>-GGUF (case-insensitive). The "-GGUF" suffix is retained
# on ModelConfig.identifier for HF GGUF repos, so this matches E2B / E4B / 31B /
# 26B-A4B and any future unsloth/gemma-4-*-GGUF, while excluding gemma-3,
# non-Unsloth, and non-GGUF identifiers (e.g. the bf16 "unsloth/gemma-4-E2B-it").
_GEMMA4_GGUF_RE = re.compile(r"^unsloth/gemma-4-.+-gguf$", re.IGNORECASE)

# Google ships two distinct gemma-4 chat templates: E2B/E4B omit the empty
# "<|channel>thought<channel|>" block on enable_thinking=false, while the
# 12b/26B-A4B/31B family emits it. Route the two GGUF families to the matching
# bundled template so each keeps its model's intended behavior.
_GEMMA4_EDGE_GGUF_RE = re.compile(r"^unsloth/gemma-4-e[24]b-it-gguf$", re.IGNORECASE)

_GEMMA4_TEMPLATE_FILE = "gemma-4.jinja"            # 12b / 26B-A4B / 31B
_GEMMA4_EDGE_TEMPLATE_FILE = "gemma-4-edge.jinja"  # E2B / E4B


def _canonical_repo_id(model_identifier: str) -> str:
    """Mirror ``ModelConfig.from_identifier``: a bare HF shorthand with no owner
    (e.g. ``gemma-4-E2B-it-GGUF``) defaults to the ``unsloth/`` org. The resolver
    runs on the raw ``request.model_path`` (before that canonicalization), so apply
    the same rule here, otherwise shorthand loads would skip the override.
    """
    mid = model_identifier.strip()
    if mid and "/" not in mid:
        mid = f"unsloth/{mid}"
    return mid


def is_unsloth_gemma4_gguf(model_identifier: Optional[str]) -> bool:
    """True for canonical ``unsloth/gemma-4-*-GGUF`` repo identifiers (and the
    owner-less shorthand that resolves to the same Unsloth repo)."""
    if not model_identifier:
        return False
    return bool(_GEMMA4_GGUF_RE.match(_canonical_repo_id(model_identifier)))


def is_unsloth_gemma4_edge_gguf(model_identifier: Optional[str]) -> bool:
    """True for the E2B / E4B GGUF repos, which use the edge-variant template."""
    if not model_identifier:
        return False
    return bool(_GEMMA4_EDGE_GGUF_RE.match(_canonical_repo_id(model_identifier)))


def _gemma4_template_file(model_identifier: Optional[str]) -> Optional[str]:
    """Return the bundled template filename for a gemma-4 GGUF id, else None."""
    if is_unsloth_gemma4_edge_gguf(model_identifier):
        return _GEMMA4_EDGE_TEMPLATE_FILE
    if is_unsloth_gemma4_gguf(model_identifier):
        return _GEMMA4_TEMPLATE_FILE
    return None


@lru_cache(maxsize=8)
def load_bundled_chat_template(name: str) -> str:
    """Read a bundled chat-template asset by filename (cached for the process)."""
    return (_ASSETS_DIR / name).read_text(encoding="utf-8")


def resolve_effective_chat_template_override(
    *,
    model_identifier: Optional[str],
    user_override: Optional[str],
) -> Optional[str]:
    """Resolve which chat-template text to launch llama-server with.

    Precedence:
      1. An explicit, non-empty user override always wins (advanced users).
      2. For ``unsloth/gemma-4-*-GGUF``, return the bundled gemma-4 template
         (adds ``preserve_thinking``, default off) so the embedded GGUF template
         is overridden without re-downloading quants. E2B/E4B get the edge
         variant; 12b/26B-A4B/31B get the standard one.
      3. Otherwise ``None`` -> llama-server renders the GGUF's embedded template.

    The result is fed to ``LlamaCppBackend.load_model(chat_template_override=...)``
    and must be computed before the route-level reload-dedup check so the live
    backend state and the incoming request compare consistently.
    """
    if user_override and user_override.strip():
        return user_override
    template_file = _gemma4_template_file(model_identifier)
    if template_file is not None:
        return load_bundled_chat_template(template_file)
    return None
