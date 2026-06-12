# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Bundled chat-template selection for GGUF inference.

Some shipped GGUF quants embed an older chat template. Rather than re-cutting and
asking users to re-download every quant, Studio can override the embedded template
at llama-server launch time with a bundled, up-to-date Jinja template for known
model families. The override is wired through the existing ``chat_template_override``
-> ``--chat-template-file`` path in ``LlamaCppBackend.load_model``.

Currently this covers ``unsloth/gemma-4-*-GGUF``, which gains the upstream PR #118
``preserve_thinking`` flag (defaulted OFF here) so the Studio "Preserve thinking"
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

_GEMMA4_TEMPLATE_FILE = "gemma-4.jinja"


def is_unsloth_gemma4_gguf(model_identifier: Optional[str]) -> bool:
    """True for canonical ``unsloth/gemma-4-*-GGUF`` repo identifiers."""
    if not model_identifier:
        return False
    return bool(_GEMMA4_GGUF_RE.match(model_identifier.strip()))


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
      2. For ``unsloth/gemma-4-*-GGUF``, return the bundled ``gemma-4.jinja``
         (adds ``preserve_thinking``, default off) so the embedded GGUF template
         is overridden without re-downloading quants.
      3. Otherwise ``None`` -> llama-server renders the GGUF's embedded template.

    The result is fed to ``LlamaCppBackend.load_model(chat_template_override=...)``
    and must be computed before the route-level reload-dedup check so the live
    backend state and the incoming request compare consistently.
    """
    if user_override and user_override.strip():
        return user_override
    if is_unsloth_gemma4_gguf(model_identifier):
        return load_bundled_chat_template(_GEMMA4_TEMPLATE_FILE)
    return None
