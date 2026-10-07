# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Consent probe for FP8/FP4 compressed-tensors export (llm-compressor)."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import os
import shutil
import sys
from typing import Any, Dict

from utils.transformers_version import (
    _VENV_LLMCOMPRESSOR_DIR,
    _env_offline,
    _llmcompressor_main_disabled,
    _llmcompressor_shadow_is_valid,
)


# Keep in sync with unsloth/save.py _LLM_COMPRESSOR_SPEC; not imported to avoid torch.
_LLM_COMPRESSOR_SPEC = "llmcompressor>=0.6.0,<=0.12.0"


def _workspace_llmcompressor_ok(spec: str) -> bool:
    try:
        from packaging.requirements import Requirement
        version = importlib.metadata.version("llmcompressor")
        return Requirement(spec).specifier.contains(version, prereleases = True)
    except Exception:
        return False


def probe_llm_compressor_for_compressed_export() -> Dict[str, Any]:
    """Mirror export.py's runtime choice (shadow first, else workspace) without installing anything."""
    spec = _LLM_COMPRESSOR_SPEC
    shadow_valid = _llmcompressor_shadow_is_valid()
    shadow_disabled = _llmcompressor_main_disabled()
    offline = _env_offline()
    autoinstall_disabled = os.environ.get(
        "UNSLOTH_DISABLE_LLM_COMPRESSOR_AUTOINSTALL", "0"
    ).lower() not in ("0", "", "false", "no")
    workspace_ok = _workspace_llmcompressor_ok(spec)
    # Same pip-first choice as save.py's llm_compressor_manual_install_command.
    has_pip = importlib.util.find_spec("pip") is not None

    # Ask for a provisionable shadow even if the workspace copy exists: export.py prefers it.
    consent_kind = None
    install_summary = None
    blocked_reason = None
    # export.py ignores the shadow once UNSLOTH_DISABLE_LLMCOMPRESSOR_MAIN is set.
    if shadow_valid and not shadow_disabled:
        ready = True
    elif not shadow_disabled and not offline:
        ready = False
        consent_kind = "shadow"
        install_summary = (
            f"Unsloth will provision a one-time llm-compressor runtime at "
            f"{_VENV_LLMCOMPRESSOR_DIR} (pinned packages, your torch is not upgraded)."
        )
    elif workspace_ok:
        ready = True
    elif not autoinstall_disabled:
        ready = False
        consent_kind = "workspace"
        install_summary = (
            "Unsloth will install a pinned llm-compressor into this Studio Python "
            "environment (torch and transformers stay pinned to your current versions)."
        )
    else:
        ready = False
        blocked_reason = (
            "llm-compressor is not installed and automatic installation is disabled via "
            "UNSLOTH_DISABLE_LLM_COMPRESSOR_AUTOINSTALL. Install it manually or unset that variable."
        )

    return {
        "ready": ready,
        "needs_consent": consent_kind is not None,
        "consent_kind": consent_kind,
        "install_summary": install_summary,
        "workspace_install_command": (
            f"{sys.executable} -m pip install '{spec}'"
            if has_pip
            else f"uv pip install --python {sys.executable} '{spec}'"
        ),
        "shadow_path": _VENV_LLMCOMPRESSOR_DIR,
        "autoinstall_disabled": autoinstall_disabled,
        "shadow_disabled": shadow_disabled,
        "offline": offline,
        "blocked_reason": blocked_reason,
        "python_executable": sys.executable,
        "has_pip": has_pip,
        "has_uv": bool(shutil.which("uv")),
    }
