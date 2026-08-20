# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Probe and consent helpers for FP8/FP4 compressed-tensors export (llm-compressor)."""

from __future__ import annotations

import importlib.util
import sys
from typing import Any, Dict, Literal, Optional

from utils.transformers_version import (
    _VENV_LLMCOMPRESSOR_DIR,
    _env_offline,
    _llmcompressor_main_disabled,
    _llmcompressor_shadow_is_valid,
)


def _workspace_llmcompressor_importable() -> bool:
    try:
        import llmcompressor  # noqa: F401
        return True
    except Exception:
        return False


def probe_llm_compressor_for_compressed_export() -> Dict[str, Any]:
    """Return whether compressed export can run now and what consent would install.

    Studio calls this before starting a compressed-tensors export. When ``ready`` is false and
    ``needs_consent`` is true, the UI should show ``install_summary`` and only proceed after the
    user accepts, passing ``install_missing_dependencies=True`` on the export request.
    """
    import unsloth.save as us

    shadow_valid = _llmcompressor_shadow_is_valid()
    workspace_ok = _workspace_llmcompressor_importable()
    shadow_disabled = _llmcompressor_main_disabled()
    offline = _env_offline()
    autoinstall_disabled = us._llm_compressor_autoinstall_disabled()

    ready = shadow_valid or workspace_ok
    consent_kind: Optional[Literal["shadow", "workspace"]] = None
    if not ready:
        if not shadow_disabled and not offline and not shadow_valid:
            consent_kind = "shadow"
        elif not workspace_ok and not autoinstall_disabled:
            consent_kind = "workspace"
        elif not workspace_ok and autoinstall_disabled:
            consent_kind = None

    needs_consent = consent_kind is not None

    if consent_kind == "shadow":
        install_summary = (
            f"Unsloth will provision a one-time llm-compressor runtime at "
            f"{_VENV_LLMCOMPRESSOR_DIR} (pinned packages, your torch is not upgraded)."
        )
    elif consent_kind == "workspace":
        install_summary = (
            "Unsloth will install a pinned llm-compressor into this Studio Python "
            "environment (torch and transformers stay pinned to your current versions)."
        )
    else:
        install_summary = None

    blocked_reason = None
    if not ready and not needs_consent:
        if shadow_disabled and not workspace_ok and autoinstall_disabled:
            blocked_reason = (
                "llm-compressor is not installed and both automatic installation and the "
                "llm-compressor-main shadow are disabled. Install llm-compressor manually or "
                "unset UNSLOTH_DISABLE_LLM_COMPRESSOR_AUTOINSTALL / "
                "UNSLOTH_DISABLE_LLMCOMPRESSOR_MAIN."
            )
        elif offline and not workspace_ok:
            blocked_reason = (
                "llm-compressor is not available and offline mode prevents provisioning the "
                "llm-compressor-main shadow. Install llm-compressor while online, or disable "
                "offline mode."
            )
        elif shadow_disabled and not workspace_ok:
            blocked_reason = (
                "llm-compressor is not installed in this environment and the llm-compressor-main "
                "shadow is disabled. Install it manually or allow the shadow runtime."
            )
        else:
            blocked_reason = (
                "llm-compressor is required for FP8/FP4 compressed export but is not available."
            )

    return {
        "ready": ready,
        "needs_consent": needs_consent,
        "consent_kind": consent_kind,
        "install_summary": install_summary,
        "workspace_install_command": us.llm_compressor_manual_install_command(),
        "shadow_path": _VENV_LLMCOMPRESSOR_DIR,
        "autoinstall_disabled": autoinstall_disabled,
        "shadow_disabled": shadow_disabled,
        "offline": offline,
        "blocked_reason": blocked_reason,
        "python_executable": sys.executable,
        "has_pip": importlib.util.find_spec("pip") is not None,
        "has_uv": bool(__import__("shutil").which("uv")),
    }
