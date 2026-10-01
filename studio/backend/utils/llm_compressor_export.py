# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Consent probe for FP8/FP4 compressed-tensors export (llm-compressor)."""

from __future__ import annotations

import ast
import importlib.metadata
import importlib.util
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict

from utils.transformers_version import (
    _VENV_LLMCOMPRESSOR_DIR,
    _env_offline,
    _llmcompressor_main_disabled,
    _llmcompressor_shadow_is_valid,
)


def _llm_compressor_spec() -> str:
    # Read from unsloth/save.py without importing unsloth: this runs in the Studio parent, which must
    # not pull torch / a GPU context in (see main.py).
    try:
        spec = importlib.util.find_spec("unsloth")
        tree = ast.parse(Path(spec.origin).with_name("save.py").read_text(encoding = "utf-8"))
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                getattr(t, "id", None) == "_LLM_COMPRESSOR_SPEC" for t in node.targets
            ):
                return ast.literal_eval(node.value)
    except Exception:
        pass
    return "llmcompressor"


def _workspace_llmcompressor_ok(spec: str) -> bool:
    try:
        from packaging.requirements import Requirement
        version = importlib.metadata.version("llmcompressor")
        return Requirement(spec).specifier.contains(version, prereleases = True)
    except Exception:
        return False


def probe_llm_compressor_for_compressed_export() -> Dict[str, Any]:
    """Mirror export.py's runtime choice (shadow first, else workspace) without installing anything."""
    spec = _llm_compressor_spec()
    shadow_valid = _llmcompressor_shadow_is_valid()
    shadow_disabled = _llmcompressor_main_disabled()
    offline = _env_offline()
    autoinstall_disabled = os.environ.get(
        "UNSLOTH_DISABLE_LLM_COMPRESSOR_AUTOINSTALL", "0"
    ).lower() not in ("0", "", "false", "no")
    workspace_ok = _workspace_llmcompressor_ok(spec)

    # Any provisionable shadow is asked for even when the workspace copy exists: export.py prefers it,
    # and the workspace copy cannot run models above its transformers ceiling.
    consent_kind = None
    install_summary = None
    blocked_reason = None
    if shadow_valid:
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
        "workspace_install_command": f"uv pip install --python {sys.executable} '{spec}'",
        "shadow_path": _VENV_LLMCOMPRESSOR_DIR,
        "autoinstall_disabled": autoinstall_disabled,
        "shadow_disabled": shadow_disabled,
        "offline": offline,
        "blocked_reason": blocked_reason,
        "python_executable": sys.executable,
        "has_pip": importlib.util.find_spec("pip") is not None,
        "has_uv": bool(shutil.which("uv")),
    }
