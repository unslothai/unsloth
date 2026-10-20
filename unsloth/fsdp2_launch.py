# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""FSDP launch detection from the launcher env only (no torch): identical on every rank."""

import importlib.util
import os

__all__ = [
    "unsloth_fsdp2_requested",
    "fsdp_version_from_env",
    "apply_unsloth_fsdp2_launch_env",
]

_TRUTHY = ("1", "true", "yes", "on")
_FALSY = ("0", "false", "no", "off")


def _env_flag(name):
    return str(os.environ.get(name, "")).strip().lower() in _TRUTHY


def _env_world_size():
    try:
        return int(os.environ.get("WORLD_SIZE", "1"))
    except ValueError:
        return 1


def unsloth_fsdp2_requested():
    return _env_flag("UNSLOTH_FSDP2")


def fsdp_version_from_env():
    """0 for no FSDP, else 1 or 2. accelerate builds FSDP1 when ACCELERATE_USE_FSDP names no
    version; FSDP_VERSION="0" means "not FSDP"."""
    if unsloth_fsdp2_requested() and _env_world_size() > 1:
        return 2
    version = str(os.environ.get("FSDP_VERSION", "")).strip()
    use_fsdp = _env_flag("ACCELERATE_USE_FSDP")
    if version in ("", "0"):
        return 1 if use_fsdp else 0
    try:
        return int(version)
    except ValueError:
        return 1 if use_fsdp else 0


def _accelerate_fsdp2_support():
    """None when accelerate is not found, else whether it has FSDP2 (1.6) and FSDP2 state dict
    options (1.7; 1.6 saves FULL_STATE_DICT checkpoints as each rank's DTensor shard). Read from
    the source: importing accelerate here would import torch."""
    try:
        spec = importlib.util.find_spec("accelerate")
    except (ImportError, ValueError):
        return None
    for location in getattr(spec, "submodule_search_locations", None) or ():
        try:
            with open(os.path.join(location, "utils", "fsdp_utils.py"), encoding = "utf-8") as f:
                source = f.read()
        except OSError:
            continue
        return "def fsdp2_prepare_model" in source and "def _prepare_sd_options" in source
    return None


def apply_unsloth_fsdp2_launch_env():
    """UNSLOTH_FSDP2=1 with WORLD_SIZE > 1: export what `accelerate launch` exports for an FSDP2
    config (explicit FSDP_* values win). Must run before transformers / accelerate read the env."""
    if not unsloth_fsdp2_requested() or _env_world_size() <= 1:
        return False
    if _env_flag("ACCELERATE_USE_DEEPSPEED"):
        raise RuntimeError(
            "Unsloth: UNSLOTH_FSDP2=1 was set on a DeepSpeed launch. Unset one of them."
        )
    version = str(os.environ.get("FSDP_VERSION", "")).strip()
    if version not in ("", "2"):
        raise ValueError(
            f"Unsloth: UNSLOTH_FSDP2=1 asks for FSDP2, but FSDP_VERSION={version}. Unset one of them."
        )
    if _accelerate_fsdp2_support() is False:
        # Older accelerate ignores FSDP_VERSION and builds FSDP1 from these variables.
        raise RuntimeError(
            "Unsloth: UNSLOTH_FSDP2=1 needs accelerate >= 1.7, which this accelerate is older than. "
            "Upgrade it with `pip install -U accelerate`."
        )
    os.environ["ACCELERATE_USE_FSDP"] = "true"
    os.environ["FSDP_VERSION"] = "2"
    # accelerate's default is the root only.
    os.environ.setdefault("FSDP_AUTO_WRAP_POLICY", "TRANSFORMER_BASED_WRAP")
    reshard = os.environ.get("UNSLOTH_FSDP2_RESHARD_AFTER_FORWARD", None)
    if reshard is not None and reshard.strip() != "":
        os.environ.setdefault(
            "FSDP_RESHARD_AFTER_FORWARD",
            "false" if reshard.strip().lower() in _FALSY else "true",
        )
    os.environ.setdefault("FSDP_STATE_DICT_TYPE", "FULL_STATE_DICT")
    return True
