# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""FSDP launch detection from the launcher environment alone (no torch, no accelerate).

Every rank of one job inherits the same launcher env, so every answer here is the same on
every rank. `unsloth/__init__.py` calls `apply_unsloth_fsdp2_launch_env` before transformers or
accelerate are imported, so `UNSLOTH_FSDP2=1` under plain `torchrun` looks exactly like
`accelerate launch` with an FSDP2 config to everything downstream.
"""

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
    """`UNSLOTH_FSDP2=1`: train with FSDP2 under a launcher that exports no FSDP config (plain torchrun)."""
    return _env_flag("UNSLOTH_FSDP2")


def fsdp_version_from_env():
    """The FSDP version the launcher env asks for: 0 for none, else 1 or 2.

    `accelerate launch` exports ACCELERATE_USE_FSDP=true and FSDP_VERSION for an FSDP config and
    neither for DDP; torchrun users may export FSDP_VERSION by hand. accelerate builds FSDP1 when
    ACCELERATE_USE_FSDP names no version, and "0" is how a config says "not FSDP".
    """
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


def apply_unsloth_fsdp2_launch_env():
    """Export what `accelerate launch` exports for an FSDP2 config when `UNSLOTH_FSDP2=1`.

    Only on a multi-process launch (WORLD_SIZE > 1); a single process is left alone. Explicit
    FSDP_* settings win over the defaults filled here. `UNSLOTH_FSDP2_RESHARD_AFTER_FORWARD=0`
    keeps each block's parameters gathered from forward to backward (ZeRO-2 like: one all-gather
    per block per step instead of two, more memory; usually faster over PCIe).
    Returns True when the env was applied.
    """
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
    os.environ["ACCELERATE_USE_FSDP"] = "true"
    os.environ["FSDP_VERSION"] = "2"
    # One FSDP unit per decoder block (`_no_split_modules`); accelerate's own default is the root only.
    os.environ.setdefault("FSDP_AUTO_WRAP_POLICY", "TRANSFORMER_BASED_WRAP")
    reshard = os.environ.get("UNSLOTH_FSDP2_RESHARD_AFTER_FORWARD", None)
    if reshard is not None and reshard.strip() != "":
        os.environ.setdefault(
            "FSDP_RESHARD_AFTER_FORWARD",
            "false" if reshard.strip().lower() in _FALSY else "true",
        )
    # What TrainingArguments(fsdp = ...) picks, so trainer.save_model() writes ordinary files.
    os.environ.setdefault("FSDP_STATE_DICT_TYPE", "FULL_STATE_DICT")
    return True
