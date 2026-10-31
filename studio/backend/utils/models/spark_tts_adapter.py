# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Load Spark-TTS LoRA checkpoints whose adapter_config.json names the repo root instead of LLM/."""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional

from loggers import get_logger

logger = get_logger(__name__)


_ADAPTER_VIEW_SKIP = ("optimizer", "scheduler", "rng_state", "training_args")


def _spark_tts_llm_dir(
    base: str, hf_token: Optional[str] | bool, local_files_only: Optional[bool]
) -> Optional[Path]:
    """``<base>/LLM`` when ``base`` is a Spark-TTS repo root (no config.json of its own), else None."""
    if not base or base.replace("\\", "/").rstrip("/").lower().endswith("/llm"):
        return None
    root = Path(base).expanduser()
    if not root.is_dir():
        from huggingface_hub import snapshot_download
        from utils.hf_cache_settings import active_hf_hub_cache
        from utils.utils import hf_env_offline

        root = Path(
            snapshot_download(
                base,
                allow_patterns = ["config.json", "LLM/*"],
                token = False if hf_token is False else (hf_token or None),
                cache_dir = active_hf_hub_cache(),
                local_files_only = hf_env_offline() if local_files_only is None else local_files_only,
            )
        )
    llm = root / "LLM"
    if (root / "config.json").is_file() or not (llm / "config.json").is_file():
        return None
    return llm


@contextlib.contextmanager
def spark_tts_adapter_load_target(
    checkpoint_path: str,
    *,
    hf_token: Optional[str] | bool = None,
    local_files_only: Optional[bool] = None,
):
    """Yield a path FastModel.from_pretrained can load a Spark-TTS checkpoint from.

    Adapters saved before #13411 name the repo root (no config.json) as their base; they load through a temporary
    copy whose adapter_config.json names LLM/, leaving the checkpoint untouched.
    """
    checkpoint = Path(checkpoint_path).expanduser()
    llm = None
    try:
        adapter_config = json.loads(
            (checkpoint / "adapter_config.json").read_text(encoding = "utf-8")
        )
        base = str(adapter_config.get("base_model_name_or_path") or "").strip()
        llm = _spark_tts_llm_dir(base, hf_token, local_files_only)
    except Exception as e:  # noqa: BLE001 -- unresolvable keeps the original load and its error
        logger.debug(f"Spark-TTS adapter base not redirected: {e}")
    if llm is None:
        yield checkpoint_path
        return

    staged = Path(tempfile.mkdtemp(prefix = "unsloth_spark_adapter_"))
    try:
        for item in checkpoint.iterdir():
            if not item.is_file() or item.name == "adapter_config.json":
                continue
            if item.name.startswith(_ADAPTER_VIEW_SKIP):
                continue
            try:
                os.link(item, staged / item.name)
            except OSError:
                shutil.copy2(item, staged / item.name)
        adapter_config["base_model_name_or_path"] = str(llm)
        (staged / "adapter_config.json").write_text(
            json.dumps(adapter_config, indent = 2), encoding = "utf-8"
        )
        logger.info(f"Spark-TTS adapter base {base} has no config.json; loading it from {llm}")
        yield str(staged)
    finally:
        shutil.rmtree(staged, ignore_errors = True)
