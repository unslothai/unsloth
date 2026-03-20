# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Checkpoint scanning utilities for discovering training runs and their checkpoints.
"""

import json
import structlog
from loggers import get_logger
from pathlib import Path
from typing import List, Optional, Tuple
from utils.paths import outputs_root, resolve_output_dir

logger = get_logger(__name__)


def _read_checkpoint_loss(checkpoint_path: Path) -> Optional[float]:
    """
    Read the training loss from a checkpoint's trainer_state.json.

    Returns the loss from the last log_history entry, or None if unavailable.
    """
    trainer_state = checkpoint_path / "trainer_state.json"
    if not trainer_state.exists():
        return None
    try:
        with open(trainer_state) as f:
            state = json.load(f)
        log_history = state.get("log_history", [])
        if log_history:
            return log_history[-1].get("loss")
    except Exception as e:
        logger.debug(f"Could not read loss from {trainer_state}: {e}")
    return None


def scan_checkpoints(
    outputs_dir: str = str(outputs_root()),
) -> List[Tuple[str, List[Tuple[str, str, Optional[float]]], dict]]:
    """
    Scan outputs folder for training runs and their checkpoints.

    Returns:
        List of tuples: [(model_name, [(display_name, checkpoint_path, loss), ...], metadata), ...]
        metadata keys: base_model, peft_type, lora_rank (all optional)
        The first entry in each checkpoint list is the main adapter; its loss is
        set to the loss of the last (highest-step) intermediate checkpoint.
    """
    models = []
    outputs_path = resolve_output_dir(outputs_dir)

    if not outputs_path.exists():
        logger.warning(f"Outputs directory not found: {outputs_dir}")
        return models

    try:
        for item in outputs_path.iterdir():
            if not item.is_dir():
                continue

            config_file = item / "config.json"
            adapter_config = item / "adapter_config.json"

            if not (config_file.exists() or adapter_config.exists()):
                continue

            # Extract training metadata from adapter_config.json / config.json
            metadata: dict = {}
            try:
                if adapter_config.exists():
                    cfg = json.loads(adapter_config.read_text())
                    metadata["base_model"] = cfg.get("base_model_name_or_path")
                    metadata["peft_type"] = cfg.get("peft_type")
                    metadata["lora_rank"] = cfg.get("r")
                elif config_file.exists():
                    cfg = json.loads(config_file.read_text())
                    metadata["base_model"] = cfg.get("_name_or_path")

                # Detect BNB quantization from config.json (present in both cases)
                if config_file.exists():
                    if "cfg" not in dir():
                        cfg = json.loads(config_file.read_text())
                    quant_cfg = cfg.get("quantization_config")
                    if (
                        isinstance(quant_cfg, dict)
                        and quant_cfg.get("quant_method") == "bitsandbytes"
                    ):
                        metadata["is_quantized"] = True
                        logger.info("Detected BNB-quantized model: %s", item.name)
            except Exception:
                pass

            # Fallback: extract base model name from folder name
            # e.g. "unsloth_Llama-3.2-3B-Instruct_1771227800" → "unsloth/Llama-3.2-3B-Instruct"
            if not metadata.get("base_model"):
                parts = item.name.rsplit("_", 1)
                if len(parts) == 2 and parts[1].isdigit():
                    name_part = parts[0]
                    idx = name_part.find("_")
                    if idx > 0:
                        metadata["base_model"] = (
                            name_part[:idx] + "/" + name_part[idx + 1 :]
                        )
                    else:
                        metadata["base_model"] = name_part

            # This is a valid training run
            checkpoints = []

            # Placeholder for the main adapter — loss filled from last checkpoint below
            checkpoints.append((item.name, str(item), None))

            # Scan for intermediate checkpoints (checkpoint-N subdirs)
            for sub in sorted(item.iterdir()):
                if not sub.is_dir() or not sub.name.startswith("checkpoint-"):
                    continue
                sub_config = sub / "config.json"
                sub_adapter = sub / "adapter_config.json"
                if sub_config.exists() or sub_adapter.exists():
                    loss = _read_checkpoint_loss(sub)
                    checkpoints.append((sub.name, str(sub), loss))

            # Assign the last checkpoint's loss to the main adapter entry
            if len(checkpoints) > 1:
                last_checkpoint_loss = checkpoints[-1][2]
                checkpoints[0] = (
                    checkpoints[0][0],
                    checkpoints[0][1],
                    last_checkpoint_loss,
                )

            models.append((item.name, checkpoints, metadata))
            logger.debug(
                f"Found model: {item.name} with {len(checkpoints)} checkpoint(s)"
            )

        # Sort by modification time (newest first)
        models.sort(key = lambda x: Path(x[1][0][1]).stat().st_mtime, reverse = True)

        logger.info(f"Found {len(models)} training runs in {outputs_dir}")
        return models

    except Exception as e:
        logger.error(f"Error scanning checkpoints: {e}")
        return []
