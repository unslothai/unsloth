# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Checkpoint scanning utilities for discovering training runs and checkpoints."""

import json
import re
import structlog
from loggers import get_logger
from pathlib import Path
from typing import List, Optional, Tuple
from storage.studio_db import get_connection
from utils.training_runs import (
    build_default_output_dir_name,
    extract_project_name,
    model_segment_from_default_output_dir_name,
)
from utils.paths import outputs_root, resolve_output_dir

logger = get_logger(__name__)

_CHECKPOINT_STEP_RE = re.compile(r"^checkpoint-(\d+)$")


def _checkpoint_step(checkpoint_name: str) -> Optional[int]:
    match = _CHECKPOINT_STEP_RE.fullmatch(checkpoint_name)
    if match is None:
        return None
    return int(match.group(1))


def _checkpoint_sort_key(checkpoint_path: Path) -> tuple[int, int, str]:
    step = _checkpoint_step(checkpoint_path.name)
    if step is not None:
        return (0, -step, checkpoint_path.name)
    return (1, 0, str(checkpoint_path))


def _infer_base_model_from_history(checkpoint_dir: Path) -> Optional[str]:
    """Best-effort base-model lookup using persisted Unsloth run metadata."""
    checkpoint_name = checkpoint_dir.name
    resolved_checkpoint_dir = str(checkpoint_dir.resolve())

    try:
        conn = get_connection()
    except Exception:
        return None

    try:
        exact_rows = conn.execute(
            """
            SELECT model_name
            FROM training_runs
            WHERE output_dir IN (?, ?)
            ORDER BY started_at DESC
            """,
            (
                resolved_checkpoint_dir,
                str(checkpoint_dir),
            ),
        ).fetchall()
        for row in exact_rows:
            model_name = row["model_name"]
            if model_name:
                return model_name

        suffix_rows = conn.execute(
            """
            SELECT model_name, output_dir
            FROM training_runs
            WHERE output_dir IS NOT NULL
            ORDER BY started_at DESC
            """
        ).fetchall()
        for row in suffix_rows:
            output_dir = str(row["output_dir"] or "").rstrip("/\\")
            if not (
                output_dir.endswith(f"/{checkpoint_name}")
                or output_dir.endswith(f"\\{checkpoint_name}")
            ):
                continue
            model_name = row["model_name"]
            if model_name:
                return model_name

        parts = checkpoint_name.rsplit("_", 1)
        if len(parts) != 2 or not parts[1].isdigit():
            return None

        timestamp = int(parts[1])
        generated_rows = conn.execute(
            """
            SELECT model_name, config_json
            FROM training_runs
            ORDER BY started_at DESC
            """
        ).fetchall()
        for row in generated_rows:
            model_name = row["model_name"]
            if not model_name:
                continue

            project_name = None
            config_json = row["config_json"]
            if config_json:
                try:
                    project_name = extract_project_name(json.loads(config_json))
                except (TypeError, json.JSONDecodeError):
                    project_name = None

            expected_dir_name = build_default_output_dir_name(
                model_name,
                project_name,
                timestamp = timestamp,
            )
            if expected_dir_name == checkpoint_name:
                return model_name
    except Exception:
        return None
    finally:
        conn.close()

    return None


def _read_checkpoint_loss(checkpoint_path: Path) -> Optional[float]:
    """Read loss from the last log_history entry of trainer_state.json, or None."""
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
    """Scan outputs folder for training runs and their checkpoints.

    Returns:
        [(model_name, [(display_name, checkpoint_path, loss), ...], metadata), ...]
        metadata keys (optional): base_model, peft_type, lora_rank.
        First checkpoint entry is the main adapter; its loss mirrors the latest
        (highest-step) intermediate checkpoint. Numbered checkpoints are sorted
        by numeric step descending; non-numbered checkpoint-* dirs keep the
        previous lexicographic directory order.
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

            # Training metadata from adapter_config.json / config.json
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

                # Detect BNB quantization from config.json
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

            # Fallback: extract base model name from the folder name, e.g.
            # "unsloth_Llama-3.2-3B-Instruct_1771227800" → "unsloth/Llama-3.2-3B-Instruct"
            if not metadata.get("base_model"):
                metadata["base_model"] = _infer_base_model_from_history(item)

            if not metadata.get("base_model"):
                name_part = model_segment_from_default_output_dir_name(item.name)
                if name_part:
                    idx = name_part.find("_")
                    if idx > 0:
                        metadata["base_model"] = name_part[:idx] + "/" + name_part[idx + 1 :]
                    else:
                        metadata["base_model"] = name_part

            # Valid training run.
            checkpoints = []

            # Main adapter placeholder — loss filled from the last checkpoint below.
            checkpoints.append((item.name, str(item), None))

            # Scan for intermediate checkpoints (checkpoint-N subdirs).
            valid_checkpoints = []
            for sub in item.iterdir():
                if not sub.is_dir() or not sub.name.startswith("checkpoint-"):
                    continue
                sub_config = sub / "config.json"
                sub_adapter = sub / "adapter_config.json"
                if sub_config.exists() or sub_adapter.exists():
                    valid_checkpoints.append(sub)

            intermediate_checkpoints = []
            for sub in sorted(valid_checkpoints, key = _checkpoint_sort_key):
                loss = _read_checkpoint_loss(sub)
                intermediate_checkpoints.append((sub.name, str(sub), loss))

            checkpoints.extend(intermediate_checkpoints)

            # Assign the latest checkpoint's loss to the main adapter entry.
            if intermediate_checkpoints:
                last_checkpoint_loss = intermediate_checkpoints[0][2]
                checkpoints[0] = (
                    checkpoints[0][0],
                    checkpoints[0][1],
                    last_checkpoint_loss,
                )

            models.append((item.name, checkpoints, metadata))
            logger.debug(f"Found model: {item.name} with {len(checkpoints)} checkpoint(s)")

        # Sort by modification time (newest first)
        models.sort(key = lambda x: Path(x[1][0][1]).stat().st_mtime, reverse = True)

        logger.debug(f"Found {len(models)} training runs in {outputs_dir}")
        return models

    except Exception as e:
        logger.error(f"Error scanning checkpoints: {e}")
        return []


def _is_model_dir(path: Path) -> bool:
    return (path / "config.json").exists() or (path / "adapter_config.json").exists()


def has_preview_model(output_dir: Optional[str]) -> bool:
    """True when ``output_dir`` holds a previewable root model (what ``/p/{run}``
    resolves). A cancelled run keeps ``output_dir`` but saves no root adapter."""
    if not output_dir:
        return False
    path = Path(output_dir)
    return path.is_dir() and _is_model_dir(path)


def preview_ref(output_dir: Optional[str]) -> Optional[str]:
    """``/p`` ref (``run`` or ``run/checkpoint``) relative to outputs_root, or None.

    Posix-joined so a nested output dir keeps a working link instead of collapsing
    to its basename. None when not previewable, outside outputs_root, or deeper than
    the two path segments the ``/p`` route matches (so the UI omits a dead link).
    """
    if not has_preview_model(output_dir):
        return None
    try:
        rel = Path(output_dir).resolve().relative_to(outputs_root().resolve())
    except (ValueError, OSError):
        return None
    parts = rel.parts
    if not parts or len(parts) > 2:
        return None
    return "/".join(parts)


def resolve_preview_checkpoint(run: str, checkpoint: Optional[str] = None) -> Path:
    relative = run if not checkpoint else f"{run}/{checkpoint}"
    path = resolve_output_dir(relative)
    if not path.is_dir() or not _is_model_dir(path):
        raise FileNotFoundError(
            f"No trained checkpoint at '{relative}'. Check the run/checkpoint name (see GET /p)."
        )
    return path


def list_preview_targets(outputs_dir: str = str(outputs_root())) -> List[dict]:
    targets: List[dict] = []
    for run_name, checkpoints, metadata in scan_checkpoints(outputs_dir):
        for display_name, path, loss in checkpoints:
            is_latest = display_name == run_name
            checkpoint = None if is_latest else Path(path).name
            targets.append(
                {
                    "run": run_name,
                    "checkpoint": checkpoint,
                    "ref": run_name if is_latest else f"{run_name}/{checkpoint}",
                    "is_latest": is_latest,
                    "loss": loss,
                    "base_model": metadata.get("base_model"),
                }
            )
    return targets
