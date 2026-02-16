"""
Checkpoint scanning utilities for discovering training runs and their checkpoints.
"""
import json
import logging
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


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
    outputs_dir: str = "./outputs",
) -> List[Tuple[str, List[Tuple[str, str, Optional[float]]]]]:
    """
    Scan outputs folder for training runs and their checkpoints.

    Returns:
        List of tuples: [(model_name, [(display_name, checkpoint_path, loss), ...]), ...]
        The first entry in each checkpoint list is the main adapter; its loss is
        set to the loss of the last (highest-step) intermediate checkpoint.
    """
    models = []
    outputs_path = Path(outputs_dir)

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
                checkpoints[0] = (checkpoints[0][0], checkpoints[0][1], last_checkpoint_loss)

            models.append((item.name, checkpoints))
            logger.debug(f"Found model: {item.name} with {len(checkpoints)} checkpoint(s)")

        # Sort by modification time (newest first)
        models.sort(key=lambda x: Path(x[1][0][1]).stat().st_mtime, reverse=True)

        logger.info(f"Found {len(models)} training runs in {outputs_dir}")
        return models

    except Exception as e:
        logger.error(f"Error scanning checkpoints: {e}")
        return []
