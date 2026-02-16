"""
Checkpoint scanning utilities for discovering training runs and their checkpoints.
"""
import logging
from pathlib import Path
from typing import List, Tuple

logger = logging.getLogger(__name__)


def scan_checkpoints(outputs_dir: str = "./outputs") -> List[Tuple[str, List[Tuple[str, str]]]]:
    """
    Scan outputs folder for training runs and their checkpoints.

    Returns:
        List of tuples: [(model_name, [(display_name, checkpoint_path), ...]), ...]
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

            # Add the final model checkpoint
            checkpoints.append((item.name, str(item)))

            # Scan for intermediate checkpoints (checkpoint-N subdirs)
            for sub in sorted(item.iterdir()):
                if not sub.is_dir() or not sub.name.startswith("checkpoint-"):
                    continue
                sub_config = sub / "config.json"
                sub_adapter = sub / "adapter_config.json"
                if sub_config.exists() or sub_adapter.exists():
                    checkpoints.append((sub.name, str(sub)))

            models.append((item.name, checkpoints))
            logger.debug(f"Found model: {item.name} with {len(checkpoints)} checkpoint(s)")

        # Sort by modification time (newest first)
        models.sort(key=lambda x: Path(x[1][0][1]).stat().st_mtime, reverse=True)

        logger.info(f"Found {len(models)} training runs in {outputs_dir}")
        return models

    except Exception as e:
        logger.error(f"Error scanning checkpoints: {e}")
        return []
