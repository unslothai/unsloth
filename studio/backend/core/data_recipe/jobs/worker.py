# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import json
import structlog
import loggers
import logging
import re
import shutil
import time
import traceback
import unicodedata
from pathlib import Path
from typing import Any

from ..jsonable import to_jsonable, to_preview_jsonable
from .constants import EVENT_JOB_COMPLETED, EVENT_JOB_ERROR, EVENT_JOB_STARTED
from ..service import build_config_builder, create_data_designer
from utils.paths import ensure_dir, recipe_datasets_root

_ARTIFACT_ROOT = recipe_datasets_root()
_RE_GITHUB_CURSOR = re.compile(r"\bcursor=[^\s,]+")
_RE_SECRET_TOKEN = re.compile(
    r"\b(?:(?:ghp|gho|ghu|ghs|ghr|github_pat)_[A-Za-z0-9_]+|sk-unsloth-[A-Za-z0-9]+)"
)


def _sanitize_log_message(message: str) -> str:
    message = _RE_GITHUB_CURSOR.sub("cursor=<redacted>", message)
    return _RE_SECRET_TOKEN.sub("<redacted-token>", message)


class _QueueLogHandler(logging.Handler):
    def __init__(self, event_queue):
        super().__init__()
        self._q = event_queue

    def emit(self, record: logging.LogRecord) -> None:
        try:
            event = {
                "type": "log",
                "ts": record.created,
                "level": record.levelname,
                "logger": record.name,
                "message": _sanitize_log_message(record.getMessage()),
            }
            self._q.put(event)
        except (OSError, RuntimeError, ValueError):
            pass


def _slugify_run_name(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    ascii_only = normalized.encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", ascii_only).strip("-").lower()
    if not slug:
        return ""
    return slug[:80].strip("-")


def _build_dataset_name(
    *, run_name: str | None, job_id: str, artifact_root: Path
) -> str:
    fallback = f"recipe_{job_id}"
    slug = _slugify_run_name(run_name or "")
    base_name = f"recipe_{slug}" if slug else fallback
    candidate = base_name
    suffix = 2
    while (artifact_root / candidate).exists():
        candidate = f"{base_name}_{suffix}"
        suffix += 1
    return candidate


def run_job_process(
    *,
    event_queue,
    recipe: dict[str, Any],
    run: dict[str, Any],
) -> None:
    """
    Subprocess entrypoint.
    Sends events to `event_queue`.
    """
    import os

    os.environ["PYTHONWARNINGS"] = (
        "ignore"  # Suppress warnings at C-level before imports
    )

    import warnings
    from loggers.config import LogConfig

    if os.getenv("ENVIRONMENT_TYPE", "production") == "production":
        warnings.filterwarnings("ignore")

    LogConfig.setup_logging(
        service_name = "unsloth-studio-data-worker",
        env = os.getenv("ENVIRONMENT_TYPE", "production"),
    )

    event_queue.put({"type": EVENT_JOB_STARTED, "ts": time.time()})

    try:
        from data_designer.config.run_config import RunConfig

        rows = int(run.get("rows") or 1000)
        job_id = str(run.get("_job_id") or "").strip()
        if not job_id:
            job_id = f"{int(time.time())}"
        run_name_raw = run.get("run_name")
        run_name = run_name_raw if isinstance(run_name_raw, str) else None
        dataset_name = _build_dataset_name(
            run_name = run_name,
            job_id = job_id,
            artifact_root = _ARTIFACT_ROOT,
        )
        merge_batches = bool(run.get("merge_batches"))
        ensure_dir(_ARTIFACT_ROOT)
        run_config_raw = run.get("run_config") or {}

        builder = build_config_builder(recipe)
        designer = create_data_designer(recipe, artifact_path = str(_ARTIFACT_ROOT))

        # DataDesigner configures root logging in DataDesigner.__init__.
        # Attach queue logger directly to `data_designer` so parser events survive root resets.
        handler = _QueueLogHandler(event_queue)
        handler.setLevel(logging.INFO)
        for logger_name in (
            "data_designer",
            "scraper",
            "gh_client",
            "data_designer_github_repo_seed",
        ):
            logger = logging.getLogger(logger_name)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
            logger.propagate = True

        if run_config_raw:
            designer.set_run_config(RunConfig.model_validate(run_config_raw))

        execution_type = str(run.get("execution_type") or "full").strip().lower()
        if execution_type == "preview":
            results = designer.preview(builder, num_records = rows)
            analysis = (
                None
                if results.analysis is None
                else to_jsonable(results.analysis.model_dump(mode = "json"))
            )
            dataset = (
                []
                if results.dataset is None
                else to_preview_jsonable(results.dataset.to_dict(orient = "records"))
            )
            processor_artifacts = (
                None
                if results.processor_artifacts is None
                else to_jsonable(results.processor_artifacts)
            )
            event_queue.put(
                {
                    "type": EVENT_JOB_COMPLETED,
                    "ts": time.time(),
                    "analysis": analysis,
                    "dataset": dataset,
                    "processor_artifacts": processor_artifacts,
                    "artifact_path": None,
                    "execution_type": execution_type,
                }
            )
        else:
            results = designer.create(
                builder, num_records = rows, dataset_name = dataset_name
            )
            analysis = to_jsonable(results.load_analysis().model_dump(mode = "json"))
            if merge_batches:
                _merge_batches_to_single_parquet(
                    results.artifact_storage.base_dataset_path
                )
            artifact_path = str(results.artifact_storage.base_dataset_path)
            event_queue.put(
                {
                    "type": EVENT_JOB_COMPLETED,
                    "ts": time.time(),
                    "analysis": analysis,
                    "artifact_path": artifact_path,
                    "execution_type": execution_type,
                }
            )
    except Exception as exc:
        event_queue.put(
            {
                "type": EVENT_JOB_ERROR,
                "ts": time.time(),
                "error": _sanitize_log_message(str(exc)),
                "stack": _sanitize_log_message(traceback.format_exc(limit = 20)),
            }
        )


def _merge_batches_to_single_parquet(base_dataset_path: Path) -> None:
    parquet_dir = base_dataset_path / "parquet-files"
    parquet_files = sorted(parquet_dir.glob("*.parquet"))
    if len(parquet_files) <= 1:
        return

    try:
        from data_designer.config.utils.io_helpers import read_parquet_dataset
    except ImportError:
        return

    dataframe = read_parquet_dataset(parquet_dir)
    shutil.rmtree(parquet_dir)
    parquet_dir.mkdir(parents = True, exist_ok = True)
    merged_file = parquet_dir / "batch_00000.parquet"
    dataframe.to_parquet(merged_file, index = False)
    _rewrite_merged_metadata(
        base_dataset_path = base_dataset_path,
        parquet_file = merged_file,
    )


def _rewrite_merged_metadata(*, base_dataset_path: Path, parquet_file: Path) -> None:
    metadata_path = base_dataset_path / "metadata.json"
    if not metadata_path.exists():
        return

    try:
        metadata = json.loads(metadata_path.read_text(encoding = "utf-8"))
    except (OSError, TypeError, ValueError):
        return

    if not isinstance(metadata, dict):
        return

    relative_parquet_path = str(parquet_file.relative_to(base_dataset_path))
    file_paths = metadata.get("file_paths")
    if not isinstance(file_paths, dict):
        file_paths = {}
    file_paths["parquet-files"] = [relative_parquet_path]
    metadata["file_paths"] = file_paths
    metadata["total_num_batches"] = 1
    metadata["num_completed_batches"] = 1

    try:
        metadata_path.write_text(
            json.dumps(metadata, indent = 2, sort_keys = True),
            encoding = "utf-8",
        )
    except OSError:
        return
