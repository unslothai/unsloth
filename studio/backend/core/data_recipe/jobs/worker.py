from __future__ import annotations

import logging
import time
import traceback
from typing import Any

from ..service import build_config_builder, create_data_designer


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
                "message": record.getMessage(),
            }
            self._q.put(event)
        except Exception:
            pass


def _to_jsonable(value: Any) -> Any:
    try:
        import numpy as np  # type: ignore
    except Exception:  # pragma: no cover
        np = None  # type: ignore

    if np is not None:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()

    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(v) for v in value]

    if hasattr(value, "isoformat") and callable(value.isoformat):
        try:
            return value.isoformat()
        except Exception:
            pass

    return value


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
    event_queue.put({"type": "job.started", "ts": time.time()})

    try:
        # Importing data_designer.interface.* triggers DataDesigner logging setup (it clears root handlers),
        # so attach our queue handler after that import.
        from data_designer.config.run_config import RunConfig

        import data_designer.interface.data_designer  # noqa: F401

        handler = _QueueLogHandler(event_queue)
        handler.setLevel(logging.INFO)
        root = logging.getLogger()
        root.addHandler(handler)
        root.setLevel(logging.INFO)
        logging.getLogger("data_designer").setLevel(logging.INFO)

        rows = int(run.get("rows") or 1000)
        dataset_name = str(run.get("dataset_name") or "dataset")
        artifact_path_raw = run.get("artifact_path")
        artifact_path = None
        if isinstance(artifact_path_raw, str) and artifact_path_raw.strip():
            artifact_path = artifact_path_raw.strip()
        run_config_raw = run.get("run_config") or {}

        builder = build_config_builder(recipe)
        designer = create_data_designer(recipe, artifact_path=artifact_path)

        if run_config_raw:
            designer.set_run_config(RunConfig.model_validate(run_config_raw))

        execution_type = str(run.get("execution_type") or "full").strip().lower()
        if execution_type == "preview":
            results = designer.preview(builder, num_records=rows)
            analysis = (
                None
                if results.analysis is None
                else _to_jsonable(results.analysis.model_dump(mode="json"))
            )
            dataset = (
                []
                if results.dataset is None
                else _to_jsonable(results.dataset.to_dict(orient="records"))
            )
            processor_artifacts = (
                None
                if results.processor_artifacts is None
                else _to_jsonable(results.processor_artifacts)
            )
            event_queue.put(
                {
                    "type": "job.completed",
                    "ts": time.time(),
                    "analysis": analysis,
                    "dataset": dataset,
                    "processor_artifacts": processor_artifacts,
                    "artifact_path": None,
                    "execution_type": execution_type,
                }
            )
        else:
            results = designer.create(builder, num_records=rows, dataset_name=dataset_name)
            analysis = _to_jsonable(results.load_analysis().model_dump(mode="json"))
            artifact_path = str(results.artifact_storage.base_dataset_path)
            event_queue.put(
                {
                    "type": "job.completed",
                    "ts": time.time(),
                    "analysis": analysis,
                    "artifact_path": artifact_path,
                    "execution_type": execution_type,
                }
            )
    except Exception as exc:
        event_queue.put(
            {
                "type": "job.error",
                "ts": time.time(),
                "error": str(exc),
                "stack": traceback.format_exc(limit=20),
            }
        )
