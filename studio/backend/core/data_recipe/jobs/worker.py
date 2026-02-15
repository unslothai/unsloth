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
    handler = _QueueLogHandler(event_queue)
    handler.setLevel(logging.INFO)

    # Attach once to root. data_designer.* loggers should propagate to root.
    root = logging.getLogger()
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    dd = logging.getLogger("data_designer")
    dd.setLevel(logging.INFO)
    dd.propagate = True

    event_queue.put({"type": "job.started", "ts": time.time()})

    try:
        # lazy import so backend can boot even if data-designer isn't installed yet
        from data_designer.config.run_config import RunConfig

        rows = int(run.get("rows") or 1000)
        dataset_name = str(run.get("dataset_name") or "dataset")
        run_config_raw = run.get("run_config") or {}

        builder = build_config_builder(recipe)
        designer = create_data_designer(recipe)

        if run_config_raw:
            designer.set_run_config(RunConfig.model_validate(run_config_raw))

        results = designer.create(builder, num_records=rows, dataset_name=dataset_name)

        analysis = results.load_analysis().model_dump(mode="json")
        artifact_path = str(results.artifact_storage.base_dataset_path)

        event_queue.put(
            {
                "type": "job.completed",
                "ts": time.time(),
                "analysis": analysis,
                "artifact_path": artifact_path,
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

