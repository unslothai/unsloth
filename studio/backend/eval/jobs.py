# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from storage import studio_db

# run_fn(config, *, on_result, should_cancel) -> EvalSummary
# on_result(idx, score, prediction, input_text, reference, breakdown, error)
RunFn = Callable[..., Any]


class EvalBusyError(RuntimeError):
    """Raised when an eval is already running (single shared model)."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class EvalJobManager:
    def __init__(self, run_fn: RunFn):
        self._run_fn = run_fn
        self._lock = threading.Lock()
        self._active_run_id: Optional[str] = None
        self._cancel = threading.Event()
        self._progress: dict[str, dict] = {}
        self._thread: Optional[threading.Thread] = None

    def is_running(self) -> bool:
        return self._active_run_id is not None

    def start(self, req) -> str:
        with self._lock:
            if self._active_run_id is not None:
                raise EvalBusyError("an eval is already running")
            run_id = uuid.uuid4().hex
            self._active_run_id = run_id
            self._cancel.clear()
        total = req.limit if req.limit is not None else 0
        dataset_ref = req.dataset.path or req.dataset.name or "?"
        studio_db.create_eval_run(
            id=run_id, model_identifier=req.model_identifier,
            dataset_ref=str(dataset_ref), metric_name=req.metric_name,
            config_json=req.model_dump_json(), started_at=_now(),
            num_examples=req.limit,
        )
        self._progress[run_id] = {
            "run_id": run_id, "status": "running", "done": 0,
            "total": total, "avg_score": 0.0, "last_result": None,
        }
        self._thread = threading.Thread(
            target=self._run, args=(run_id, req), daemon=True
        )
        self._thread.start()
        return run_id

    def _run(self, run_id: str, req) -> None:
        running_total = {"sum": 0.0, "n": 0}

        def on_result(idx, score, prediction, input_text, reference,
                      breakdown, error):
            studio_db.insert_eval_result(
                run_id=run_id, idx=idx, input_text=str(input_text),
                prediction_text=str(prediction),
                reference_text=reference if isinstance(reference, str)
                else json.dumps(reference),
                score=score,
                breakdown_json=json.dumps(breakdown) if breakdown else None,
                error=error,
            )
            running_total["sum"] += score or 0.0
            running_total["n"] += 1
            prog = self._progress[run_id]
            prog["done"] = running_total["n"]
            prog["avg_score"] = running_total["sum"] / running_total["n"]
            prog["last_result"] = {"idx": idx, "score": score, "error": error}

        status = "error"
        avg = 0.0
        err_msg = None
        try:
            summary = self._run_fn(
                req, on_result=on_result,
                should_cancel=lambda: self._cancel.is_set(),
            )
            status = summary.status
            avg = summary.avg_score
        except Exception as exc:
            err_msg = f"{type(exc).__name__}: {exc}"
        finally:
            studio_db.finish_eval_run(
                id=run_id, status=status, ended_at=_now(),
                avg_score=avg, error_message=err_msg,
            )
            prog = self._progress.get(run_id, {})
            prog["status"] = status
            prog["avg_score"] = avg
            with self._lock:
                self._active_run_id = None

    def cancel(self, run_id: str) -> bool:
        if self._active_run_id != run_id:
            return False
        self._cancel.set()
        return True

    def get(self, run_id: str) -> Optional[dict]:
        """Live progress if active, else the persisted run summary."""
        if run_id in self._progress and self._active_run_id == run_id:
            return dict(self._progress[run_id])
        run = studio_db.get_eval_run(run_id)
        if run is None:
            return self._progress.get(run_id)
        prog = self._progress.get(run_id, {})
        return {
            "run_id": run_id, "status": run["status"],
            "done": prog.get("done", 0), "total": prog.get("total", 0),
            "avg_score": run["avg_score"] if run["avg_score"] is not None
            else prog.get("avg_score", 0.0),
            "last_result": prog.get("last_result"),
        }


def build_eval_run_fn() -> RunFn:
    """Production run_fn: loads model + dataset, scores via the real inference path."""
    from core.inference import get_inference_backend
    from .dataset import DatasetRef, load_eval_examples
    from .inference_adapter import ensure_model_loaded, make_generate
    from .metrics.registry import make_scorer
    from .runner import run_eval

    def run_fn(req, *, on_result, should_cancel):
        backend = get_inference_backend()
        ensure_model_loaded(backend, req.model_identifier)
        ref = DatasetRef(
            is_local=req.dataset.is_local, path=req.dataset.path,
            name=req.dataset.name, split=req.dataset.split, subset=req.dataset.subset,
        )
        examples = load_eval_examples(
            ref, input_col=req.input_column,
            reference_col=req.reference_column, limit=req.limit,
        )
        generate = make_generate(backend, max_new_tokens=req.max_new_tokens,
                                 temperature=req.temperature)
        scorer = make_scorer(req.metric_name, req.metric_config)

        def _on_result(idx, result, prediction, input_text, reference):
            # bridge run_eval's 5-arg callback to the manager's 7-arg on_result
            on_result(idx, result.score, prediction, input_text, reference,
                      result.breakdown, result.error)

        return run_eval(
            examples=examples, generate=generate, scorer=scorer,
            system_prompt=req.system_prompt, template=req.template,
            gen_params={}, should_cancel=should_cancel, on_result=_on_result,
        )

    return run_fn
