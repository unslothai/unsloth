# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Benchmark API routes for model benchmarking."""

import asyncio
import copy
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, AsyncGenerator, Optional

import yaml

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from loggers import get_logger
from utils.utils import safe_error_detail

backend_path = Path(__file__).parent.parent.parent
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))

from auth.authentication import get_current_subject
from core.training.account_jobs import _has_managed_accounts
from utils.account_context import AccountContext, current_account
from utils.host_policy import LOOPBACK_FALLBACK_HOST, dial_host
from utils.paths.storage_roots import outputs_root

from models.benchmark import (
    BenchmarkRunRequest,
    BenchmarkRunSummary,
    BenchmarkRunListResponse,
    BenchmarkRunDetail,
    BenchmarkStatusResponse,
    BenchmarkOperationResponse,
    BenchmarkTaskConfigResponse,
    BenchmarkTaskInfo,
    BenchmarkTasksResponse,
    BenchmarkGraphRequest,
    BenchmarkExportRequest,
)

from core.benchmark import (
    resolve_model_details,
    parse_run_summary,
    extract_samples,
    get_benchmark_backend,
)


# ── Curated benchmark tasks ──────────────────────────
_CURATED_TASKS: list[BenchmarkTaskInfo] = [
    BenchmarkTaskInfo(id = "mmlu",             name = "MMLU",             description = "Massive Multitask Language Understanding",             task_type = "log_likelihood"),
    BenchmarkTaskInfo(id = "mmlu_pro",          name = "MMLU-Pro",         description = "MMLU-Pro (more challenging MMLU variant)",               task_type = "log_likelihood"),
    BenchmarkTaskInfo(id = "gpqa_main_cot_zeroshot", name = "GPQA",        description = "Google Proof Q&A (PhD-level science, CoT zero-shot)",     task_type = "generation"),
    BenchmarkTaskInfo(id = "hellaswag",         name = "HellaSwag",        description = "Commonsense reasoning (sentence completion)",            task_type = "log_likelihood"),
    BenchmarkTaskInfo(id = "arc_challenge",     name = "ARC-Challenge",    description = "AI2 Reasoning Challenge (challenge split)",              task_type = "log_likelihood"),
    BenchmarkTaskInfo(id = "winogrande",        name = "WinoGrande",       description = "Winograd schema coreference",                            task_type = "log_likelihood"),
    BenchmarkTaskInfo(id = "gsm8k",             name = "GSM8K",            description = "Grade-school math (5-shot, chain-of-thought)",            task_type = "generation"),
    BenchmarkTaskInfo(id = "hendrycks_math",    name = "MATH",             description = "Hendrycks MATH (mathematical reasoning)",                task_type = "generation"),
    BenchmarkTaskInfo(id = "ifeval",            name = "IFEval",           description = "Instruction Following Evaluation",                       task_type = "generation"),
    BenchmarkTaskInfo(id = "humaneval",         name = "HumanEval",        description = "HumanEval (code generation)",                            task_type = "generation"),
    BenchmarkTaskInfo(id = "truthfulqa_mc1",    name = "TruthfulQA MC1",  description = "Truthfulness & safety (single-true)",                    task_type = "log_likelihood"),
    BenchmarkTaskInfo(id = "longbench",          name = "LongBench",        description = "Long-context understanding (multi-task)",                 task_type = "generation"),
    BenchmarkTaskInfo(id = "bbh",               name = "BBH",              description = "Big Bench Hard (chain-of-thought few-shot)",              task_type = "log_likelihood"),
    BenchmarkTaskInfo(id = "bbh_fewshot",       name = "BBH (few-shot)",   description = "Big Bench Hard (few-shot, no CoT)",                       task_type = "log_likelihood"),
    BenchmarkTaskInfo(id = "bbh_zeroshot",      name = "BBH (zero-shot)",  description = "Big Bench Hard (zero-shot, no CoT)",                      task_type = "log_likelihood"),
]

_task_cache: list[BenchmarkTaskInfo] | None = None


def _build_task_cache() -> list[BenchmarkTaskInfo]:
    curated_map = {t.id: t for t in _CURATED_TASKS}
    try:
        if _lm_eval_available:
            logger.info("lm_eval available — building task index...")
            all_names = _get_task_manager().all_tasks
            logger.info(f"Loaded {len(all_names)} tasks from lm_eval: first 5 = {all_names[:5]}")
        else:
            logger.warning("lm_eval not available — using curated tasks only")
            all_names = list(curated_map.keys())
    except Exception as e:
        logger.warning(f"Failed to load lm_eval tasks: {e}")
        all_names = list(curated_map.keys())

    seen: set[str] = set()
    for name in all_names:
        if name not in curated_map:
            parent = None
            for t in _CURATED_TASKS:
                if name.startswith(t.id + "_"):
                    parent = t
                    break
            if parent is not None:
                curated_map[name] = BenchmarkTaskInfo(
                    id = name, name = name,
                    description = parent.description,
                    task_type = parent.task_type,
                )
            else:
                curated_map[name] = BenchmarkTaskInfo(id = name, name = name)
        seen.add(name)
    for t in _CURATED_TASKS:
        if t.id not in seen:
            all_names.append(t.id)
    return [curated_map[n] for n in all_names]


def _get_task_cache() -> list[BenchmarkTaskInfo]:
    global _task_cache
    if _task_cache is None:
        try:
            logger.info("Building task cache...")
            _task_cache = _build_task_cache()
            logger.info(f"Task cache built: {len(_task_cache)} tasks")
        except Exception as e:
            logger.error(f"Task cache build failed: {e}", exc_info = True)
            _task_cache = [t for t in _CURATED_TASKS]
    return _task_cache


_task_mgr_cache = None

def _get_task_manager():
    global _task_mgr_cache
    if _task_mgr_cache is None:
        _task_mgr_cache = _lm_eval_tasks.TaskManager()
    return _task_mgr_cache


try:
    import lm_eval.tasks as _lm_eval_tasks
    _lm_eval_available = True
except ImportError:
    _lm_eval_available = False

router = APIRouter()
logger = get_logger(__name__)

# Warm the task cache in the background at startup so the first /tasks
# request doesn't pay the cost of building the full lm_eval task index.
threading.Thread(target = _get_task_cache, daemon = True).start()


@router.get("/tasks", response_model = BenchmarkTasksResponse)
async def list_benchmark_tasks(current_subject: str = Depends(get_current_subject)):
    """List available benchmark tasks — curated shortlist merged with all lm_eval tasks."""
    try:
        tasks = _get_task_cache()
        logger.info(f"Returning {len(tasks)} tasks")
        return BenchmarkTasksResponse(tasks = tasks)
    except Exception as e:
        logger.error(f"Error listing benchmark tasks: {e}", exc_info = True)
        raise HTTPException(
            status_code = 500,
            detail = "Failed to list benchmark tasks",
        )


@router.get("/task/{task_id}/config", response_model = BenchmarkTaskConfigResponse)
async def get_task_config(
    task_id: str,
    current_subject: str = Depends(get_current_subject),
):
    """Load a single task's YAML config and return its metadata (num_fewshot, etc.)."""
    if not _lm_eval_available:
        return BenchmarkTaskConfigResponse(task_id = task_id, num_fewshot = None)

    try:
        task_mgr = _get_task_manager()
        idx = task_mgr.task_index.get(task_id)
        if idx is None:
            return BenchmarkTaskConfigResponse(task_id = task_id, num_fewshot = None)

        # task_index entries may be dicts or Entry objects — handle both
        yaml_path = idx.get("yaml_path") if hasattr(idx, "get") else getattr(idx, "yaml_path", None)
        if not yaml_path or yaml_path == -1:
            return BenchmarkTaskConfigResponse(task_id = task_id, num_fewshot = None)

        # Read the raw YAML — num_fewshot is null if absent (user can customize)
        # or 0 if explicitly disabled (slider hidden).
        # lm_eval YAMLs use !function tags; register a no-op constructor for them.
        yaml.add_constructor("!function", lambda loader, node: None, Loader = yaml.FullLoader)
        with open(yaml_path) as f:
            cfg = yaml.full_load(f)
        if not isinstance(cfg, dict):
            return BenchmarkTaskConfigResponse(task_id = task_id, num_fewshot = None)
        nf = cfg.get("num_fewshot")
        # lm_eval treats null as "no default" (user can still set it),
        # and 0 as "explicitly disabled"
        if nf is None:
            num_fewshot: int | None = None
        else:
            num_fewshot = int(nf)
        return BenchmarkTaskConfigResponse(task_id = task_id, num_fewshot = num_fewshot)
    except Exception as e:
        logger.warning(f"Failed to load config for task '{task_id}': {e}")
        return BenchmarkTaskConfigResponse(task_id = task_id, num_fewshot = None)

# lm_eval with the gguf backend is not on PyPI yet (EleutherAI/lm-evaluation-harness#4012).
_LM_EVAL_INSTALL_HINT = (
    "lm_eval is not installed. Install it into Studio's Python environment with: "
    "pip install 'lm_eval[hf,api] @ git+https://github.com/EleutherAI/lm-evaluation-harness.git"
    "@ad8737ae7fad24cf64e50fc7fc31397bff586b9e'"
)


def _run_hidden_from(backend, viewer: AccountContext) -> bool:
    """True when the current/last eval belongs to another account (multi-account
    installs only): its log, outcome and Stop are then not this viewer's."""
    owner = getattr(backend, "_result_account", None)
    return (
        isinstance(owner, AccountContext)
        and owner.account_id != viewer.account_id
        and _has_managed_accounts()
    )


def _studio_base_url(http_request: Request) -> Optional[str]:
    """The address this Studio actually listens on, for lm_eval's requests back to it."""
    state = http_request.app.state
    port = getattr(state, "server_port", None)
    if not isinstance(port, int) or port <= 0:
        return None
    host = getattr(state, "server_request_host", None)
    if not isinstance(host, str) or not host:
        host = LOOPBACK_FALLBACK_HOST
    return f"http://{dial_host(host)}:{port}"


@router.post("/run", response_model = BenchmarkOperationResponse)
async def run_benchmark(
    request: BenchmarkRunRequest,
    http_request: Request,
    current_subject: str = Depends(get_current_subject),
):
    """Run a benchmark on the specified model using lm_eval."""
    backend = get_benchmark_backend()

    api_key_raw: str | None = None
    api_key_id: int | None = None
    run_seq: int | None = None

    try:
        if backend.is_active():
            raise HTTPException(
                status_code = 409,
                detail = "A benchmark is already running.",
            )

        if not _lm_eval_available:
            raise HTTPException(
                status_code = 400,
                detail = _LM_EVAL_INSTALL_HINT,
            )

        logger.info("=== Benchmark Run Request =========================")
        logger.info(f"checkpoint_path: {request.checkpoint_path}")
        logger.info(f"model_source:    {request.model_source}")
        logger.info(f"hf_token:        {'<provided>' if request.hf_token else '<not provided>'}")
        logger.info(f"hf_token length: {len(request.hf_token) if request.hf_token else 0}")
        logger.info(f"request model_dump: {request.model_dump(exclude = {'hf_token'})}")
        logger.info("===================================================")

        _run_start = time.monotonic()

        # Results are always written to an auto-managed directory; there is no
        # user-supplied output path.
        ts = time.strftime("%Y%m%d_%H%M%S")
        output_path = str(outputs_root() / "evals" / f"benchmark_{request.task}_{ts}")

        model_lines, lm_eval_kwargs = resolve_model_details(
            request.checkpoint_path,
            task = request.task,
            server_url = _studio_base_url(http_request),
            batch_size = request.batch_size,
            num_fewshot = request.num_fewshot,
            max_tokens = request.max_tokens,
        )
        log_kwargs = copy.deepcopy(lm_eval_kwargs)
        log_model_args = log_kwargs.get("model_args", {})
        if "auth_token" in log_model_args:
            log_model_args["auth_token"] = f"<redacted ({len(log_model_args['auth_token'])} chars)>"
        kwargs_json = json.dumps(log_kwargs, indent = 2, default = str)
        logger.info(f"lm_eval.simple_evaluate kwargs:\n{kwargs_json}")
        model_lines.append(f"lm_eval kwargs: {kwargs_json}")

        if lm_eval_kwargs is None:
            error_lines = [l for l in model_lines if l.startswith("ERROR:")]
            detail = "; ".join(error_lines) if error_lines else (
                "Could not resolve model path for benchmarking. "
                "Ensure the model is downloaded and accessible."
            )
            raise HTTPException(
                status_code = 400,
                detail = detail,
            )

        if lm_eval_kwargs.get("model") == "unsloth-studio-gguf":
            from datetime import datetime, timedelta, timezone
            from auth.storage import create_api_key

            expires_at = (datetime.now(timezone.utc) + timedelta(days=3)).isoformat()
            raw_key, row = create_api_key(
                username = current_subject,
                name = "benchmark-run",
                expires_at = expires_at,
                internal = True,
            )
            api_key_raw = raw_key
            api_key_id = row["id"]
            # StudioGGUFLM sends it as a bearer header on every request.
            lm_eval_kwargs["model_args"]["api_key"] = raw_key
            logger.info("Minted ephemeral API key for benchmark run")

        # ── Run lm_eval in the orchestrator worker thread ─────────
        backend.clear_logs()
        for line in model_lines:
            backend._append_log("stdout", line)

        loop = asyncio.get_event_loop()
        run_seq = backend.get_op_seq() + 1
        account = current_account()
        results = await loop.run_in_executor(None, lambda: backend.run(lm_eval_kwargs, account))

        # A cancelled run returns {}; by then a new run may already own the backend.
        if not results or backend.was_cancelled():
            backend._append_log("status", "Benchmark cancelled.")
            return BenchmarkOperationResponse(
                success = False,
                message = "Benchmark cancelled",
                details = {"checkpoint_path": request.checkpoint_path, "cancelled": True},
            )

        # ── Post-run: save to disk + DB ───────────────────────────
        if output_path:
            try:
                os.makedirs(output_path, exist_ok = True)
                results_path = os.path.join(output_path, "results.json")
                with open(results_path, "w") as f:
                    json.dump(results, f, indent = 2, default = str)
                backend._append_log("stdout", f"Results saved to {results_path}")
            except Exception as e:
                logger.warning(f"Failed to save benchmark results: {e}")

        try:
            from storage.studio_db import (
                insert_benchmark_run,
                insert_benchmark_samples,
            )
            run_summary = parse_run_summary(
                os.path.basename(output_path) if output_path else f"benchmark_{request.task}",
                results,
            )
            if run_summary is not None:
                insert_benchmark_run(
                    id = run_summary["id"],
                    task = run_summary["task"],
                    model = run_summary["model"],
                    metrics = run_summary["metrics"],
                    n_samples = run_summary["n_samples"],
                    output_path = output_path or "",
                    duration_seconds = time.monotonic() - _run_start,
                    num_fewshot = run_summary.get("num_fewshot"),
                )
                samples = extract_samples(results, run_summary["task"])
                if samples:
                    insert_benchmark_samples(run_summary["id"], samples)
                backend._append_log("stdout", "Results stored in database")
        except Exception as e:
            logger.warning(f"Failed to store benchmark results in DB: {e}")

        task_results = results.get("results", {})
        backend._append_log("stdout", "Results:")
        for task_name, metrics in task_results.items():
            acc = metrics.get("acc,none", metrics.get("acc", "N/A"))
            stderr = metrics.get("acc_stderr,none", None)
            line = f"  {task_name}: acc={acc}" + (f" ± {stderr}" if stderr else "")
            backend._append_log("stdout", line)
        backend._append_log("status", "Benchmark complete.")
        backend.finish(op_seq = run_seq)

        return BenchmarkOperationResponse(
            success = True,
            message = f"Benchmark completed for {request.checkpoint_path}",
            details = {
                "checkpoint_path": request.checkpoint_path,
                "results": task_results,
            },
        )
    except HTTPException:
        raise
    except Exception as e:
        backend._append_log("stderr", f"Error: {safe_error_detail(e)}")
        logger.error(f"Error running benchmark: {e}", exc_info = True)
        raise HTTPException(
            status_code = 500,
            detail = f"Failed to run benchmark: {safe_error_detail(e)}",
        )
    finally:
        if run_seq is not None:
            # No-op unless this request's run is still the active one, i.e.
            # lm_eval returned but handling its results raised.
            backend.finish(op_seq = run_seq, error = "Failed to store benchmark results")
        if api_key_id is not None:
            try:
                from auth.storage import revoke_internal_api_key
                revoke_internal_api_key(api_key_id)
                logger.info("Revoked ephemeral benchmark API key")
            except Exception as e:
                logger.warning("Failed to revoke benchmark API key: %s", e)


@router.post("/cancel", response_model = BenchmarkOperationResponse)
async def cancel_benchmark(current_subject: str = Depends(get_current_subject)):
    """Cancel the in-flight benchmark."""
    try:
        backend = get_benchmark_backend()
        if _run_hidden_from(backend, current_account()):
            return BenchmarkOperationResponse(
                success = True,
                message = "No active benchmark to cancel",
            )
        was_active = backend.cancel()
        return BenchmarkOperationResponse(
            success = True,
            message = "Benchmark cancelled" if was_active else "No active benchmark to cancel",
        )
    except Exception as e:
        logger.error(f"Error cancelling benchmark: {e}", exc_info = True)
        raise HTTPException(
            status_code = 500,
            detail = "Failed to cancel benchmark",
        )


@router.get("/status", response_model = BenchmarkStatusResponse)
async def get_benchmark_status(current_subject: str = Depends(get_current_subject)):
    """Get benchmark backend status."""
    try:
        backend = get_benchmark_backend()
        if _run_hidden_from(backend, current_account()):
            # Busy-ness stays visible so Run reports the 409, nothing else does.
            return BenchmarkStatusResponse(is_benchmark_active = backend.is_active())
        return BenchmarkStatusResponse(
            is_benchmark_active = backend.is_active(),
            last_op_status = backend.get_last_op_status(),
            last_op_error = backend.get_last_error(),
            last_op_seq = backend.get_op_seq(),
        )
    except Exception as e:
        logger.error(f"Error getting benchmark status: {e}", exc_info = True)
        raise HTTPException(
            status_code = 500,
            detail = "Failed to get benchmark status",
        )


@router.get("/logs/stream")
async def stream_benchmark_logs(
    request: Request,
    since: Optional[int] = Query(
        None,
        description = "Return log entries with seq strictly greater than this cursor.",
    ),
    current_subject: str = Depends(get_current_subject),
):
    """
    Stream benchmark logs as Server-Sent Events.

    This single long-lived stream replaces the old `/logs` poll + short-lived
    `/logs/stream` pair: it replays everything from the start of the current run,
    then pushes each new log line as it arrives, keeping the connection open
    (with periodic heartbeats) until the client disconnects. Clients resume on
    reconnect via `?since=` or the `Last-Event-ID` header.

    Events:
      - `log`      : a single log line (data: {"stream","line","ts"})
      - `heartbeat`: periodic keepalive carrying {"active": bool} when idle

    Each event's `id:` is the log entry's monotonic seq, used for resume.
    """
    backend = get_benchmark_backend()
    viewer = current_account()

    # Resume cursor: explicit `since` wins, then Last-Event-ID on reconnect.
    last_event_id = request.headers.get("last-event-id")
    if since is None and last_event_id is not None:
        try:
            since = int(last_event_id)
        except ValueError:
            pass

    if since is not None:
        cursor = max(0, int(since))
    elif backend.is_active():
        # First connect mid-run: replay the whole current run.
        cursor = backend.get_run_start_seq()
    else:
        # First connect while idle — e.g. the stream opens just before the
        # `/run` POST lands — so don't replay stale logs from an earlier run;
        # wait for the live ones instead.
        cursor = backend.get_current_log_seq()

    async def event_generator() -> AsyncGenerator[str, None]:
        nonlocal cursor
        # Reconnect after 3 seconds if the connection drops.
        yield "retry: 3000\n\n"

        last_yield = time.monotonic()
        last_progress = None
        try:
            while True:
                if await request.is_disconnected():
                    return

                entries, new_cursor = backend.get_logs_since(cursor)
                if entries and _run_hidden_from(backend, viewer):
                    cursor = new_cursor
                    entries = []
                if entries:
                    for entry in entries:
                        payload = json.dumps(
                            {
                                "stream": entry.get("stream", "stdout"),
                                "line": entry.get("line", ""),
                                "ts": entry.get("ts"),
                            }
                        )
                        yield _format_sse(
                            payload,
                            event = "log",
                            event_id = int(entry.get("seq", 0)),
                        )
                    cursor = new_cursor
                    last_yield = time.monotonic()
                else:
                    now = time.monotonic()
                    if now - last_yield > 10.0:
                        yield _format_sse(
                            json.dumps({"active": backend.is_active()}),
                            event = "heartbeat",
                            event_id = cursor,
                        )
                        last_yield = now

                # Surface lm_eval tqdm progress as a dedicated SSE event.
                progress = backend.get_latest_progress()
                if (
                    progress is not None
                    and progress != last_progress
                    and not _run_hidden_from(backend, viewer)
                ):
                    last_progress = progress
                    yield _format_sse(json.dumps(progress), event = "progress")

                await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            # Client disconnected mid-yield: end cleanly so StreamingResponse finalizes.
            return
        except Exception as exc:
            logger.error("Benchmark log stream failed: %s", exc, exc_info = True)
            try:
                yield _format_sse(
                    json.dumps({"error": safe_error_detail(exc)}),
                    event = "error",
                )
            except Exception:
                pass

    return StreamingResponse(
        event_generator(),
        media_type = "text/event-stream",
        headers = {
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ── Past eval runs ─────────────────────────────

@router.get("/runs", response_model = BenchmarkRunListResponse)
async def list_benchmark_runs(current_subject: str = Depends(get_current_subject)):
    """List past benchmark runs, newest first."""
    try:
        from storage.studio_db import list_benchmark_runs as db_list_runs
        runs = db_list_runs()
        return BenchmarkRunListResponse(runs = [BenchmarkRunSummary(**r) for r in runs])
    except Exception as e:
        logger.error(f"Error listing benchmark runs: {e}", exc_info = True)
        raise HTTPException(status_code = 500, detail = "Failed to list benchmark runs")


@router.get("/runs/{run_id}", response_model = Optional[BenchmarkRunDetail])
async def get_benchmark_run_detail(
    run_id: str,
    current_subject: str = Depends(get_current_subject),
):
    """Get full detail for a single benchmark run, including per-sample results."""
    try:
        from storage.studio_db import get_benchmark_run_detail as db_get_detail
        detail = db_get_detail(run_id)
        if detail is None:
            raise HTTPException(status_code = 404, detail = "Benchmark run not found")
        return BenchmarkRunDetail(**detail)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error loading benchmark run: {e}", exc_info = True)
        raise HTTPException(status_code = 500, detail = "Failed to load benchmark run")


@router.delete("/runs/{run_id}", response_model = BenchmarkOperationResponse)
async def delete_benchmark_run(
    run_id: str,
    current_subject: str = Depends(get_current_subject),
):
    """Delete a benchmark run and its per-sample data."""
    import shutil

    try:
        from storage.studio_db import get_benchmark_run as db_get_run
        from storage.studio_db import delete_benchmark_run as db_delete_run

        run = db_get_run(run_id)
        if run is None:
            raise HTTPException(status_code = 404, detail = "Benchmark run not found")

        # Remove results.json directory if it exists
        output_path = run.get("output_path")
        if output_path and os.path.isdir(output_path):
            try:
                shutil.rmtree(output_path)
            except Exception as e:
                logger.warning(f"Failed to delete output directory {output_path}: {e}")

        db_delete_run(run_id)
        return BenchmarkOperationResponse(
            success = True,
            message = f"Deleted benchmark run {run_id}",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting benchmark run: {e}", exc_info = True)
        raise HTTPException(status_code = 500, detail = "Failed to delete benchmark run")


def _format_sse(
    data: str,
    event: str,
    event_id: Optional[int] = None,
) -> str:
    lines = []
    if event_id is not None:
        lines.append(f"id: {event_id}")
    lines.append(f"event: {event}")
    lines.append(f"data: {data}")
    lines.append("")
    lines.append("")
    return "\n".join(lines)


@router.post("/graph")
async def generate_benchmark_graph(
    request: BenchmarkGraphRequest,
    current_subject: str = Depends(get_current_subject),
):
    """Generate a benchmark comparison chart as a PNG image."""
    from fastapi.responses import Response
    from storage.studio_db import get_benchmark_run_detail as db_get_run_detail

    try:
        if len(request.run_ids) < 1:
            raise HTTPException(
                status_code = 400,
                detail = "At least one run ID is required.",
            )

        if request.chart_type not in ("bar", "line", "grouped_bar", "radar"):
            raise HTTPException(
                status_code = 400,
                detail = f"Invalid chart type: {request.chart_type}. Must be 'bar', 'line', 'grouped_bar', or 'radar'.",
            )

        _VALID_THEMES = {
            "light", "dark", "unsloth-dark", "ggplot", "fivethirtyeight",
            "bmh", "grayscale", "seaborn-v0_8", "seaborn-v0_8-darkgrid",
        }
        if request.theme not in _VALID_THEMES:
            import matplotlib.style
            if request.theme not in matplotlib.style.available:
                raise HTTPException(
                    status_code = 400,
                    detail = f"Invalid theme: '{request.theme}'. Not a built-in theme or a known matplotlib style.",
                )

        runs = []
        for run_id in request.run_ids:
            run = db_get_run_detail(run_id)
            if run is None:
                raise HTTPException(
                    status_code = 404,
                    detail = f"Benchmark run not found: {run_id}",
                )
            runs.append(run)

        from core.benchmark.graph import generate_chart

        png_bytes = await asyncio.to_thread(
            generate_chart,
            runs = runs,
            chart_type = request.chart_type,
            metric = request.metric,
            width = request.width,
            height = request.height,
            theme = request.theme,
        )

        return Response(
            content = png_bytes,
            media_type = "image/png",
            headers = {
                "Content-Disposition": 'attachment; filename="benchmark-chart.png"',
            },
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating benchmark graph: {e}", exc_info = True)
        raise HTTPException(
            status_code = 500,
            detail = f"Failed to generate chart: {safe_error_detail(e)}",
        )


@router.post("/export")
async def export_benchmark_runs(
    request: BenchmarkExportRequest,
    current_subject: str = Depends(get_current_subject),
):
    """Export benchmark runs as JSON with metadata and results.json contents."""
    from fastapi.responses import Response
    from storage.studio_db import get_benchmark_run_detail as db_get_run_detail

    try:
        if len(request.run_ids) < 1:
            raise HTTPException(
                status_code = 400,
                detail = "At least one run ID is required.",
            )

        exported = []
        for run_id in request.run_ids:
            run = db_get_run_detail(run_id)
            if run is None:
                raise HTTPException(
                    status_code = 404,
                    detail = f"Benchmark run not found: {run_id}",
                )

            output_path = run.get("output_path", "")
            results_json = None
            if output_path:
                results_path = os.path.join(output_path, "results.json")
                if os.path.isfile(results_path):
                    try:
                        with open(results_path, "r") as f:
                            results_json = json.load(f)
                    except Exception as e:
                        logger.warning(f"Failed to read results.json for {run_id}: {e}")

            entry = {
                "id": run.get("id"),
                "task": run.get("task"),
                "model": run.get("model"),
                "metrics": run.get("metrics", []),
                "n_samples": run.get("n_samples", 0),
                "num_fewshot": run.get("num_fewshot"),
                "created_at": run.get("created_at"),
                "output_path": output_path,
                "correct_count": run.get("correct_count", 0),
                "total_count": run.get("total_count", 0),
                "samples": run.get("samples", []),
                "results": results_json,
            }
            exported.append(entry)

        payload = json.dumps(exported, indent = 2, default = str)
        return Response(
            content = payload,
            media_type = "application/json",
            headers = {
                "Content-Disposition": 'attachment; filename="benchmark-export.json"',
            },
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error exporting benchmark runs: {e}", exc_info = True)
        raise HTTPException(
            status_code = 500,
            detail = f"Failed to export runs: {safe_error_detail(e)}",
        )
