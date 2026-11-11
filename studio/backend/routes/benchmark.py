# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Benchmark API routes for model benchmarking."""

from __future__ import annotations

import asyncio
import copy
import importlib.util
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Optional


from fastapi import APIRouter, Depends, HTTPException, Query, Request
from loggers import get_logger
from utils.utils import safe_error_detail

backend_path = Path(__file__).parent.parent.parent
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))

from auth.authentication import (
    authenticated_via_api_key,
    get_current_credential,
    get_current_subject,
)
from hub.utils.host_paths import host_paths_visible, redact_host_paths
from core.training.account_jobs import _has_managed_accounts
from utils.account_context import AccountContext, current_account
from utils.host_policy import LOOPBACK_FALLBACK_HOST, dial_host
from utils.paths.storage_roots import outputs_root

from models.benchmark import (
    BenchmarkRunRequest,
    BenchmarkRunDetail,
    BenchmarkOperationResponse,
)

from core.benchmark import (
    resolve_model_details,
    parse_run_summary,
    extract_samples,
    get_benchmark_backend,
)
from core.benchmark.recipe_heldout import RECIPE_HELDOUT_TASK_ID


# Only whether it is installed: importing lm_eval builds its task index, seconds at startup.
_lm_eval_available = importlib.util.find_spec("lm_eval") is not None

router = APIRouter()
logger = get_logger(__name__)


def _run_output_path(task: str) -> str:
    """The run's results folder; its name is also the run id, so two runs in one second need a suffix."""
    ts = time.strftime("%Y%m%d_%H%M%S")
    return str(outputs_root() / "evals" / f"benchmark_{task}_{ts}_{uuid.uuid4().hex[:6]}")


# lm_eval with the gguf backend is not on PyPI yet (EleutherAI/lm-evaluation-harness#4012).
_LM_EVAL_INSTALL_HINT = (
    "lm_eval is not installed. Install it into Studio's Python environment with: "
    "pip install 'lm_eval[hf,api] @ git+https://github.com/EleutherAI/lm-evaluation-harness.git"
    "@ad8737ae7fad24cf64e50fc7fc31397bff586b9e'"
)


def _without_api_keys(value: Any) -> Any:
    """lm_eval copies model_args (with the run's key) into its results; keep it off disk."""
    if isinstance(value, dict):
        return {
            k: ("<redacted>" if k == "api_key" else _without_api_keys(v)) for k, v in value.items()
        }
    if isinstance(value, list):
        return [_without_api_keys(v) for v in value]
    return value


def _redact_eval_run(run: dict, *, via_api_key: bool) -> dict:
    """``model`` and ``output_path`` are absolute host paths, hidden from API-key callers."""
    if host_paths_visible(via_api_key):
        return run
    ids = redact_host_paths({"active_model": run.get("model")}, via_api_key = via_api_key)
    out = dict(run)
    out["model"] = ids["active_model"]
    if out.get("output_path"):
        out["output_path"] = os.path.basename(str(out["output_path"]))
    return out


def _run_hidden_from(backend, viewer: AccountContext) -> bool:
    """True when the current/last eval belongs to another account (multi-account
    installs only): its log, outcome and Stop are then not this viewer's."""
    owner = getattr(backend, "_result_account", None)
    return (
        isinstance(owner, AccountContext)
        and owner.account_id != viewer.account_id
        and _has_managed_accounts()
    )


def _gguf_backend_supported() -> bool:
    """PyPI lm_eval (and unsloth[eval]) ships an older ``gguf`` model that
    StudioGGUFLM cannot drive; only the pinned commit has the server-slot API."""
    try:
        from lm_eval.models.gguf import GGUFLM
    except Exception:
        return False
    return hasattr(GGUFLM, "_detect_total_slots") and hasattr(GGUFLM, "_post_with_retries")


def _run_model_label(model: str) -> str:
    """A Hub repo holds every quant, so name the loaded one: two quants of a repo stay apart."""
    try:
        from core.inference.openai_auto_download import split_model_ref
        from hub.utils.paths import is_valid_repo_id
        from routes.inference import get_llama_cpp_backend

        base, variant = split_model_ref(model)
        if variant is not None or not is_valid_repo_id(base):
            return model  # already pinned, or a local file whose name carries the quant
        llama = get_llama_cpp_backend()
        loaded = llama.is_loaded and split_model_ref(llama.model_identifier or "")[0] == base
        if loaded and llama.hf_variant:
            return f"{base}:{llama.hf_variant}"
    except Exception as e:
        logger.debug("Could not read the loaded GGUF variant: %s", e)
    return model


def _studio_base_url(http_request: Request) -> Optional[str]:
    """The address this Studio actually listens on, for lm_eval's requests back to it."""
    from utils.studio_tls import loopback_http_port

    state = http_request.app.state
    # Under native TLS server_port speaks HTTPS; the plain loopback listener is the one to call.
    loopback = loopback_http_port(state)
    if loopback is not None:
        return f"http://127.0.0.1:{loopback}"
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
    credential: tuple = Depends(get_current_credential),
    via_api_key: bool = Depends(authenticated_via_api_key),
):
    """Run a benchmark on the specified model using lm_eval."""
    if request.task == RECIPE_HELDOUT_TASK_ID:
        from routes.benchmark_recipe import run_recipe_heldout
        return await run_recipe_heldout(request, http_request, credential, via_api_key)
    backend = get_benchmark_backend()

    api_key_id: int | None = None
    run_seq: int | None = None
    started = False

    try:
        if not _lm_eval_available or not _gguf_backend_supported():
            raise HTTPException(
                status_code = 400,
                detail = _LM_EVAL_INSTALL_HINT,
            )

        # Claimed before the first await: checking here and claiming later on
        # the executor thread let two concurrent requests both start a run.
        account = current_account()
        run_seq = backend.try_begin(account, request.run_token)
        if run_seq is None:
            raise HTTPException(
                status_code = 409,
                detail = "The stopped benchmark is still finishing. Try again in a moment."
                if backend.is_draining()
                else "A benchmark is already running.",
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
        output_path = _run_output_path(request.task)

        model_lines, lm_eval_kwargs = resolve_model_details(
            request.checkpoint_path,
            task = request.task,
            server_url = _studio_base_url(http_request),
            batch_size = request.batch_size,
            num_fewshot = request.num_fewshot,
            max_tokens = request.max_tokens,
            limit = request.limit,
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
            detail = (
                "; ".join(error_lines)
                if error_lines
                else (
                    "Could not resolve model path for benchmarking. "
                    "Ensure the model is downloaded and accessible."
                )
            )
            raise HTTPException(
                status_code = 400,
                detail = detail,
            )

        if lm_eval_kwargs.get("model") == "unsloth-studio-gguf":
            from datetime import datetime, timedelta, timezone
            from auth.storage import create_api_key

            expires_at = (datetime.now(timezone.utc) + timedelta(days = 3)).isoformat()
            raw_key, row = create_api_key(
                username = current_subject,
                name = "benchmark-run",
                expires_at = expires_at,
                internal = True,
            )
            api_key_id = row["id"]
            # StudioGGUFLM sends it as a bearer header on every request.
            lm_eval_kwargs["model_args"]["api_key"] = raw_key
            logger.info("Minted ephemeral API key for benchmark run")

        model_label = _run_model_label(request.checkpoint_path)
        # Checked under the scoring hold: the model the caller loaded is still the one loaded.
        lm_eval_kwargs["_expected_model"] = request.expected_model

        # ── Run lm_eval in the orchestrator worker thread ─────────
        backend.clear_logs()
        for line in model_lines:
            backend._append_log("stdout", line)

        loop = asyncio.get_event_loop()
        started = True
        results = await loop.run_in_executor(
            None, lambda: backend.run(lm_eval_kwargs, account, op_seq = run_seq)
        )

        # A cancelled run returns {}; by then a new run may already own the backend.
        if not results or backend.was_cancelled(run_seq):
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
                with open(results_path, "w", encoding = "utf-8") as f:
                    json.dump(_without_api_keys(results), f, indent = 2, default = str)
                backend._append_log("stdout", f"Results saved to {results_path}")
            except Exception as e:
                logger.warning(f"Failed to save benchmark results: {e}")

        stored_run_id = None
        try:
            from storage.studio_db import (
                insert_benchmark_run,
                insert_benchmark_samples,
            )
            run_summary = parse_run_summary(
                os.path.basename(output_path) if output_path else f"benchmark_{request.task}",
                results,
                task = request.task,
            )
            if run_summary is not None:
                insert_benchmark_run(
                    id = run_summary["id"],
                    task = run_summary["task"],
                    model = model_label
                    if run_summary["model"] == request.checkpoint_path
                    else run_summary["model"],
                    metrics = run_summary["metrics"],
                    n_samples = run_summary["n_samples"],
                    output_path = output_path or "",
                    duration_seconds = time.monotonic() - _run_start,
                    num_fewshot = run_summary.get("num_fewshot"),
                )
                samples = extract_samples(results, run_summary["task"])
                if samples:
                    insert_benchmark_samples(run_summary["id"], samples)
                stored_run_id = run_summary["id"]
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
                "run_id": stored_run_id,
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
        if run_seq is not None and not started:
            # Refused before lm_eval started (bad request, key minting): free the slot.
            backend.release(run_seq)
        elif run_seq is not None:
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


@router.post("/reserve", response_model = BenchmarkOperationResponse)
async def reserve_benchmark(
    token: str = Query(..., min_length = 8, max_length = 64),
    current_subject: str = Depends(get_current_subject),
):
    """Hold the run slot for the run ``token`` starts next (a Flows Benchmark loads its model first,
    which must not swap the model under a run in progress). Freed by ``/cancel?token=``."""
    backend = get_benchmark_backend()
    if not backend.reserve(current_account(), token):
        raise HTTPException(
            status_code = 409,
            detail = "The stopped benchmark is still finishing. Try again in a moment."
            if backend.is_draining()
            else "A benchmark is already running.",
        )
    return BenchmarkOperationResponse(success = True, message = "Benchmark slot reserved")


@router.post("/cancel", response_model = BenchmarkOperationResponse)
async def cancel_benchmark(
    token: Optional[str] = Query(None, max_length = 64),
    current_subject: str = Depends(get_current_subject),
):
    """Cancel the in-flight benchmark; with ``token``, only the run started with that token."""
    try:
        backend = get_benchmark_backend()
        if _run_hidden_from(backend, current_account()):
            return BenchmarkOperationResponse(
                success = True,
                message = "No active benchmark to cancel",
            )
        was_active = backend.cancel(token)
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


# ── Past eval runs ─────────────────────────────


@router.get("/runs/{run_id}", response_model = Optional[BenchmarkRunDetail])
async def get_benchmark_run_detail(
    run_id: str,
    current_subject: str = Depends(get_current_subject),
    via_api_key: bool = Depends(authenticated_via_api_key),
):
    """Get full detail for a single benchmark run, including per-sample results."""
    try:
        from storage.studio_db import get_benchmark_run_detail as db_get_detail

        detail = db_get_detail(run_id)
        if detail is None:
            raise HTTPException(status_code = 404, detail = "Benchmark run not found")
        return BenchmarkRunDetail(**_redact_eval_run(detail, via_api_key = via_api_key))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error loading benchmark run: {e}", exc_info = True)
        raise HTTPException(status_code = 500, detail = "Failed to load benchmark run")
