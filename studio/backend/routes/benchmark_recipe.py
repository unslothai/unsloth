# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The Recipe held-out task's branch of ``POST /api/benchmarks/evals/run``: same run slot,
logs, progress and run storage as the lm-eval tasks, its own scorer."""

from __future__ import annotations

import asyncio
import copy
import time
from typing import Any, Optional

import httpx
from fastapi import HTTPException, Request

from auth.storage import CredentialRotated
from core.benchmark import get_benchmark_backend
from core.benchmark.orchestrator import _report_progress
from core.benchmark.recipe_heldout import (
    RECIPE_HELDOUT_TASK_ID,
    eval_metrics,
    post_chat,
    resolve_chat_target,
    run_benchmark,
)
from core.data_recipe.service import require_ui_session_for_host_env_credentials
from core.training.account_jobs import managed_account
from loggers import get_logger
from models.benchmark import BenchmarkOperationResponse, BenchmarkRunRequest
from utils.account_context import current_account
from utils.utils import log_and_http_error, safe_curated_detail

from routes.data_recipe.jobs import (
    _inject_local_providers,
    _loaded_local_model_identity,
    _revoke_internal_api_key_safe,
)

logger = get_logger(__name__)


def _recipe_task(params: dict) -> dict:
    targets = params["targets"]
    with httpx.Client() as client:
        return run_benchmark(
            params["spec"],
            chat = lambda target, messages: post_chat(client, targets[target], messages),
            on_progress = lambda done, total: _report_progress(done, total, "Recipe held-out"),
            is_cancelled = params["is_cancelled"],
        )


def _model_label(alias: str, configs: dict[str, dict], local_names: set) -> str:
    mc = configs.get(alias) or {}
    model = str(mc.get("model") or "").strip()
    if mc.get("provider") in local_names and (not model or model.lower() == "local"):
        model = _loaded_local_model_identity()[1]
    return model or alias


def _store(run_id: str, model: str, summary: dict, samples: list[dict], duration: float) -> None:
    from storage.studio_db import insert_benchmark_run, insert_benchmark_samples

    insert_benchmark_run(
        id = run_id,
        task = RECIPE_HELDOUT_TASK_ID,
        model = model,
        metrics = eval_metrics(summary),
        n_samples = summary["scored"],
        output_path = "",
        duration_seconds = duration,
    )
    insert_benchmark_samples(run_id, samples)


async def run_recipe_heldout(
    request: BenchmarkRunRequest,
    http_request: Request,
    credential: tuple,
    via_api_key: bool,
) -> BenchmarkOperationResponse:
    spec = request.recipe
    if spec is None:
        raise HTTPException(
            status_code = 400,
            detail = "Recipe held-out runs from a Benchmark node in Flows.",
        )
    rows = spec.rows[: spec.max_rows]
    if not rows:
        raise HTTPException(status_code = 400, detail = "No held-out rows to score.")
    if spec.scorer == "judge" and not spec.judge_alias:
        raise HTTPException(status_code = 400, detail = "Pick a judge model for LLM judge scoring.")
    if spec.baseline.mode == "alias" and not spec.baseline.alias:
        raise HTTPException(status_code = 400, detail = "Pick a baseline model.")

    aliases = [spec.model_alias]
    if spec.baseline.mode == "alias":
        aliases.append(spec.baseline.alias)
    if spec.scorer == "judge":
        aliases.append(spec.judge_alias)
    recipe = {
        "model_providers": copy.deepcopy(spec.model_providers),
        "model_configs": copy.deepcopy(spec.model_configs),
        "columns": [
            {"name": f"benchmark_{i}", "column_type": "llm-text", "model_alias": alias}
            for i, alias in enumerate(aliases)
        ],
    }
    require_ui_session_for_host_env_credentials(recipe, via_api_key)
    local_names = {
        p.get("name")
        for p in recipe["model_providers"]
        if isinstance(p, dict) and p.get("is_local") is True
    }
    configs = {mc.get("alias"): mc for mc in recipe["model_configs"] if isinstance(mc, dict)}
    if managed_account() and any(
        configs.get(alias, {}).get("provider") not in local_names for alias in aliases
    ):
        raise HTTPException(
            status_code = 403,
            detail = "Managed accounts can only benchmark with the local model.",
        )

    backend = get_benchmark_backend()
    account = current_account()
    run_seq = backend.try_begin(account)
    if run_seq is None:
        raise HTTPException(status_code = 409, detail = "A benchmark is already running.")
    started = False
    key_id: Optional[int] = None
    try:
        try:
            key_id = _inject_local_providers(recipe, http_request, credential[1])
        except CredentialRotated as exc:
            raise HTTPException(status_code = 401, detail = "Invalid or expired token") from exc
        except ValueError as exc:
            raise log_and_http_error(
                exc,
                400,
                safe_curated_detail(exc),
                event = "benchmark.recipe_heldout.inject_local_providers_failed",
                log = logger,
            ) from exc
        providers = {p.get("name"): p for p in recipe["model_providers"] if isinstance(p, dict)}
        try:
            candidate = resolve_chat_target(spec.model_alias, configs, providers)
            targets: dict[str, Any] = {"candidate": candidate}
            if spec.baseline.mode == "adapter_off":
                # One loaded model serves both sides: Studio's /v1 toggles the LoRA per request.
                candidate["body"]["use_adapter"] = True
                targets["baseline"] = copy.deepcopy(candidate)
                targets["baseline"]["body"]["use_adapter"] = False
            elif spec.baseline.mode == "alias":
                targets["baseline"] = resolve_chat_target(spec.baseline.alias, configs, providers)
            if spec.scorer == "judge":
                targets["judge"] = resolve_chat_target(spec.judge_alias, configs, providers)
                judge_provider = configs[spec.judge_alias].get("provider")
                if (spec.baseline.mode == "adapter_off" or spec.local_adapter) and judge_provider in local_names:
                    # The local model can judge, but never with the adapter it is grading.
                    targets["judge"]["body"]["use_adapter"] = False
        except ValueError as exc:
            raise HTTPException(status_code = 400, detail = str(exc)) from exc

        candidate_model = _model_label(spec.model_alias, configs, local_names)
        if spec.baseline.mode == "adapter_off":
            baseline_model = f"{candidate_model} (adapter off)"
        elif spec.baseline.mode == "alias":
            baseline_model = _model_label(spec.baseline.alias, configs, local_names)
        else:
            baseline_model = ""
        params = {
            "spec": {
                "rows": rows,
                "prompt_column": spec.prompt_column,
                "reference_column": spec.reference_column,
                "system_prompt": spec.system_prompt,
                "scorer": spec.scorer,
                "rubric": spec.rubric,
                "with_baseline": "baseline" in targets,
            },
            "targets": targets,
            "is_cancelled": backend.was_cancelled,
        }
        backend.clear_logs()
        backend._append_log(
            "stdout",
            f"Recipe held-out: {len(rows)} rows, {spec.scorer} scorer, model {candidate_model}",
        )
        started = True
        t0 = time.monotonic()
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: backend.run(params, account, op_seq = run_seq, task_fn = _recipe_task),
        )
        if not result or backend.was_cancelled():
            backend._append_log("status", "Benchmark cancelled.")
            return BenchmarkOperationResponse(
                success = False,
                message = "Benchmark cancelled",
                details = {"cancelled": True},
            )

        duration = time.monotonic() - t0
        run_id = f"benchmark_{RECIPE_HELDOUT_TASK_ID}_{time.strftime('%Y%m%d_%H%M%S')}"
        run_ids = {"candidate": run_id}
        _store(run_id, candidate_model, result["candidate"], result["samples"]["candidate"], duration)
        if "baseline" in result:
            run_ids["baseline"] = f"{run_id}_baseline"
            _store(
                run_ids["baseline"],
                baseline_model,
                result["baseline"],
                result["samples"]["baseline"],
                duration,
            )
        score = result["candidate"]["score"]
        backend._append_log(
            "stdout",
            f"Results: {RECIPE_HELDOUT_TASK_ID}: acc={score}"
            + (f", baseline acc={result['baseline']['score']}" if "baseline" in result else ""),
        )
        backend._append_log("status", "Benchmark complete.")
        backend.finish(op_seq = run_seq)
        details = {k: v for k, v in result.items() if k != "samples"}
        details["run_ids"] = run_ids
        return BenchmarkOperationResponse(
            success = True,
            message = f"Recipe held-out scored {result['candidate']['scored']} rows",
            details = {"recipe": details},
        )
    except HTTPException:
        raise
    except Exception as exc:
        backend._append_log("stderr", f"Error: {safe_curated_detail(exc)}")
        raise log_and_http_error(
            exc,
            500,
            "Recipe held-out benchmark failed",
            event = "benchmark.recipe_heldout.failed",
            log = logger,
        ) from exc
    finally:
        if not started:
            backend.release(run_seq)
        else:
            backend.finish(op_seq = run_seq, error = "Failed to store benchmark results")
        if key_id is not None:
            _revoke_internal_api_key_safe(key_id)
