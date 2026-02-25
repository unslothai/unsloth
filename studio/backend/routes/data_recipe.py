"""
Data Recipe routes (DataDesigner runner).
"""

from __future__ import annotations

import base64
import binascii
import sys
from itertools import islice
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse

# same thing as other files do
backend_path = Path(__file__).parent.parent.parent
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))

from core.data_recipe.jobs import get_job_manager
from core.data_recipe.service import (
    build_config_builder,
    create_data_designer,
    validate_recipe,
)
from models.data_recipe import (
    JobCreateResponse,
    RecipePayload,
    SeedInspectRequest,
    SeedInspectUploadRequest,
    SeedInspectResponse,
    ValidateError,
    ValidateResponse,
)

router = APIRouter()
DATA_EXTS = (".parquet", ".jsonl", ".json", ".csv")
DEFAULT_SPLIT = "train"
LOCAL_UPLOAD_EXTS = {".csv", ".json", ".jsonl"}
SEED_UPLOAD_DIR = Path.home() / ".cache" / "unsloth" / "data-recipe" / "seed-uploads"


def _serialize_preview_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _serialize_preview_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize_preview_value(item) for item in value]
    return str(value)


def _serialize_preview_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {str(key): _serialize_preview_value(value) for key, value in row.items()}
        for row in rows
    ]


def _normalize_optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    trimmed = value.strip()
    return trimmed if trimmed else None


def _list_hf_data_files(*, dataset_name: str, token: str | None) -> list[str]:
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        repo_files = api.list_repo_files(dataset_name, repo_type="dataset", token=token)
        return [file for file in repo_files if file.lower().endswith(DATA_EXTS)]
    except Exception:
        return []


def _select_best_file(data_files: list[str], split: str | None) -> str | None:
    if not data_files:
        return None
    if not split:
        return data_files[0]
    split_lower = split.lower()

    def score(path: str) -> tuple[int, int]:
        name = path.lower()
        if f"/{split_lower}/" in name:
            return (0, len(path))
        if (
            f"_{split_lower}." in name
            or f"-{split_lower}." in name
            or f"/{split_lower}." in name
            or f"/{split_lower}_" in name
            or f"/{split_lower}-" in name
        ):
            return (1, len(path))
        return (2, len(path))

    return sorted(data_files, key=score)[0]


def _resolve_seed_hf_path(dataset_name: str, data_files: list[str], split: str | None) -> str | None:
    selected = _select_best_file(data_files, split)
    if not selected:
        return None

    ext = Path(selected).suffix.lower()
    if ext not in DATA_EXTS:
        return f"datasets/{dataset_name}/{selected}"

    parent = Path(selected).parent.as_posix()
    if not parent or parent == ".":
        return f"datasets/{dataset_name}/**/*{ext}"
    return f"datasets/{dataset_name}/{parent}/**/*{ext}"


def _build_stream_load_kwargs(
    *,
    dataset_name: str,
    split: str,
    subset: str | None,
    token: str | None,
    data_file: str | None = None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "path": dataset_name,
        "split": split,
        "streaming": True,
    }
    if data_file:
        kwargs["data_files"] = [data_file]
    if subset:
        kwargs["name"] = subset
    if token:
        kwargs["token"] = token
    return kwargs


def _load_preview_rows(
    *,
    load_dataset_fn,
    load_kwargs: dict[str, Any],
    preview_size: int,
) -> list[dict[str, Any]]:
    streamed_ds = load_dataset_fn(**load_kwargs)
    return [row for row in islice(streamed_ds, preview_size)]


def _extract_columns(rows: list[dict[str, Any]]) -> list[str]:
    columns_seen: dict[str, None] = {}
    for row in rows:
        for key in row.keys():
            columns_seen[str(key)] = None
    return list(columns_seen.keys())


def _sanitize_filename(filename: str) -> str:
    name = Path(filename).name.strip().replace("\x00", "")
    if not name:
        return "seed_upload"
    return name


def _decode_base64_payload(content_base64: str) -> bytes:
    raw = content_base64.strip()
    if "," in raw and raw.lower().startswith("data:"):
        raw = raw.split(",", 1)[1]
    try:
        return base64.b64decode(raw, validate=True)
    except binascii.Error as exc:
        raise HTTPException(status_code=400, detail="invalid base64 payload") from exc


def _read_preview_rows_from_local_file(path: Path, preview_size: int) -> list[dict[str, Any]]:
    try:
        import pandas as pd
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"seed inspect dependencies unavailable: {exc}") from exc

    ext = path.suffix.lower()
    try:
        if ext == ".csv":
            df = pd.read_csv(path, nrows=preview_size)
        elif ext == ".jsonl":
            df = pd.read_json(path, lines=True).head(preview_size)
        elif ext == ".json":
            try:
                df = pd.read_json(path, lines=True).head(preview_size)
            except Exception:
                df = pd.read_json(path).head(preview_size)
        else:
            raise HTTPException(status_code=422, detail=f"unsupported file type: {ext}")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"seed inspect failed: {exc}") from exc

    rows = df.to_dict(orient="records")
    return _serialize_preview_rows(rows)


def _collect_validation_errors(recipe: dict[str, Any]) -> list[ValidateError]:
    try:
        from data_designer.engine.compiler import (
            _add_internal_row_id_column_if_needed,
            _get_allowed_references,
            _resolve_and_add_seed_columns,
        )
        from data_designer.engine.validation import (
            ViolationLevel,
            validate_data_designer_config,
        )
    except Exception:
        return []

    try:
        builder = build_config_builder(recipe)
        designer = create_data_designer(recipe)
        resource_provider = designer._create_resource_provider(  # type: ignore[attr-defined]
            "validate-configuration",
            builder,
        )
        config = builder.build()
        _resolve_and_add_seed_columns(config, resource_provider.seed_reader)
        _add_internal_row_id_column_if_needed(config)
        violations = validate_data_designer_config(
            columns=config.columns,
            processor_configs=config.processors or [],
            allowed_references=_get_allowed_references(config),
        )
    except Exception:
        return []

    errors: list[ValidateError] = []
    for violation in violations:
        if violation.level != ViolationLevel.ERROR:
            continue
        code = getattr(violation.type, "value", None)
        path = violation.column if violation.column else None
        message = str(violation.message).strip() or "Validation failed."
        errors.append(
            ValidateError(
                message=message,
                path=path,
                code=code,
            )
        )
    return errors


@router.post("/seed/inspect", response_model=SeedInspectResponse)
def inspect_seed_dataset(payload: SeedInspectRequest) -> SeedInspectResponse:
    dataset_name = payload.dataset_name.strip()
    if not dataset_name or dataset_name.count("/") < 1:
        raise HTTPException(status_code=400, detail="dataset_name must be a Hugging Face repo id like org/repo")

    try:
        from datasets import load_dataset
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"seed inspect dependencies unavailable: {exc}") from exc

    split = (payload.split or DEFAULT_SPLIT).strip() or DEFAULT_SPLIT
    subset = _normalize_optional_text(payload.subset)
    token = _normalize_optional_text(payload.hf_token)
    preview_size = int(payload.preview_size)

    preview_rows: list[dict[str, Any]] = []
    data_files = _list_hf_data_files(dataset_name=dataset_name, token=token)

    selected_file = _select_best_file(data_files, split)
    if selected_file:
        try:
            single_file_kwargs = _build_stream_load_kwargs(
                dataset_name=dataset_name,
                split=DEFAULT_SPLIT,
                subset=subset,
                token=token,
                data_file=selected_file,
            )
            preview_rows = _load_preview_rows(
                load_dataset_fn=load_dataset,
                load_kwargs=single_file_kwargs,
                preview_size=preview_size,
            )
        except Exception:
            preview_rows = []

    if not preview_rows:
        try:
            split_kwargs = _build_stream_load_kwargs(
                dataset_name=dataset_name,
                split=split,
                subset=subset,
                token=token,
            )
            preview_rows = _load_preview_rows(
                load_dataset_fn=load_dataset,
                load_kwargs=split_kwargs,
                preview_size=preview_size,
            )
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"seed inspect failed: {exc}") from exc

    if not preview_rows:
        raise HTTPException(status_code=422, detail="dataset appears empty or unreadable")
    preview_rows = _serialize_preview_rows(preview_rows)
    columns = _extract_columns(preview_rows)

    if not data_files:
        # Best effort path fallback when file list is unavailable.
        resolved_path = f"datasets/{dataset_name}/**/*.parquet"
    else:
        resolved_path = _resolve_seed_hf_path(dataset_name, data_files, split)
        if not resolved_path:
            raise HTTPException(status_code=422, detail="unable to resolve seed dataset path")

    return SeedInspectResponse(
        dataset_name=dataset_name,
        resolved_path=resolved_path,
        columns=columns,
        preview_rows=preview_rows,
        split=split,
        subset=subset,
    )


@router.post("/seed/inspect-upload", response_model=SeedInspectResponse)
def inspect_seed_upload(payload: SeedInspectUploadRequest) -> SeedInspectResponse:
    filename = _sanitize_filename(payload.filename)
    ext = Path(filename).suffix.lower()
    if ext not in LOCAL_UPLOAD_EXTS:
        allowed = ", ".join(sorted(LOCAL_UPLOAD_EXTS))
        raise HTTPException(status_code=400, detail=f"unsupported file type: {ext}. allowed: {allowed}")

    file_bytes = _decode_base64_payload(payload.content_base64)
    if not file_bytes:
        raise HTTPException(status_code=400, detail="empty upload payload")
    max_size_bytes = 50 * 1024 * 1024
    if len(file_bytes) > max_size_bytes:
        raise HTTPException(status_code=413, detail="file too large (max 50MB)")

    SEED_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    stored_name = f"{uuid4().hex}_{filename}"
    stored_path = SEED_UPLOAD_DIR / stored_name
    stored_path.write_bytes(file_bytes)

    preview_rows = _read_preview_rows_from_local_file(
        stored_path,
        int(payload.preview_size),
    )
    if not preview_rows:
        raise HTTPException(status_code=422, detail="dataset appears empty or unreadable")
    columns = _extract_columns(preview_rows)

    return SeedInspectResponse(
        dataset_name=filename,
        resolved_path=str(stored_path),
        columns=columns,
        preview_rows=preview_rows,
        split=None,
        subset=None,
    )


@router.post("/validate", response_model=ValidateResponse)
def validate(payload: RecipePayload) -> ValidateResponse:
    recipe = payload.recipe
    if not recipe.get("columns"):
        return ValidateResponse(
            valid=False,
            errors=[ValidateError(message="Recipe must include columns.")],
        )

    try:
        validate_recipe(recipe)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        detail = str(exc).strip() or "Validation failed."
        parsed_errors = _collect_validation_errors(recipe)
        return ValidateResponse(
            valid=False,
            errors=parsed_errors or [ValidateError(message=detail)],
            raw_detail=detail,
        )

    return ValidateResponse(valid=True)


@router.post("/jobs", response_class=JSONResponse, response_model=JobCreateResponse)
def create_job(payload: RecipePayload):
    recipe = payload.recipe
    if not recipe.get("columns"):
        raise HTTPException(status_code=400, detail="Recipe must include columns.")

    run: dict[str, Any] = payload.run or {}
    run.pop("artifact_path", None)
    run.pop("dataset_name", None)
    execution_type = str(run.get("execution_type") or "full").strip().lower()
    if execution_type not in {"preview", "full"}:
        raise HTTPException(status_code=400, detail="invalid execution_type: must be 'preview' or 'full'")
    run["execution_type"] = execution_type
    run_config_raw = run.get("run_config")
    if run_config_raw is not None:
        try:
            from data_designer.config.run_config import RunConfig

            RunConfig.model_validate(run_config_raw)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"invalid run_config: {exc}") from exc

    mgr = get_job_manager()
    try:
        job_id = mgr.start(recipe=recipe, run=run)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {"job_id": job_id}


@router.get("/jobs/{job_id}/status")
def job_status(job_id: str):
    mgr = get_job_manager()
    state = mgr.get_status(job_id)
    if state is None:
        raise HTTPException(status_code=404, detail="job not found")
    return state


@router.get("/jobs/current")
def current_job():
    mgr = get_job_manager()
    state = mgr.get_current_status()
    if state is None:
        raise HTTPException(status_code=404, detail="no job")
    return state


@router.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    mgr = get_job_manager()
    ok = mgr.cancel(job_id)
    if not ok:
        raise HTTPException(status_code=404, detail="job not found")
    return mgr.get_status(job_id)


@router.get("/jobs/{job_id}/analysis")
def job_analysis(job_id: str):
    mgr = get_job_manager()
    analysis = mgr.get_analysis(job_id)
    if analysis is None:
        raise HTTPException(status_code=404, detail="analysis not ready")
    return analysis


@router.get("/jobs/{job_id}/dataset")
def job_dataset(
    job_id: str,
    limit: int = Query(default=20, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
):
    mgr = get_job_manager()
    result = mgr.get_dataset(job_id, limit=limit, offset=offset)
    if result is None:
        raise HTTPException(status_code=404, detail="dataset not ready")
    if "error" in result:
        raise HTTPException(status_code=422, detail=result["error"])
    return {
        "dataset": result["dataset"],
        "total": result["total"],
        "limit": limit,
        "offset": offset,
    }


@router.get("/jobs/{job_id}/events")
async def job_events(request: Request, job_id: str):
    mgr = get_job_manager()
    last_id = request.headers.get("last-event-id")
    after_seq: int | None = None
    if last_id:
        try:
            after_seq = int(str(last_id).strip())
        except Exception:
            after_seq = None

    # EventSource can't set custom headers on first connect after a full page refresh,
    # so allow resume via query param too: /events?after=<seq>
    after_q = request.query_params.get("after")
    if after_q:
        try:
            after_seq = int(str(after_q).strip())
        except Exception:
            pass

    sub = mgr.subscribe(job_id, after_seq=after_seq)
    if sub is None:
        raise HTTPException(status_code=404, detail="job not found")

    async def gen():
        try:
            for event in sub.replay:
                yield sub.format_sse(event)

            while True:
                if await request.is_disconnected():
                    break
                event = await sub.next_event(timeout_sec=1.0)
                if event is None:
                    continue
                yield sub.format_sse(event)
        finally:
            mgr.unsubscribe(sub)

    return StreamingResponse(gen(), media_type="text/event-stream")
