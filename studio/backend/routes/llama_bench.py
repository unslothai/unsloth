# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""llama-bench on the loaded GGUF. The model is unloaded for the run so llama-bench has the
GPU to itself, and the finished run is saved beside the config sweeps as kind "llama-bench".
The client never names a file: the GGUF is whatever chat has loaded, already resolved."""

import json
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Any, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, field_validator

from auth.authentication import authenticated_via_api_key, get_current_subject
from hub.services.models import account_access
from hub.utils.host_paths import host_paths_visible, redact_host_paths, redact_paths_in_text
from storage.benchmark_runs_db import upsert_run

router = APIRouter()

KIND = "llama-bench"
_PROGRESS_RE = re.compile(r"^llama-bench: benchmark (\d+)/(\d+): (.+)$")
# Fields worth keeping per row; the full jsonl row carries ~40, mostly echoes of the flags.
_ROW_KEYS = (
    "n_prompt",
    "n_gen",
    "n_depth",
    "avg_ts",
    "stddev_ts",
    "samples_ts",
    "n_gpu_layers",
    "flash_attn",
)
_META_KEYS = (
    "build_commit",
    "build_number",
    "cpu_info",
    "gpu_info",
    "backends",
    "model_type",
    "model_size",
    "model_n_params",
)


class LlamaBenchRequest(BaseModel):
    prompt_tokens: list[int] = Field(default_factory = lambda: [512], max_length = 8)
    gen_tokens: list[int] = Field(default_factory = lambda: [128], max_length = 8)
    depths: list[int] = Field(default_factory = lambda: [0], min_length = 1, max_length = 8)
    repetitions: int = Field(default = 5, ge = 1, le = 20)
    flash_attn: Literal["auto", "on", "off"] = "auto"
    n_gpu_layers: Optional[int] = Field(default = None, ge = 0, le = 999)

    @field_validator("prompt_tokens", "gen_tokens", "depths")
    @classmethod
    def _in_range(cls, v: list[int]) -> list[int]:
        if any(n < 0 or n > 262_144 for n in v):
            raise ValueError("token counts must be between 0 and 262144")
        return sorted(set(v))

    def test_count(self) -> int:
        tests = sum(1 for n in self.prompt_tokens if n > 0) + sum(
            1 for n in self.gen_tokens if n > 0
        )
        return tests * len(self.depths)


class _Job:
    def __init__(self, request: LlamaBenchRequest, model: str, variant: Optional[str]):
        self.id = f"llama-bench-{uuid.uuid4().hex[:12]}"
        self.request = request
        self.model = model
        self.variant = variant
        self.status = "running"
        self.stage = "Starting llama-bench"
        self.error: Optional[str] = None
        self.rows: list[dict[str, Any]] = []
        self.meta: dict[str, Any] = {}
        self.log: deque[str] = deque(maxlen = 200)
        self.total = request.test_count()
        self.created_at = int(time.time() * 1000)
        self.finished_at: Optional[int] = None
        self.proc: Optional[subprocess.Popen] = None
        self.cancelled = False

    def snapshot(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "status": self.status,
            "stage": self.stage,
            "error": self.error,
            "model": self.model,
            "ggufVariant": self.variant,
            "config": self.request.model_dump(),
            "rows": list(self.rows),
            "meta": dict(self.meta),
            "done": len(self.rows),
            "total": self.total,
            "log": list(self.log)[-40:],
            "createdAt": self.created_at,
            "finishedAt": self.finished_at,
        }


def _public(snapshot: Optional[dict[str, Any]], via_api_key: bool) -> Optional[dict[str, Any]]:
    """A local GGUF's model is its absolute path; API-key callers get it redacted, as on /runs.
    llama-bench's own errors name the file too, so the error and log lose their paths as well."""
    if snapshot is None:
        return None
    # The run and the resident model are the owner's; /api/inference/status hides them from
    # managed accounts the same way.
    if account_access.managed_account():
        return None
    if host_paths_visible(via_api_key):
        return snapshot
    ids = redact_host_paths({"active_model": snapshot.get("model")}, via_api_key = via_api_key)
    public = {**snapshot, "model": ids["active_model"]}
    if "error" in public:
        public["error"] = redact_paths_in_text(public["error"]) or None
    if "log" in public:
        public["log"] = [redact_paths_in_text(line) for line in public["log"]]
    return public


_lock = threading.Lock()
_job: Optional[_Job] = None


def find_llama_bench() -> Optional[Path]:
    """llama-bench ships beside llama-server, in the directory its libraries live in."""
    from core.inference.llama_cpp import LlamaCppBackend, _llama_lib_dir

    server = LlamaCppBackend._find_llama_server_binary()
    if not server:
        return None
    name = "llama-bench.exe" if sys.platform == "win32" else "llama-bench"
    for folder in (_llama_lib_dir(server), Path(server).parent):
        candidate = folder / name
        # A non-executable copy would unload chat's model and then fail to start.
        if candidate.is_file() and (sys.platform == "win32" or os.access(candidate, os.X_OK)):
            return candidate
    return None


def _command(binary: Path, gguf: str, request: LlamaBenchRequest) -> list[str]:
    cmd = [str(binary), "-m", gguf, "-o", "jsonl", "--progress"]
    cmd += ["-p", ",".join(map(str, request.prompt_tokens or [0]))]
    cmd += ["-n", ",".join(map(str, request.gen_tokens or [0]))]
    # -d is newer than the other flags; leave it off unless asked so older builds still run.
    if any(request.depths):
        cmd += ["-d", ",".join(map(str, request.depths))]
    cmd += ["-r", str(request.repetitions)]
    if request.flash_attn != "auto":
        cmd += ["-fa", "1" if request.flash_attn == "on" else "0"]
    if request.n_gpu_layers is not None:
        cmd += ["-ngl", str(request.n_gpu_layers)]
    return cmd


def _row(raw: dict[str, Any]) -> dict[str, Any]:
    row = {k: raw.get(k) for k in _ROW_KEYS}
    row["test"] = (
        f"pp{raw.get('n_prompt')}" if raw.get("n_prompt") else f"tg{raw.get('n_gen')}"
    ) + (f" @ d{raw['n_depth']}" if raw.get("n_depth") else "")
    return row


def _save(job: _Job) -> None:
    upsert_run(
        {
            "id": job.id,
            "kind": KIND,
            "sweep": KIND,
            "model": job.model,
            "ggufVariant": job.variant,
            "config": job.request.model_dump(),
            "meta": job.meta,
            "outcomes": job.rows,
            "results": [],
            "createdAt": job.created_at,
            "finishedAt": job.finished_at,
        }
    )


def _run(job: _Job, binary: Path, gguf: str) -> None:
    from core.inference.llama_cpp import LlamaCppBackend
    try:
        env = LlamaCppBackend._llama_server_env_for_binary(str(binary))
        job.proc = subprocess.Popen(
            _command(binary, gguf, job.request),
            stdout = subprocess.PIPE,
            stderr = subprocess.PIPE,
            stdin = subprocess.DEVNULL,
            env = env,
            text = True,
            encoding = "utf-8",
            errors = "replace",
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if job.cancelled:
            job.proc.terminate()

        def read_stderr(stream) -> None:
            for line in stream:
                line = line.rstrip()
                if not line:
                    continue
                job.log.append(line)
                m = _PROGRESS_RE.match(line)
                if m:
                    job.stage = m.group(3).capitalize()

        reader = threading.Thread(target = read_stderr, args = (job.proc.stderr,), daemon = True)
        reader.start()
        for line in job.proc.stdout:
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                raw = json.loads(line)
            except ValueError:
                continue
            if not job.meta:
                job.meta = {k: raw.get(k) for k in _META_KEYS}
            job.rows.append(_row(raw))
        code = job.proc.wait()
        reader.join(timeout = 2)
        if job.cancelled:
            job.status = "cancelled"
        elif code != 0:
            job.status = "error"
            tail = [l for l in job.log if "error" in l.lower()] or list(job.log)
            job.error = (tail[-1] if tail else f"llama-bench exited with code {code}")[:500]
        else:
            job.status = "done"
    except Exception as exc:  # the job must end in a terminal state whatever went wrong
        job.status = "error"
        job.error = str(exc)[:500]
    finally:
        job.finished_at = int(time.time() * 1000)
        job.stage = ""
        if job.rows:
            try:
                _save(job)
            except Exception as exc:
                job.error = job.error or f"Couldn't save the run: {exc}"[:500]


def _owner_only() -> None:
    if account_access.managed_account():
        raise HTTPException(status_code = 403, detail = "llama-bench is for the owner account")


@router.get("/status")
def status(
    current_subject: str = Depends(get_current_subject),
    via_api_key: bool = Depends(authenticated_via_api_key),
):
    from routes.inference import get_llama_cpp_backend

    backend = get_llama_cpp_backend()
    loaded = backend.is_loaded and bool(backend.gguf_path)
    with _lock:
        job = _job.snapshot() if _job else None
    loaded_model = _public({"model": backend.model_identifier if loaded else None}, via_api_key)
    return {
        "available": find_llama_bench() is not None,
        "model": loaded_model["model"] if loaded_model else None,
        "ggufVariant": backend.hf_variant if loaded and loaded_model else None,
        "job": _public(job, via_api_key),
    }


@router.post("/run")
@account_access.gpu_busy_route
async def run(
    request: LlamaBenchRequest,
    current_subject: str = Depends(get_current_subject),
    via_api_key: bool = Depends(authenticated_via_api_key),
):
    global _job
    _owner_only()
    if request.test_count() == 0:
        raise HTTPException(
            status_code = 400, detail = "Nothing to measure: add a prompt or a generation size"
        )
    binary = find_llama_bench()
    if binary is None:
        raise HTTPException(
            status_code = 409,
            detail = {
                "error": "llama_bench_missing",
                "message": "This llama.cpp install doesn't include llama-bench. It comes with the next "
                "llama.cpp update.",
            },
        )
    from models.inference import UnloadRequest
    from routes.inference import _unload_model_impl, get_llama_cpp_backend

    backend = get_llama_cpp_backend()
    gguf, model, variant = backend.gguf_path, backend.model_identifier, backend.hf_variant
    if not (backend.is_loaded and gguf and model):
        raise HTTPException(status_code = 409, detail = "Load a GGUF model first")
    with _lock:
        if _job is not None and _job.status == "running":
            raise HTTPException(status_code = 409, detail = "A llama-bench run is already going")
        job = _Job(request, model, variant)
        _job = job
    try:
        # The real unload route, so chats still generating refuse it with 409 and keep-warm
        # forgets the model instead of reloading it under llama-bench.
        await _unload_model_impl(UnloadRequest(model_path = model), current_subject)
    except BaseException:
        with _lock:
            if _job is job:
                _job = None
        raise
    threading.Thread(target = _run, args = (job, binary, gguf), daemon = True).start()
    return _public(job.snapshot(), via_api_key)


@router.get("/run")
def current(
    current_subject: str = Depends(get_current_subject),
    via_api_key: bool = Depends(authenticated_via_api_key),
):
    with _lock:
        job = _job.snapshot() if _job else None
    return {"job": _public(job, via_api_key)}


@router.delete("/run")
def cancel(
    current_subject: str = Depends(get_current_subject),
    via_api_key: bool = Depends(authenticated_via_api_key),
):
    _owner_only()
    with _lock:
        job = _job
    if job is not None and job.status == "running":
        job.cancelled = True
        if job.proc is not None and job.proc.poll() is None:
            job.proc.terminate()
    return {"job": _public(job.snapshot() if job else None, via_api_key)}
