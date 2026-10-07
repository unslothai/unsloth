# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import logging
import re
import subprocess
import sys
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel

from auth import policy
from auth.authentication import get_current_subject
from utils.paths import exports_root
from utils.process_lifetime import (
    adopt_pid,
    child_popen_kwargs,
    is_process_shutting_down,
    spawn_on_lifetime_thread,
    terminate_pid,
)

logger = logging.getLogger(__name__)


async def _require_installation_owner(current_subject: str = Depends(get_current_subject)) -> None:
    await policy.require_owner()


# Spawns a model load and export: owner only, like the NPU routes.
router = APIRouter(
    prefix = "/api/convert",
    tags = ["convert"],
    dependencies = [Depends(_require_installation_owner)],
)

# model_id -> {"state": "running" | "done" | "error", "stage": str}
# ponytail: in-process dict, lost on restart; persist if conversions must survive one.
_jobs: dict[str, dict] = {}

# Runs out of process so the export neither blocks the event loop nor fights the server's GPU
# state. Arguments go through argv, never into the source.
_SCRIPT = """
import sys
from unsloth import FastLanguageModel

model_id, format_type, output_dir = sys.argv[1], sys.argv[2], sys.argv[3]
quant_method = "int4" if "int4" in format_type else "int8"
print("STAGE loading", flush=True)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_id, max_seq_length=2048, dtype=None, load_in_4bit=False,
)
print("STAGE saving", flush=True)
model.save_pretrained_openvino(
    output_dir, tokenizer=tokenizer, quantization_type=quant_method,
)
"""


class ConvertRequest(BaseModel):
    model_id: str
    format: str  # "ov_int4" or "ov_int8"


def output_dir_for(model_id: str, format_type: str) -> Path:
    """``exports/<name>-openvino/<format>``: the {run}/{checkpoint} layout the exports scan lists,
    so the IR shows up in the model picker. A relative path would land in the server's cwd."""
    name = re.sub(r"[^A-Za-z0-9._-]", "_", Path(model_id.rstrip("/\\")).name).strip("._")
    return exports_root() / f"{name or 'model'}-openvino" / format_type


def run_conversion_task(model_id: str, format_type: str, output_dir: Path) -> None:
    job = _jobs[model_id]
    log_path = output_dir.parent / f"convert_{format_type}.log"
    try:
        output_dir.parent.mkdir(parents = True, exist_ok = True)
        with open(log_path, "w", encoding = "utf-8") as log:
            # -P: the server's cwd stays off sys.path, so a folder named "unsloth" there cannot shadow the package.
            argv = [sys.executable, "-P", "-c", _SCRIPT, model_id, format_type, str(output_dir)]
            proc = spawn_on_lifetime_thread(
                lambda: subprocess.Popen(
                    argv,
                    stdout = subprocess.PIPE,
                    stderr = subprocess.STDOUT,
                    text = True,
                    **child_popen_kwargs(),
                )
            )
            adopt_pid(proc.pid)
            if is_process_shutting_down():
                terminate_pid(proc.pid, timeout = 5.0, owner_verified = True)
                raise RuntimeError("Unsloth is shutting down; not converting.")
            for line in proc.stdout:
                log.write(line)
                if line.startswith("STAGE "):
                    job["stage"] = line.split(maxsplit = 1)[1].strip()
            proc.wait()
        job["state"] = "done" if proc.returncode == 0 else "error"
        if proc.returncode:
            job["stage"] = f"failed, see {log_path}"
    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        job.update(state = "error", stage = str(e))


@router.post("")
async def convert_model(req: ConvertRequest, background_tasks: BackgroundTasks):
    if req.format not in ("ov_int4", "ov_int8"):
        raise HTTPException(400, "format must be ov_int4 or ov_int8")
    if _jobs.get(req.model_id, {}).get("state") == "running":
        raise HTTPException(409, "Conversion already running")
    output_dir = output_dir_for(req.model_id, req.format)
    _jobs[req.model_id] = {"state": "running", "stage": "starting"}
    background_tasks.add_task(run_conversion_task, req.model_id, req.format, output_dir)
    return {"status": "started", "model_id": req.model_id}


@router.get("/status")
async def conversion_status(model_id: str):
    job = _jobs.get(model_id)
    if job is None:
        raise HTTPException(404, "No conversion for this model")
    return job
