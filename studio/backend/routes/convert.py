# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import logging
import subprocess
import sys
import tempfile
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter(prefix = "/api/convert", tags = ["convert"])

# model_id -> {"state": "running" | "done" | "error", "stage": str}
# ponytail: in-process dict, lost on restart; persist if conversions must survive one.
_jobs: dict[str, dict] = {}

# Runs out of process so the export neither blocks the event loop nor fights the server's GPU
# state. Arguments go through argv, never into the source.
_SCRIPT = """
import sys
from unsloth import FastLanguageModel

model_id, format_type = sys.argv[1], sys.argv[2]
quant_method = "int4" if "int4" in format_type else "int8"
print("STAGE loading", flush=True)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_id, max_seq_length=2048, dtype=None, load_in_4bit=False,
)
print("STAGE saving", flush=True)
model.save_pretrained_openvino(
    f"{model_id}-{format_type}", tokenizer=tokenizer, quant_method=quant_method,
)
"""


class ConvertRequest(BaseModel):
    model_id: str
    format: str  # "ov_int4" or "ov_int8"


def run_conversion_task(model_id: str, format_type: str) -> None:
    job = _jobs[model_id]
    log_path = Path(tempfile.gettempdir()) / f"convert_{model_id.replace('/', '_')}.log"
    try:
        with open(log_path, "w") as log:
            proc = subprocess.Popen(
                [sys.executable, "-c", _SCRIPT, model_id, format_type],
                stdout = subprocess.PIPE,
                stderr = subprocess.STDOUT,
                text = True,
            )
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
    _jobs[req.model_id] = {"state": "running", "stage": "starting"}
    background_tasks.add_task(run_conversion_task, req.model_id, req.format)
    return {"status": "started", "model_id": req.model_id}


@router.get("/status")
async def conversion_status(model_id: str):
    job = _jobs.get(model_id)
    if job is None:
        raise HTTPException(404, "No conversion for this model")
    return job
