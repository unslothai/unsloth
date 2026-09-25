import asyncio
import logging
from pathlib import Path
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/convert", tags=["convert"])

class ConvertRequest(BaseModel):
    model_id: str
    format: str # "ov_int4" or "ov_int8"

def run_conversion_task(model_id: str, format_type: str):
    import subprocess
    import os
    
    # We will write a small standalone python script and run it
    # so we do not block the fastapi loop or conflict with GPU state
    script = f"""
from unsloth import FastLanguageModel
import sys

model_id = "{model_id}"
format_type = "{format_type}"
quant_method = "int4" if "int4" in format_type else "int8"

try:
    print(f"Loading {{model_id}}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_id,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=False,
    )
    
    # Save the model
    save_path = f"{{model_id}}-{{format_type}}"
    print(f"Saving to {{save_path}} with {{quant_method}}...")
    model.save_pretrained_openvino(save_path, tokenizer=tokenizer, quant_method=quant_method)
    print("Done!")
except Exception as e:
    print(f"Error: {{e}}")
    sys.exit(1)
"""
    script_path = Path(f"convert_script_{model_id.replace('/', '_')}.py")
    script_path.write_text(script)
    
    try:
        # Run it and redirect output to a log file
        log_path = Path(f"convert_{model_id.replace('/', '_')}.log")
        with open(log_path, "w") as f:
            subprocess.run(["python", str(script_path)], stdout=f, stderr=subprocess.STDOUT)
    except Exception as e:
        logger.error(f"Conversion failed: {e}")

@router.post("")
async def convert_model(req: ConvertRequest, background_tasks: BackgroundTasks):
    background_tasks.add_task(run_conversion_task, req.model_id, req.format)
    return {"status": "started", "model_id": req.model_id}
