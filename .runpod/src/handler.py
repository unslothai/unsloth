import runpod
import subprocess
import os

import train


def handler(job):
    job_input = job["input"]

    try:
        train.run(job_input)
    except subprocess.CalledProcessError as e:
        return {"error": str(e)}

    return {
        "status": "Training complete",
        "output_dir": job_input.get("output_dir", "outputs"),
    }


runpod.serverless.start({"handler": handler})
