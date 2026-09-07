"""CI-only production route/tool bridge; never used by the Studio application."""

import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
EVIDENCE = Path(os.environ["SRT_BROWSER_ARTIFACTS"]).resolve()
os.environ["UNSLOTH_STUDIO_HOME"] = str(EVIDENCE / "studio-home")
os.environ["UNSLOTH_STUDIO_SANDBOX_HOME"] = str(EVIDENCE / "sandboxes")
sys.path.insert(0, str(ROOT / "studio/backend"))

from dataclasses import asdict
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from auth.authentication import get_current_subject, authenticated_via_api_key
from routes import inference
from core.inference import tools

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5197"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
app.include_router(inference.studio_router, prefix="/api/inference")
app.dependency_overrides[get_current_subject] = lambda: "srt-browser-fixture"
app.dependency_overrides[authenticated_via_api_key] = lambda: False


@app.get("/fixture/health")
def health():
    return {"ready": True}


@app.post("/fixture/execute")
def execute(body: dict):
    kind = body.get("kind")
    if kind not in ("python", "terminal"):
        raise HTTPException(400, "unsupported fixed fixture tool")
    mode = body.get("mode", "os_isolation_required")
    records = []
    payload = (
        {"code": "print('SRT_BROWSER_PYTHON_EXECUTED')"}
        if kind == "python"
        else {"command": "echo SRT_BROWSER_TERMINAL_EXECUTED"}
    )
    result = tools.execute_tool(
        kind,
        payload,
        session_id="srt-browser-fixture",
        timeout=30,
        tool_execution_mode=mode,
        current_subject="srt-browser-fixture",
        tool_ui_session_id=body.get("session"),
        limited_grant=body.get("grant"),
        disable_sandbox=mode == "full",
        launch_record_callback=records.append,
    )
    return {"result": result, "records": [asdict(r) for r in records], "python": sys.executable}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=5198)
