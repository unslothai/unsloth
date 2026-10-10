# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Cached image model: show both missing asset stages, then cancel/retry/complete.

Runs the real picker, staging hook, and download panel with deterministic API
responses. No model download or GPU. PW_EXPECT=before records the missing row;
otherwise asserts parity. Use BASE_URL and PW_ART_DIR for isolated comparisons.
"""

import json
import os
import time
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from playwright.sync_api import expect, sync_playwright
from playwright_image_model_footprint import (
    BASE_URL,
    REPO_ID,
    CHECKPOINT_BYTES,
    _api_payload,
    _json,
)
from playwright_image_download_cancel_retry import _open_quant

ENCODER = "unsloth/FLUX.2-klein-4B-FP8"
DECODER = "unsloth/FLUX.2-klein-4B"
ENTRIES = [
    {
        "repo_id": ENCODER,
        "files": ["text_encoder/model.safetensors"],
        "bytes": 4_000_000_000,
        "checkpoint": False,
        "gguf_filename": None,
    },
    {
        "repo_id": DECODER,
        "files": ["vae/diffusion_pytorch_model.safetensors", "model_index.json"],
        "bytes": 200_000_000,
        "checkpoint": False,
        "gguf_filename": None,
    },
]


def main():
    before = os.environ.get("PW_EXPECT", "after") == "before"
    art = Path(os.environ.get("PW_ART_DIR", "logs/playwright_staged_download_rows"))
    art.mkdir(parents = True, exist_ok = True)
    state = {"jobs": {}, "starts": [], "loads": 0, "released": [], "plans": []}
    errors = []
    facts = {}
    with sync_playwright() as pw:
        browser = getattr(pw, os.environ.get("PW_BROWSER", "chromium")).launch(headless = True)
        context = browser.new_context(
            viewport = {"width": 1440, "height": 1000}, reduced_motion = "reduce"
        )
        context.add_init_script(
            "localStorage.setItem('unsloth_auth_token','rendered-ui-test');localStorage.setItem('unsloth_download_transport','http')"
        )

        def route(r):
            url = urlparse(r.request.url)
            path = url.path
            query = parse_qs(url.query)
            if not path.startswith("/api/"):
                if url.hostname in ("localhost", "127.0.0.1"):
                    r.continue_()
                else:
                    _json(r, [])
                return
            data = json.loads(r.request.post_data or "{}")
            repo = data.get("repo_id") or query.get("repo_id", [None])[0]
            if path == "/api/inference/images/download-plan":
                state["plans"].append(data)
                _json(
                    r,
                    {
                        "entries": ENTRIES,
                        "total_bytes": 4_200_000_000,
                        "required_bytes": CHECKPOINT_BYTES + 4_200_000_000,
                        "checkpoint_bytes": CHECKPOINT_BYTES,
                    },
                )
            elif path == "/api/studio/download-transport-capabilities":
                _json(
                    r,
                    {
                        "http": {"available": True},
                        "xet": {"available": False},
                        "auto_resolves_to": "http",
                    },
                )
            elif path == "/api/hub/transport-status":
                _json(r, {"has_partial": False, "last_transport": None, "resumable": False})
            elif path == "/api/hub/download" and r.request.method == "POST":
                assert data["scope_id"] == "diffusion", data
                assert repo != REPO_ID, "cached model must not download again"
                state["starts"].append(repo)
                job = {"state": "running", "generation": len(state["starts"])}
                state["jobs"][repo] = job
                _json(
                    r,
                    {
                        **job,
                        "accepted": True,
                        "job_key": f"model:{repo}:@diffusion",
                        "transport": "http",
                    },
                )
            elif path == "/api/hub/download/cancel":
                state["jobs"][repo]["state"] = "cancelled"
                _json(r, {"success": True, "state": "cancelled"})
            elif path == "/api/hub/download-status":
                job = state["jobs"].get(repo, {"state": "idle"})
                if job["state"] == "running" and repo in state["released"]:
                    job["state"] = "complete"
                _json(r, {**job, "error": None})
            elif path in ("/api/hub/download-progress", "/api/hub/gguf-download-progress"):
                complete = state["jobs"].get(repo, {}).get("state") == "complete"
                size = next((e["bytes"] for e in ENTRIES if e["repo_id"] == repo), CHECKPOINT_BYTES)
                _json(
                    r,
                    {
                        "downloaded_bytes": size if complete else size // 4,
                        "completed_bytes": size if complete else 0,
                        "complete_on_disk": complete,
                        "expected_bytes": size,
                        "progress": 1 if complete else 0.25,
                        "cache_path": "/simulated-cache",
                    },
                )
            elif path == "/api/inference/images/load":
                state["loads"] += 1
                _json(r, {"status": "loading"})
            else:
                result = _api_payload(path, query, full_footprint = True)
                if path in ("/api/hub/gguf-variants", "/api/models/gguf-variants"):
                    result["variants"][0]["downloaded"] = True
                _json(r, result)

        context.route("**/*", route)
        page = context.new_page()
        page.on("pageerror", lambda e: errors.append(str(e)))

        def wait_for(predicate):
            deadline = time.monotonic() + 30
            while not predicate() and time.monotonic() < deadline:
                page.wait_for_timeout(100)
            assert predicate(), state

        try:
            page.goto(BASE_URL + "/images", wait_until = "domcontentloaded")
            expect(page.locator(".hub-download-panel")).to_have_count(0)
            _open_quant(page, navigate = False)
            wait_for(lambda: state["starts"] == [ENCODER])
            panel = page.locator(".hub-download-panel")
            expect(panel).to_be_visible()
            expected_rows = 1 if before else 2
            expect(panel.locator("li")).to_have_count(expected_rows)
            expect(panel).to_contain_text(f"Downloading {expected_rows} item")
            expect(panel).to_contain_text("Text encoder")
            expect(panel).to_contain_text("1.0 GB / 4.0 GB")
            if not before:
                expect(panel).to_contain_text("Decoder & configuration")
                expect(panel).to_contain_text("Queued")
            page.evaluate("""async () => {
                await document.fonts.load('12px "Inter Variable"');
                await document.fonts.ready;
            }""")
            facts["initial"] = {
                "rows": panel.locator("li").count(),
                "text": panel.inner_text(),
                "starts": list(state["starts"]),
            }
            panel.screenshot(path = str(art / "panel.png"))
            page.screenshot(path = str(art / "page.png"))
            panel.get_by_role("button", name = "Cancel download", exact = True).click()
            wait_for(lambda: state["jobs"][ENCODER]["state"] == "cancelled")
            expect(panel).not_to_contain_text("Queued")
            expect(panel).not_to_contain_text(DECODER + " ·")
            assert state["loads"] == 0
            _open_quant(page, navigate = False)
            wait_for(lambda: state["starts"] == [ENCODER, ENCODER])
            expect(panel.locator("li")).to_have_count(expected_rows)
            state["released"].append(ENCODER)
            wait_for(lambda: DECODER in state["starts"])
            expect(panel).not_to_contain_text("Queued")
            expect(panel.get_by_role("button", name = "Cancel download", exact = True)).to_have_count(1)
            state["released"].append(DECODER)
            wait_for(lambda: state["loads"] == 1)
            expect(panel).not_to_contain_text("Downloading")
            assert state["starts"] == [ENCODER, ENCODER, DECODER]
            for _ in range(panel.get_by_role("button", name = "Dismiss", exact = True).count()):
                panel.get_by_role("button", name = "Dismiss", exact = True).first.click()
            expect(panel).to_have_count(0)
            assert not errors, errors
            facts["lifecycle"] = (
                "cancel removes pending row; retry restores it; stages advance once; load runs once"
            )
        finally:
            (art / "facts.json").write_text(
                json.dumps(
                    {
                        "expect": "cached model: active encoder plus queued decoder",
                        "side": "before" if before else "after",
                        "url": BASE_URL,
                        "facts": facts,
                        "state": state,
                        "page_errors": errors,
                    },
                    indent = 2,
                )
            )
            browser.close()
    print(f"PASS staged rows ({'before' if before else 'after'}): {facts}")


if __name__ == "__main__":
    main()
