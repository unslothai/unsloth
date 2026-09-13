# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Keep a download-only plan when another selection arrives mid-checkpoint.

Uses the rendered UI with deterministic API responses; no model bytes or GPU.
PW_LOAD_SECOND=1 also checks loading the same model after its queued download.
PW_HOLD_FIRST_PLAN=1 delays the first plan until the second selection has staged.
PW_RESOLVE_FIRST=1 delays the first selection's GGUF filename lookup.
"""

import json
import os
import time
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from playwright.sync_api import sync_playwright

from playwright_image_model_footprint import BASE_URL, REPO_ID, FILENAME, _api_payload, _json
from playwright_image_download_cancel_retry import _open_quant

LOAD_SECOND = os.environ.get("PW_LOAD_SECOND", "0") == "1"
HOLD_FIRST_PLAN = os.environ.get("PW_HOLD_FIRST_PLAN", "0") == "1"
RESOLVE_FIRST = os.environ.get("PW_RESOLVE_FIRST", "0") == "1"
SECOND = REPO_ID if LOAD_SECOND else "Tongyi-MAI/Z-Image-Turbo"
COMPANION = "black-forest-labs/FLUX.2-klein-4B"
ART = Path(os.environ.get("PW_ART_DIR", "logs/playwright_image_download_queue"))
ART.mkdir(parents = True, exist_ok = True)


def main() -> None:
    state = {"jobs": {}, "starts": [], "calls": [], "plans": [], "release": False}
    errors = []
    held_plans = []
    held_listings = []
    hold_plans = HOLD_FIRST_PLAN

    def plan(repo):
        entries = [
            {
                "repo_id": repo,
                "files": [FILENAME if repo == REPO_ID else "transformer/model.safetensors"],
                "bytes": 1024,
                "checkpoint": True,
            }
        ]
        if repo == REPO_ID:
            entries.append(
                {
                    "repo_id": COMPANION,
                    "files": ["vae/model.safetensors"],
                    "bytes": 2048,
                    "checkpoint": False,
                }
            )
        return {"entries": entries, "total_bytes": sum(e["bytes"] for e in entries)}

    with sync_playwright() as p:
        engine = getattr(p, os.environ.get("PW_BROWSER", "chromium"))
        executable = os.environ.get("PW_EXECUTABLE")
        browser = engine.launch(
            headless = True, **({"executable_path": executable} if executable else {})
        )
        context = browser.new_context(
            viewport = {"width": 1440, "height": 1000}, reduced_motion = "reduce"
        )
        context.add_init_script(
            "localStorage.setItem('unsloth_auth_token','rendered-ui-test');localStorage.setItem('unsloth_download_transport','http')"
        )

        def route(r):
            path = urlparse(r.request.url).path
            query = parse_qs(urlparse(r.request.url).query)
            if not path.startswith("/api/"):
                if urlparse(r.request.url).hostname in ("127.0.0.1", "localhost"):
                    r.continue_()
                else:
                    _json(r, [])
                return
            payload = json.loads(r.request.post_data or "{}")
            repo = payload.get("repo_id") or query.get("repo_id", [None])[0]
            if path == "/api/inference/images/download-plan":
                selected = payload["model_path"]
                state["plans"].append(selected)
                if hold_plans:
                    held_plans.append((r, selected))
                else:
                    _json(r, plan(selected))
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
                assert payload["scope_id"] == "diffusion", payload
                state["starts"].append(repo)
                job = {"state": "running", "generation": len(state["starts"]), "polls": 0}
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
            elif path == "/api/hub/download-status":
                job = state["jobs"].get(repo, {"state": "idle"})
                if job["state"] == "running" and state["release"]:
                    job["polls"] += 1
                    if job["polls"] >= 2:
                        job["state"] = "complete"
                _json(r, {**job, "error": None})
            elif path in ("/api/hub/download-progress", "/api/hub/gguf-download-progress"):
                complete = state["jobs"].get(repo, {}).get("state") == "complete"
                size = 2048 if repo == COMPANION else 1024
                _json(
                    r,
                    {
                        "downloaded_bytes": size if complete else 256,
                        "completed_bytes": size if complete else 256,
                        "complete_on_disk": complete,
                        "expected_bytes": size,
                        "progress": 1 if complete else 0.25,
                        "cache_path": "C:\\mock-cache",
                    },
                )
            elif path in ("/api/inference/images/load", "/api/inference/images/unload"):
                state["calls"].append(path)
                _json(r, _api_payload("/api/inference/images/status", {}, full_footprint = True))
            else:
                result = _api_payload(path, query, full_footprint = True)
                if path in ("/api/hub/gguf-variants", "/api/models/gguf-variants"):
                    result["variants"][0]["downloaded"] = False
                    if RESOLVE_FIRST:
                        held_listings.append((r, result))
                        return
                _json(r, result)

        context.route("**/*", route)
        page = context.new_page()
        page.on("pageerror", lambda e: errors.append(str(e)))

        def wait_for(predicate):
            deadline = time.monotonic() + 15
            while not predicate() and time.monotonic() < deadline:
                page.wait_for_timeout(100)
            assert predicate(), state

        page.goto(BASE_URL + "/images", wait_until = "domcontentloaded")
        page.get_by_role("button", name = "Advanced", exact = True).click()
        page.get_by_role("combobox", name = "On model selection").click()
        page.get_by_role("option", name = "Download only", exact = True).click()
        page.get_by_role("button", name = "Advanced", exact = True).click()
        if RESOLVE_FIRST:
            page.evaluate(
                """url => {
                history.pushState(null, '', url);
                dispatchEvent(new PopStateEvent('popstate'));
            }""",
                f"/images?model={REPO_ID}&ggufQuant=Q4_K_M",
            )
            wait_for(lambda: len(held_listings) > 0)
        else:
            _open_quant(page, navigate = False)
        if RESOLVE_FIRST:
            assert not state["starts"]
        elif HOLD_FIRST_PLAN:
            wait_for(lambda: state["plans"].count(REPO_ID) >= 2)
        else:
            wait_for(lambda: REPO_ID in state["starts"])
        hold_plans = False
        plan_count = len(state["plans"])
        if LOAD_SECOND:
            page.get_by_role("button", name = "Advanced", exact = True).click()
            page.get_by_role("combobox", name = "On model selection").click()
            page.get_by_role("option", name = "Download and load", exact = True).click()
            page.get_by_role("button", name = "Advanced", exact = True).click()
            _open_quant(page, navigate = False)
        else:
            trigger = page.locator(".unsloth-model-selector-trigger:visible")
            trigger.scroll_into_view_if_needed()
            trigger.click()
            page.locator(".unsloth-model-selector-menu [data-model-picker-option]").filter(
                has_text = "Z-Image-Turbo"
            ).filter(has_text = "BF16").first.click()
        wait_for(lambda: len(state["plans"]) > plan_count and SECOND in state["plans"])
        page.wait_for_timeout(300)
        assert state["starts"] == [SECOND if HOLD_FIRST_PLAN or RESOLVE_FIRST else REPO_ID], state
        for route, result in held_listings:
            _json(route, result)
        held_listings.clear()
        for route, selected in held_plans:
            _json(route, plan(selected))
        held_plans.clear()
        state["release"] = True
        expected = [REPO_ID, COMPANION] if LOAD_SECOND else [REPO_ID, COMPANION, SECOND]
        if (HOLD_FIRST_PLAN or RESOLVE_FIRST) and not LOAD_SECOND:
            expected = [SECOND, REPO_ID, COMPANION]
        try:
            wait_for(
                lambda: len(state["starts"]) >= len(expected)
                and all(j["state"] == "complete" for j in state["jobs"].values())
            )
            if LOAD_SECOND:
                wait_for(lambda: len(state["calls"]) > 0)
                assert set(state["starts"]) == set(expected), state
                assert all(job["state"] == "complete" for job in state["jobs"].values()), state
            else:
                assert state["starts"] == expected, state
            expected_calls = ["/api/inference/images/load"] if LOAD_SECOND else []
            assert state["calls"] == expected_calls, state
            assert not errors, errors
        finally:
            (ART / "result.json").write_text(
                json.dumps({**state, "page_errors": errors}, indent = 2), encoding = "utf-8"
            )
            page.screenshot(path = str(ART / "queue.png"), full_page = True)
            browser.close()
        print(f'PASS download-only queue: {state["starts"]}', flush = True)


if __name__ == "__main__":
    main()
