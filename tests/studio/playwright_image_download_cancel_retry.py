# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Rendered cancel/retry regression for staged diffusion downloads.

The browser runs the real Studio UI against a deterministic API simulation of
the reported sequence:

1. Start with the 2.6 GB Klein GGUF already cached.
2. Stage its missing 8.2 GB companion assets through the Downloads panel.
3. Cancel, pick the same quant again, and resume only those companion assets.
4. Load exactly once and observe a short 100% GPU-finalization state that reaches ready.

No model bytes are downloaded and no GPU is required.
"""

import json
import os
import time
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from playwright.sync_api import Route, sync_playwright

from playwright_image_model_footprint import (
    BASE_URL,
    CHECKPOINT_BYTES,
    COMPANION_BYTES,
    FILENAME,
    REPO_ID,
    REQUIRED_BYTES,
    _api_payload,
    _json,
    klein_row,
)


ART_DIR = Path(os.environ.get("PW_ART_DIR", "logs/playwright_image_cancel_retry"))
ART_DIR.mkdir(parents = True, exist_ok = True)
EXPECT = os.environ.get("PW_EXPECT", "after").strip().lower()
if EXPECT not in {"before", "after"}:
    raise ValueError("PW_EXPECT must be 'before' or 'after'")
CHECKPOINT_CACHED = os.environ.get("PW_CHECKPOINT_CACHED", "1").strip() != "0"
if EXPECT == "before" and not CHECKPOINT_CACHED:
    raise ValueError("The before-state requires PW_CHECKPOINT_CACHED=1")

COMPANION_REPO = "black-forest-labs/FLUX.2-klein-4B"
COMPANION_FILES = [
    "text_encoder/model-00001-of-00002.safetensors",
    "text_encoder/model-00002-of-00002.safetensors",
    "vae/diffusion_pytorch_model.safetensors",
]


def _status(*, loaded: bool) -> dict[str, object]:
    return {
        "loaded": loaded,
        "repo_id": REPO_ID if loaded else None,
        "family": "flux2-klein" if loaded else None,
        "base_repo": COMPANION_REPO if loaded else None,
        "device": "cuda:0" if loaded else None,
        "dtype": "bfloat16" if loaded else None,
        "model_kind": "gguf" if loaded else None,
        "cpu_offload": False,
        "workflows": ["text2img"] if loaded else [],
        "supports_lora": False,
        "supports_controlnet": False,
    }


def _entry(repo_id: str) -> dict[str, object]:
    if repo_id == REPO_ID:
        return {
            "repo_id": REPO_ID,
            "files": [FILENAME],
            "bytes": CHECKPOINT_BYTES,
            "gguf_filename": FILENAME,
        }
    return {
        "repo_id": COMPANION_REPO,
        "files": COMPANION_FILES,
        "bytes": COMPANION_BYTES,
        "gguf_filename": None,
    }


def _open_quant(page, *, navigate: bool) -> None:
    if navigate:
        page.goto(f"{BASE_URL}/images", wait_until = "domcontentloaded")
    trigger = page.get_by_role("button", name = "Select image model")
    trigger.wait_for(state = "visible", timeout = 30_000)
    trigger.click()
    klein_row(page).click()
    gguf = page.get_by_text("GGUF", exact = True)
    if gguf.count() == 1:
        gguf.click()
    quant = page.locator("button").filter(has_text = "Q4_K_M")
    quant.wait_for(state = "visible")
    assert quant.count() == 1
    quant.click()


def main() -> None:
    state: dict[str, object] = {
        "checkpoint_cached": CHECKPOINT_CACHED,
        "companion_attempt": 0,
        "jobs": {},
        "starts": [],
        "cancelled": False,
        "load_calls": 0,
        "load_payloads": [],
        "load_progress_polls": 0,
        "loaded": False,
        "plan_snapshots": [],
    }
    page_errors: list[str] = []

    def download_plan() -> dict[str, object]:
        entries = [_entry(COMPANION_REPO)]
        if not state["checkpoint_cached"]:
            entries.insert(0, _entry(REPO_ID))
        snapshot = [entry["repo_id"] for entry in entries]
        state["plan_snapshots"].append(snapshot)
        return {
            "entries": entries,
            "total_bytes": sum(int(entry["bytes"]) for entry in entries),
            "required_bytes": REQUIRED_BYTES,
            "checkpoint_bytes": CHECKPOINT_BYTES,
        }

    def start_download(payload: dict[str, object]) -> dict[str, object]:
        repo_id = str(payload["repo_id"])
        assert payload.get("scope_id") == "diffusion", payload
        assert payload.get("gguf_variant") is None, payload
        generation = len(state["starts"]) + 1
        state["starts"].append(repo_id)
        if repo_id == COMPANION_REPO:
            state["companion_attempt"] = int(state["companion_attempt"]) + 1
        state["jobs"][repo_id] = {
            "state": "running",
            "generation": generation,
            "polls": 0,
            "attempt": int(state["companion_attempt"]) if repo_id == COMPANION_REPO else 1,
        }
        return {
            "job_key": f"model:{repo_id}:@diffusion",
            "accepted": True,
            "state": "running",
            "generation": generation,
            "transport": "http",
            "cancel_transport": None,
        }

    def job_status(repo_id: str) -> dict[str, object]:
        job = state["jobs"].get(repo_id)
        if job is None:
            return {"state": "idle", "error": None}
        if job["state"] == "running":
            job["polls"] = int(job["polls"]) + 1
            should_complete = repo_id == REPO_ID or int(job["attempt"]) >= 2
            if should_complete and int(job["polls"]) >= 3:
                job["state"] = "complete"
                if repo_id == REPO_ID:
                    state["checkpoint_cached"] = True
        return {
            "state": job["state"],
            "error": None,
            "generation": job["generation"],
        }

    def job_progress(repo_id: str) -> dict[str, object]:
        job = state["jobs"].get(repo_id)
        expected = CHECKPOINT_BYTES if repo_id == REPO_ID else COMPANION_BYTES
        complete = bool(job and job["state"] == "complete")
        if complete:
            downloaded = expected
        elif repo_id == COMPANION_REPO:
            downloaded = 1_100_000_000
        else:
            downloaded = 1_900_000_000
        return {
            "downloaded_bytes": downloaded,
            "completed_bytes": downloaded,
            "complete_on_disk": complete,
            "expected_bytes": expected,
            "progress": downloaded / expected,
            "cache_path": f"C:\\mock-cache\\{repo_id.replace('/', '--')}",
        }

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless = True)
        context = browser.new_context(
            viewport = {"width": 1440, "height": 900},
            reduced_motion = "reduce",
            color_scheme = "dark",
        )
        context.add_init_script(
            "localStorage.setItem('unsloth_auth_token', 'rendered-ui-test');"
            "localStorage.setItem('unsloth_download_transport', 'http');"
        )

        def route_request(route: Route) -> None:
            request = route.request
            parsed = urlparse(request.url)
            path = parsed.path
            query = parse_qs(parsed.query)
            if parsed.netloc == "huggingface.co":
                _json(route, [])
                return
            if not path.startswith("/api/"):
                route.continue_()
                return

            payload = json.loads(request.post_data or "{}")
            if path in {"/api/hub/gguf-variants", "/api/models/gguf-variants"}:
                variants = _api_payload(path, query, full_footprint = True)
                # This is the critical reported precondition: the picker knows the 2.6 GB
                # checkpoint is cached, but its separate 8.2 GB requirements are absent.
                variants["variants"][0]["downloaded"] = CHECKPOINT_CACHED
                _json(route, variants)
                return
            if path == "/api/inference/images/download-plan":
                _json(route, download_plan())
                return
            if path == "/api/studio/download-transport-capabilities":
                _json(
                    route,
                    {
                        "http": {"available": True, "reason": None},
                        "xet": {"available": False, "reason": "Rendered test uses HTTP"},
                        "auto_resolves_to": "http",
                        "auto_reason": "Deterministic rendered test",
                    },
                )
                return
            if path == "/api/hub/transport-status":
                _json(
                    route,
                    {"has_partial": False, "last_transport": None, "resumable": False},
                )
                return
            if path == "/api/hub/download" and request.method == "POST":
                _json(route, start_download(payload))
                return
            if path == "/api/hub/download/cancel" and request.method == "POST":
                repo_id = str(payload["repo_id"])
                job = state["jobs"][repo_id]
                job["state"] = "cancelled"
                state["cancelled"] = True
                _json(
                    route,
                    {
                        "job_key": f"model:{repo_id}:@diffusion",
                        "state": "cancelled",
                    },
                )
                return
            if path == "/api/hub/download-status":
                _json(route, job_status(query["repo_id"][0]))
                return
            if path in {"/api/hub/gguf-download-progress", "/api/hub/download-progress"}:
                _json(route, job_progress(query["repo_id"][0]))
                return
            if path == "/api/inference/images/load" and request.method == "POST":
                state["load_calls"] = int(state["load_calls"]) + 1
                state["load_payloads"].append(payload)
                state["load_progress_polls"] = 0
                _json(route, _status(loaded = False))
                return
            if path == "/api/inference/images/load-progress":
                if int(state["load_calls"]) == 0:
                    progress = {
                        "phase": None,
                        "bytes_downloaded": 0,
                        "bytes_total": 0,
                        "fraction": 0,
                        "error": None,
                    }
                else:
                    state["load_progress_polls"] = int(state["load_progress_polls"]) + 1
                    direct_legacy_load = len(state["starts"]) == 0
                    downloading_inline = (
                        direct_legacy_load and int(state["load_progress_polls"]) < 5
                    )
                    ready = not downloading_inline and int(state["load_progress_polls"]) >= 5
                    if ready:
                        state["loaded"] = True
                    progress = {
                        "phase": (
                            "ready"
                            if ready
                            else "downloading"
                            if downloading_inline
                            else "finalizing"
                        ),
                        "bytes_downloaded": (
                            CHECKPOINT_BYTES if downloading_inline else REQUIRED_BYTES
                        ),
                        "bytes_total": REQUIRED_BYTES,
                        "fraction": (
                            CHECKPOINT_BYTES / REQUIRED_BYTES if downloading_inline else 1
                        ),
                        "error": None,
                    }
                _json(route, progress)
                return
            if path == "/api/inference/images/status":
                _json(route, _status(loaded = bool(state["loaded"])))
                return
            _json(route, _api_payload(path, query, full_footprint = True))

        context.route("**/*", route_request)
        page = context.new_page()
        page.on("pageerror", lambda exc: page_errors.append(str(exc)))

        _open_quant(page, navigate = True)
        if EXPECT == "before":
            deadline = time.monotonic() + 20
            while int(state["load_calls"]) < 1 and time.monotonic() < deadline:
                page.wait_for_timeout(250)
            legacy_toast = page.locator("[data-sonner-toast]").filter(has_text = "Downloading model")
            legacy_toast.wait_for(state = "visible", timeout = 10_000)
            assert "2.6 GB of 11 GB" in legacy_toast.inner_text()
            assert state["starts"] == [], state["starts"]
            assert state["plan_snapshots"] == [], state["plan_snapshots"]
            assert state["load_calls"] == 1
            assert page.get_by_role("button", name = "Cancel download").count() == 0
            assert not page_errors, page_errors
            page.screenshot(
                path = str(ART_DIR / "before-legacy-model-download-toast.png"),
                full_page = True,
            )
            result = {
                "expectation": EXPECT,
                "legacy_toast": True,
                "download_manager_starts": state["starts"],
                "load_calls_while_companion_missing": state["load_calls"],
                "real_cancel_available": False,
            }
            (ART_DIR / "result.json").write_text(json.dumps(result, indent = 2), encoding = "utf-8")
            browser.close()
            print(f"PASS expected before-state; evidence: {ART_DIR.resolve()}")
            return

        expected_initial_starts = (
            [COMPANION_REPO] if CHECKPOINT_CACHED else [REPO_ID, COMPANION_REPO]
        )
        deadline = time.monotonic() + 20
        while len(state["starts"]) < len(expected_initial_starts) and time.monotonic() < deadline:
            page.wait_for_timeout(250)
        if len(state["starts"]) < len(expected_initial_starts):
            page.screenshot(
                path = str(ART_DIR / "image-download-companion-timeout.png"), full_page = True
            )
            print("companion timeout state:", json.dumps(state, default = str, indent = 2))
            print(page.locator("body").inner_text()[:8_000])
        assert state["starts"] == expected_initial_starts, state["starts"]
        companion_row = page.locator(".hub-download-panel").filter(has_text = COMPANION_REPO)
        companion_row.wait_for(state = "visible", timeout = 10_000)
        panel_text = companion_row.inner_text()
        assert "Required assets" in panel_text
        assert "@diffusion" not in panel_text
        if not CHECKPOINT_CACHED:
            assert "Model file" in panel_text
        assert page.get_by_text("Downloading model requirements", exact = False).count() == 0
        assert page.get_by_text("Downloading model…", exact = True).count() == 0
        assert page.get_by_text("Loading to GPU…", exact = True).count() == 0
        assert state["load_calls"] == 0, "missing companion bypassed staging"
        page.screenshot(path = str(ART_DIR / "image-companion-download.png"), full_page = True)

        cancel = page.get_by_role("button", name = "Cancel download")
        assert cancel.count() == 1
        cancel.click()
        page.get_by_text("Cancelled. Partial files kept.", exact = True).wait_for(
            state = "visible", timeout = 10_000
        )
        page.screenshot(path = str(ART_DIR / "image-download-cancelled.png"), full_page = True)
        assert state["cancelled"] is True
        assert state["load_calls"] == 0, "cancelled staging unexpectedly loaded the model"

        _open_quant(page, navigate = False)
        page.locator("[data-sonner-toast]").filter(has_text = "Loading to GPU").wait_for(
            state = "visible", timeout = 20_000
        )
        finalizing_toast = page.locator("[data-sonner-toast]").filter(has_text = "Loading to GPU")
        assert "100%" in finalizing_toast.inner_text()
        page.screenshot(path = str(ART_DIR / "image-download-retry-finalizing.png"), full_page = True)

        page.get_by_text("Model loaded", exact = True).wait_for(state = "visible", timeout = 15_000)
        page.screenshot(path = str(ART_DIR / "image-download-retry-loaded.png"), full_page = True)

        assert state["starts"] == [*expected_initial_starts, COMPANION_REPO], state["starts"]
        assert state["load_calls"] == 1
        assert state["load_payloads"][0]["model_path"] == REPO_ID
        assert state["load_payloads"][0]["model_kind"] == "gguf"
        assert state["load_payloads"][0]["gguf_filename"] == FILENAME
        assert state["loaded"] is True
        assert [COMPANION_REPO] in state["plan_snapshots"]
        if not CHECKPOINT_CACHED:
            assert [REPO_ID, COMPANION_REPO] in state["plan_snapshots"]
        assert not page_errors, page_errors

        result = {
            "expectation": EXPECT,
            "legacy_toast": False,
            "starts": state["starts"],
            "cancelled_before_load": True,
            "checkpoint_redownloaded": state["starts"].count(REPO_ID) > 1,
            "load_calls": state["load_calls"],
            "finalization_progress_polls": state["load_progress_polls"],
            "loaded": state["loaded"],
        }
        (ART_DIR / "result.json").write_text(json.dumps(result, indent = 2), encoding = "utf-8")
        browser.close()

    print(f"PASS rendered image cancel/retry; evidence: {ART_DIR.resolve()}")


if __name__ == "__main__":
    main()
