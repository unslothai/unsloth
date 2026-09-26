# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Exercise encoder precision through the rendered UI without model downloads."""

import json
import os
import re
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from playwright.sync_api import expect, sync_playwright

from playwright_image_model_footprint import BASE_URL, REPO_ID, _api_payload, _json, klein_row

# A refusal: value "off", status "fell_back". The only shape the echoing stub cannot produce.
DECLINE = os.environ.get("PW_DECLINE", "0") == "1"
ART = Path(os.environ.get("PW_ART_DIR", "logs/playwright_image_text_encoder"))
ART.mkdir(parents = True, exist_ok = True)


def _record(page, state, loads, plans, errors):
    """State, requests and a screenshot, so a failed CI run uploads something readable."""
    (ART / "result.json").write_text(
        json.dumps(
            {
                "state": state,
                "loads": loads,
                "plans": plans,
                "page_errors": errors,
                "declined": DECLINE,
            },
            indent = 2,
        ),
        encoding = "utf-8",
    )
    page.screenshot(path = str(ART / "text-encoder.png"), full_page = True)


def main():
    state = {"cached": False, "complete": False, "started": False, "loaded": False}
    plans, loads, errors = [], [], []
    hold_status = False
    held_status = []

    def status():
        result = _api_payload("/api/inference/images/status", {}, full_footprint = True)
        if state["loaded"]:
            requested = loads[-1].get("text_encoder_quant")
            value = "fp8" if requested == "int8" else requested
            if DECLINE and requested:
                value = "off"
            result.update(
                loaded = True,
                repo_id = REPO_ID,
                base_repo = "black-forest-labs/FLUX.2-klein-4B",
                family = "flux.2-klein",
                model_kind = "gguf",
                dtype = "bfloat16",
                text_encoder_quant = value,
                resolved = {
                    "text_encoder_quant": {
                        "value": value,
                        "requested": requested,
                        "source": "explicit" if requested else "auto",
                        "status": (
                            "fell_back"
                            if (requested == "int8" or (DECLINE and requested))
                            else "applied"
                        ),
                        "reason": "Test precision outcome",
                    }
                },
            )
        return result

    with sync_playwright() as playwright:
        engine = getattr(playwright, os.environ.get("PW_BROWSER", "chromium"))
        executable = os.environ.get("PW_EXECUTABLE")
        browser = engine.launch(
            headless = True, **({"executable_path": executable} if executable else {})
        )
        context = browser.new_context(
            viewport = {"width": 1440, "height": 1000}, reduced_motion = "reduce"
        )
        context.add_init_script(
            "localStorage.setItem('unsloth_auth_token', 'rendered-ui-test');"
            "localStorage.setItem('unsloth_download_transport', 'http');"
        )

        def route_request(route):
            parsed = urlparse(route.request.url)
            path, query = parsed.path, parse_qs(parsed.query)
            if not path.startswith("/api/"):
                if parsed.hostname not in ("127.0.0.1", "localhost"):
                    _json(route, [])
                else:
                    route.continue_()
                return
            payload = json.loads(route.request.post_data or "{}")
            if path == "/api/inference/images/download-plan":
                plans.append(payload)
                _json(
                    route,
                    {
                        "entries": []
                        if state["cached"]
                        else [
                            {
                                "repo_id": "test/encoder",
                                "files": ["text_encoder/model.safetensors"],
                                "bytes": 1024,
                                "checkpoint": False,
                            }
                        ],
                        "total_bytes": 1024,
                    },
                )
            elif path == "/api/studio/download-transport-capabilities":
                _json(
                    route,
                    {
                        "http": {"available": True},
                        "xet": {"available": False},
                        "auto_resolves_to": "http",
                    },
                )
            elif path == "/api/hub/transport-status":
                _json(route, {"has_partial": False, "last_transport": None, "resumable": False})
            elif path == "/api/hub/download":
                state["started"] = True
                _json(
                    route,
                    {"accepted": True, "state": "running", "generation": 1, "transport": "http"},
                )
            elif path == "/api/hub/download-status":
                state["cached"] = state["complete"]
                _json(
                    route,
                    {
                        "state": "complete"
                        if state["complete"]
                        else "running"
                        if state["started"]
                        else "idle",
                        "generation": 1,
                    },
                )
            elif path == "/api/hub/download-progress":
                _json(
                    route,
                    {
                        "status": "complete" if state["complete"] else "downloading",
                        "downloaded_bytes": 1024 if state["complete"] else 0,
                        "expected_bytes": 1024,
                        "progress": 1 if state["complete"] else 0,
                    },
                )
            elif path == "/api/inference/images/load":
                loads.append(payload)
                state["loaded"] = True
                _json(route, status())
            elif path == "/api/inference/images/status":
                if hold_status:
                    held_status.append(route)
                else:
                    _json(route, status())
            elif path == "/api/inference/images/load-progress":
                _json(
                    route,
                    {
                        "phase": "ready" if state["loaded"] else None,
                        "bytes_downloaded": 0,
                        "bytes_total": 0,
                        "error": None,
                    },
                )
            else:
                _json(route, _api_payload(path, query, full_footprint = True))

        context.route("**/*", route_request)
        page = context.new_page()
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(f"{BASE_URL}/images", wait_until = "domcontentloaded")
        page.get_by_role("button", name = "Advanced", exact = True).click()
        encoder = page.get_by_role("combobox", name = "Text encoder precision", exact = True)
        expect(encoder).to_have_text("Default")

        def choose(label):
            encoder.click()
            page.get_by_role("option", name = label, exact = True).click()

        choose("FP8 (storage)")
        page.get_by_role("button", name = "Select image model").click()
        klein_row(page).click()
        gguf = page.get_by_text("GGUF", exact = True)
        if gguf.count() == 1:
            gguf.click()
        with page.expect_request(lambda request: urlparse(request.url).path == "/api/hub/download"):
            page.locator("button[data-model-picker-option]").filter(has_text = "Q4_K_M").click()
        assert plans and all(plan.get("text_encoder_quant") == "fp8" for plan in plans), plans
        assert not loads
        choose("INT8")
        state["complete"] = True
        expect(page.get_by_role("button", name = "Reapply", exact = True)).to_be_enabled(timeout = 20_000)
        if DECLINE:
            # The select must show what RAN, or the page advertises a precision nothing is using.
            # A declined scheme runs the dense encoder ("off"), and since #11539 a family default
            # can pick a scheme on its own, so Default no longer means dense: the select shows the
            # opt-out that did run (images-page.tsx maps an engaged "off" to "none", not "auto").
            expect(encoder).to_have_text("Dense (bf16)")
            assert loads[-1]["text_encoder_quant"] == "fp8", loads
            assert not errors, errors
            _record(page, state, loads, plans, errors)
            print(
                f"Passed: a declined encoder precision reseeds to Dense (bf16) ({browser.version})"
            )
            context.close()
            browser.close()
            return
        expect(encoder).to_have_text("FP8 (storage)")
        assert loads[-1]["text_encoder_quant"] == "fp8", loads

        for label, requested, displayed in [
            ("FP8 (compute)", "fp8_dynamic", "FP8 (compute)"),
            ("NVFP4 (Blackwell)", "nvfp4", "NVFP4 (Blackwell)"),
            ("INT8", "int8", "FP8 (storage)"),
            ("Default", None, "Default"),
        ]:
            choose(label)
            with page.expect_request(
                lambda request: urlparse(request.url).path == "/api/inference/images/load"
            ):
                page.get_by_role("button", name = "Reapply", exact = True).click()
            expect(page.get_by_role("button", name = "Reapply", exact = True)).to_be_enabled()
            expect(encoder).to_have_text(displayed)
            assert loads[-1].get("text_encoder_quant") == requested, loads[-1]
            if requested is None:
                assert "text_encoder_quant" not in loads[-1]
        # A completed reload can have the same precision record as its predecessor.
        state.update(cached = False, complete = False, started = False)
        page.get_by_role("button", name = "Reapply", exact = True).scroll_into_view_if_needed()
        page.locator(".unsloth-model-selector-trigger:visible").click()
        klein_row(page).click()
        gguf = page.get_by_text("GGUF", exact = True)
        if gguf.count() == 1:
            gguf.click()
        with page.expect_request(lambda request: urlparse(request.url).path == "/api/hub/download"):
            page.locator("button[data-model-picker-option]").filter(has_text = "Q4_K_M").click()
        choose("FP8 (storage)")
        hold_status = True
        with page.expect_request(
            lambda request: urlparse(request.url).path == "/api/inference/images/load"
        ):
            state["complete"] = True
        for _ in range(200):
            if held_status:
                break
            page.wait_for_timeout(50)
        assert len(held_status) == 1
        page.get_by_test_id("nav-row-hub").click()
        # Path only: the Hub appends its own ?tab= on mount and an exact-URL assertion loses that race.
        expect(page).to_have_url(re.compile(r"/hub(\?|$)"))
        page.get_by_test_id("nav-row-images").click()
        expect(page).to_have_url(re.compile(r"/images(\?|$)"))
        for _ in range(200):
            if len(held_status) >= 2:
                break
            page.wait_for_timeout(50)
        assert len(held_status) == 2
        hold_status = False
        replies = (
            held_status if os.environ.get("PW_COMPLETION_FIRST") == "1" else reversed(held_status)
        )
        for response in replies:
            _json(response, status())
        held_status.clear()
        # The staged load carries the precision pinned when it was queued, so this one is Default.
        assert "text_encoder_quant" not in loads[-1], loads[-1]
        # Either order ends the same, and the edit made while the load staged survives: the reseed
        # follows a change of BUILD, not every completed load, and Reapply is how the user applies it.
        expect(encoder).to_have_text("FP8 (storage)")
        expect(page.get_by_role("button", name = "Reapply", exact = True)).to_be_enabled()
        assert not errors, errors
        _record(page, state, loads, plans, errors)
        print(
            f"Passed: planned and pinned precision, four explicit modes, fallback reseeding and default omission ({browser.version})"
        )
        context.close()
        browser.close()


if __name__ == "__main__":
    main()
