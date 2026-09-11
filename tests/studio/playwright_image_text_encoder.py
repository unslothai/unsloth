# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Exercise encoder precision through the rendered UI without model downloads."""

import json
import os
from urllib.parse import parse_qs, urlparse

from playwright.sync_api import expect, sync_playwright

from playwright_image_model_footprint import BASE_URL, REPO_ID, _api_payload, _json, klein_row


def main():
    state = {"cached": False, "complete": False, "started": False, "loaded": False}
    plans, loads, errors = [], [], []

    def status():
        result = _api_payload("/api/inference/images/status", {}, full_footprint = True)
        if state["loaded"]:
            requested = loads[-1].get("text_encoder_quant")
            value = "fp8" if requested == "int8" else requested
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
                        "status": "fell_back" if requested == "int8" else "applied",
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
        expect(page.get_by_role("button", name = "Reapply to loaded model")).to_be_enabled(
            timeout = 20_000
        )
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
                page.get_by_role("button", name = "Reapply to loaded model").click()
            expect(page.get_by_role("button", name = "Reapply to loaded model")).to_be_enabled()
            expect(encoder).to_have_text(displayed)
            assert loads[-1].get("text_encoder_quant") == requested, loads[-1]
            if requested is None:
                assert "text_encoder_quant" not in loads[-1]
        # A completed reload can have the same precision record as its predecessor.
        state.update(cached = False, complete = False, started = False)
        page.get_by_role("button", name = "Reapply to loaded model").scroll_into_view_if_needed()
        page.locator(".unsloth-model-selector-trigger:visible").click()
        klein_row(page).click()
        gguf = page.get_by_text("GGUF", exact = True)
        if gguf.count() == 1:
            gguf.click()
        with page.expect_request(lambda request: urlparse(request.url).path == "/api/hub/download"):
            page.locator("button[data-model-picker-option]").filter(has_text = "Q4_K_M").click()
        choose("FP8 (storage)")
        with page.expect_request(
            lambda request: urlparse(request.url).path == "/api/inference/images/load"
        ):
            state["complete"] = True
        expect(encoder).to_have_text("Default")
        assert "text_encoder_quant" not in loads[-1], loads[-1]
        assert not errors, errors
        print(
            f"Passed: planned and pinned precision, four explicit modes, fallback reseeding and default omission ({browser.version})"
        )
        context.close()
        browser.close()


if __name__ == "__main__":
    main()
