# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Rendered regression for the image-model selector's full disk footprint.

The browser runs the real Vite application. Network data is deterministic so
the test proves the UI contract without downloading a model or depending on
live Hugging Face metadata.
"""

import os
import re
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from playwright.sync_api import Route, sync_playwright


BASE_URL = os.environ.get("BASE_URL", "http://127.0.0.1:4173")
ART_DIR = Path(os.environ.get("PW_ART_DIR", "logs/playwright_image_footprint"))
ART_DIR.mkdir(parents = True, exist_ok = True)

CHECKPOINT_BYTES = 2_600_000_000
COMPANION_BYTES = 8_200_000_000
REQUIRED_BYTES = CHECKPOINT_BYTES + COMPANION_BYTES
REPO_ID = "unsloth/FLUX.2-klein-4B-GGUF"
FILENAME = "FLUX.2-klein-4B-Q4_K_M.gguf"
# The row is labelled by the catalogue's displayName ("FLUX.2 klein 4B"), not by the artifact repo id, so matching the
KLEIN_ROW = re.compile(r"FLUX\.2[\s\-]klein[\s\-]4B")


def _json(route: Route, payload: object) -> None:
    route.fulfill(status = 200, content_type = "application/json", json = payload)


def _api_payload(path: str, query: dict[str, list[str]], *, full_footprint: bool) -> object:
    if path == "/api/auth/status":
        return {"initialized": True, "requires_password_change": False}
    if path == "/api/health":
        return {
            "version": "ui-test",
            "device_type": "windows",
            "chat_only": False,
            "hardware_detecting": False,
        }
    if path == "/api/system":
        device = {
            "index": 0,
            "index_kind": "physical",
            "name": "Rendered test GPU",
            "memory_total_gb": 24,
            "vram_free_gb": 22,
        }
        return {
            "platform": "Windows",
            "python_version": "3.13",
            "device_backend": "cuda",
            "uptime_seconds": 1,
            "cpu": {
                "logical_count": 16,
                "physical_count": 8,
                "usage_percent": 1,
                "frequency_mhz": 4000,
            },
            "memory": {
                "total_gb": 64,
                "available_gb": 48,
                "percent_used": 25,
                "process_used_mb": 512,
            },
            "disk": {"total_gb": 1000, "free_gb": 800, "percent_used": 20},
            "gpu": {"available": True, "backend": "cuda", "devices": [device]},
            "inference_gpu": {
                "available": True,
                "backend": "cuda",
                "devices": [device],
            },
            "ml_packages": {},
        }
    if path == "/api/inference/images/status":
        return {
            "loaded": False,
            "repo_id": None,
            "family": None,
            "base_repo": None,
            "device": None,
            "dtype": None,
            "model_kind": None,
            "workflows": [],
        }
    if path == "/api/inference/images/load-progress":
        return {"phase": None, "bytes_downloaded": 0, "bytes_total": 0, "error": None}
    if path == "/api/inference/images/generate-progress":
        return {"active": False, "step": 0, "total_steps": 0, "eta_seconds": None}
    if path == "/api/inference/images/info":
        return {"families": []}
    if path == "/api/inference/monitor":
        return {
            "status": "idle",
            "active_model": None,
            "active_requests": 0,
            "entries": [],
        }
    if path == "/api/inference/images/gallery":
        return {"images": [], "has_more": False}
    if path == "/api/models/diffusion-loras":
        return {"loras": []}
    if path == "/api/models/diffusion-controlnets":
        return {"controlnets": []}
    if path == "/api/hub/local":
        return {"models_dir": "C:\\models", "lmstudio_dirs": [], "models": []}
    if path in {"/api/hub/cached-gguf", "/api/hub/cached-models"}:
        return {"cached": []}
    if path == "/api/hub/hidden-models":
        return {"patterns": []}
    if path == "/api/hub/active-downloads":
        return {"downloads": []}
    if path == "/api/hub/datasets/active-downloads":
        return {"downloads": []}
    if path == "/api/chat/threads":
        return {"threads": []}
    if path == "/api/chat/projects":
        return {"projects": []}
    if path == "/api/settings/personalization":
        return {
            "version": 1,
            "profile": {
                "displayName": "",
                "nickname": "",
                "avatarDataUrl": None,
                "avatarShape": "circle",
                "showGreetingSloth": True,
            },
            "appearance": {
                "theme": "dark",
                "palette": "standard",
                "language": None,
                "customization": {},
            },
            "saved": False,
            "customizationSaved": False,
            "paletteSaved": False,
            "greetingSlothSaved": False,
        }
    if path == "/api/export/status":
        return {
            "current_checkpoint": None,
            "is_vision": False,
            "is_peft": False,
            "is_export_active": False,
        }
    if path in {"/api/hub/gguf-variants", "/api/models/gguf-variants"}:
        assert query.get("repo_id") == [REPO_ID], query
        return {
            "repo_id": REPO_ID,
            "variants": [
                {
                    "filename": FILENAME,
                    "quant": "Q4_K_M",
                    "size_bytes": CHECKPOINT_BYTES,
                    "download_size_bytes": CHECKPOINT_BYTES,
                    "downloaded": False,
                }
            ],
            "has_vision": False,
            "default_variant": "Q4_K_M",
            "context_length": None,
        }
    if path == "/api/inference/images/download-plan":
        base = {"entries": [], "total_bytes": REQUIRED_BYTES}
        if full_footprint:
            base.update(
                required_bytes = REQUIRED_BYTES,
                checkpoint_bytes = CHECKPOINT_BYTES,
            )
        return base
    # Nonessential background probes are allowed to settle to an empty object;
    return {}


def klein_row(page):
    """The klein row inside the open picker, and nothing else on the page.

    An unscoped search is not specific enough once a download has been started:
    the hub download panel labels itself "black-forest-labs/FLUX.2-klein-4B ·
    Required assets", so a page-wide match returns it first and clicking it
    leaves the picker where it was. The old exact repo-id text never matched
    that panel, so scoping only became necessary with the pattern.
    """
    return page.locator(".unsloth-model-selector-menu").get_by_text(KLEIN_ROW).first


def _open_klein_quant(page) -> None:
    page.goto(f"{BASE_URL}/images", wait_until = "domcontentloaded")
    trigger = page.get_by_role("button", name = "Select image model")
    try:
        trigger.wait_for(state = "visible", timeout = 30_000)
    except Exception:
        print(f"selector startup URL: {page.url}")
        print(page.locator("body").inner_text()[:4_000])
        raise
    trigger.click()
    klein = klein_row(page)
    try:
        klein.wait_for(state = "visible", timeout = 30_000)
    except Exception:
        print(page.locator("body").inner_text()[:8_000])
        raise
    klein.click()
    gguf = page.get_by_text("GGUF", exact = True)
    if gguf.count() == 1:
        gguf.click()
    page.get_by_text("Q4_K_M", exact = True).wait_for(state = "visible")


def main() -> None:
    full_footprint = {"enabled": False}
    page_errors: list[str] = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless = True)
        context = browser.new_context(
            viewport = {"width": 1440, "height": 900},
            reduced_motion = "reduce",
            color_scheme = "dark",
        )
        context.add_init_script("localStorage.setItem('unsloth_auth_token', 'rendered-ui-test');")

        def route_request(route: Route) -> None:
            parsed = urlparse(route.request.url)
            if parsed.netloc == "huggingface.co":
                _json(route, [])
                return
            if parsed.path.startswith("/api/"):
                _json(
                    route,
                    _api_payload(
                        parsed.path,
                        parse_qs(parsed.query),
                        full_footprint = full_footprint["enabled"],
                    ),
                )
                return
            route.continue_()

        context.route("**/*", route_request)
        page = context.new_page()
        page.on("pageerror", lambda exc: page_errors.append(str(exc)))

        _open_klein_quant(page)
        quant_row = page.locator("button").filter(has_text = "Q4_K_M")
        assert quant_row.count() == 1
        assert "2.6GB" in "".join(quant_row.inner_text().split())
        assert page.locator("[data-model-download-footprint]").count() == 0
        picker = page.locator(".unsloth-model-selector-menu")
        assert picker.count() == 1
        picker.screenshot(path = str(ART_DIR / "image-model-footprint-before.png"))
        page.screenshot(path = str(ART_DIR / "image-model-footprint-before-full.png"), full_page = True)

        full_footprint["enabled"] = True
        _open_klein_quant(page)
        footprint = page.locator("[data-model-download-footprint]")
        footprint.wait_for(state = "visible")
        assert footprint.count() == 1
        footprint_text = "".join(footprint.inner_text().split())
        assert footprint_text == "10.8GB"
        picker = page.locator(".unsloth-model-selector-menu")
        assert picker.count() == 1
        picker.screenshot(path = str(ART_DIR / "image-model-footprint-after.png"))
        page.screenshot(path = str(ART_DIR / "image-model-footprint-after-full.png"), full_page = True)
        help_icon = page.locator("[data-model-download-footprint-help]")
        assert help_icon.count() == 1
        help_icon.hover()
        explanation = page.get_by_role("tooltip")
        explanation.wait_for(state = "visible")
        explanation_text = " ".join(explanation.inner_text().split())
        assert "Full required size" in explanation_text
        assert "2.6GBmodel+8.2GBrequiredassets" in "".join(explanation_text.split())
        page.screenshot(path = str(ART_DIR / "image-model-footprint-hover.png"), full_page = True)

        assert not page_errors, page_errors
        browser.close()

    print(f"PASS rendered image footprint; screenshots: {ART_DIR.resolve()}")


if __name__ == "__main__":
    main()
