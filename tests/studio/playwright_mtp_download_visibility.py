# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A/B browser proof that Flash Next's missing MTP is a managed download.

Run this identical driver once with ``PW_REPO_DIR`` at the base checkout and
once at the fix checkout. The backend under test evaluates a real seeded HF
cache; the browser then runs the real picker predicate and Downloads panel.
No model bytes or GPU are required.
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from urllib.error import URLError
from urllib.parse import parse_qs, urlparse
from urllib.request import urlopen


REPO_ID = "unsloth/Qwen3.8-Flash-Next-GGUF"
QUANT = "UD-Q4_K_XL"
MAIN_FILENAME = "Qwen3.8-Flash-Next-UD-Q4_K_XL.gguf"
MTP_FILENAME = "MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf"
MAIN_BYTES = 112_238_658_784
MTP_BYTES = 2_786_568_256
TOTAL_BYTES = MAIN_BYTES + MTP_BYTES

REPO_DIR = Path(os.environ.get("PW_REPO_DIR", Path(__file__).parents[2])).resolve()
FRONTEND_DIR = REPO_DIR / "studio" / "frontend"
HARNESS_FRONTEND_DIR = Path(__file__).parents[2].resolve() / "studio" / "frontend"
ART_DIR = Path(os.environ.get("PW_ART_DIR", "logs/playwright_mtp_download_visibility"))
SIDE = os.environ.get("PW_SIDE", "FIX").upper()


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_vite(url: str, process: subprocess.Popen[str]) -> None:
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Vite exited early with {process.returncode}")
        try:
            with urlopen(url, timeout = 1) as response:
                if response.status == 200:
                    return
        except (OSError, URLError):
            time.sleep(0.1)
    raise RuntimeError(f"Timed out waiting for {url}")


def _sibling(name: str, size: int, sha: str) -> SimpleNamespace:
    return SimpleNamespace(rfilename = name, size = size, lfs = {"sha256": sha})


def _backend_payload_here() -> dict[str, object]:
    """Ask this checkout's production inventory code about a main-only cache."""
    sys.path.insert(0, str(REPO_DIR / "studio" / "backend"))
    from hub.services.models import gguf_variants
    from hub.utils import state_dir

    async def _run_inline(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    with tempfile.TemporaryDirectory(prefix = "mtp-download-ui-") as tmp:
        root = Path(tmp)
        snapshot = (
            root / "cache" / "models--unsloth--Qwen3.8-Flash-Next-GGUF" / "snapshots" / "rev0"
        )
        snapshot.mkdir(parents = True)
        with (snapshot / MAIN_FILENAME).open("wb") as cached_main:
            cached_main.truncate(MAIN_BYTES)

        gguf_variants.asyncio.to_thread = _run_inline
        gguf_variants.list_gguf_variants = lambda *_args, **_kwargs: (
            [
                SimpleNamespace(
                    filename = MAIN_FILENAME,
                    quant = QUANT,
                    display_label = None,
                    size_bytes = MAIN_BYTES,
                )
            ],
            False,
            [
                _sibling(MAIN_FILENAME, MAIN_BYTES, "main"),
                _sibling(MTP_FILENAME, MTP_BYTES, "mtp"),
            ],
        )
        gguf_variants.iter_hf_cache_snapshots = lambda *_args, **_kwargs: [snapshot]
        gguf_variants.download_registry.incomplete_blob_hashes = lambda *_args, **_kwargs: set()
        state_dir.cache_root = lambda: root / "state"
        response = asyncio.run(gguf_variants.get_gguf_variants_response(REPO_ID))
        return response.model_dump(mode = "json")


def _backend_payload() -> dict[str, object]:
    backend_python = os.environ.get(
        "PW_BACKEND_PYTHON",
        "/home/samle/.unsloth/studio/unsloth_studio/bin/python",
    )
    output = subprocess.check_output(
        [backend_python, str(Path(__file__).resolve()), "--backend-payload"],
        env = {**os.environ, "PW_REPO_DIR": str(REPO_DIR)},
        text = True,
    )
    return json.loads(output.strip().splitlines()[-1])


def _api_payload(path: str, query: dict[str, list[str]], variants: dict[str, object]):
    if path == "/api/models/gguf-variants":
        assert query.get("repo_id") == [REPO_ID], query
        return variants
    if path == "/api/studio/download-transport-capabilities":
        return {
            "http": {"available": True, "reason": None},
            "xet": {"available": False, "reason": "Deterministic UI fixture"},
            "auto_resolves_to": "http",
            "auto_reason": "Deterministic UI fixture",
            "partials_resumable": True,
        }
    if path == "/api/hub/transport-status":
        return {"has_partial": False, "last_transport": None, "resumable": True}
    if path == "/api/hub/active-downloads":
        return {"downloads": []}
    if path == "/api/hub/datasets/active-downloads":
        return {"downloads": []}
    if path == "/api/hub/download":
        return {"state": "running", "accepted": True, "generation": 1}
    if path == "/api/hub/download-status":
        return {"state": "running", "generation": 1}
    if path == "/api/hub/gguf-download-progress":
        return {
            "downloaded_bytes": MAIN_BYTES + 540_000_000,
            "completed_bytes": MAIN_BYTES,
            "complete_on_disk": False,
            "expected_bytes": TOTAL_BYTES,
            "progress": (MAIN_BYTES + 540_000_000) / TOTAL_BYTES,
            "cache_path": "/fixture/huggingface/cache",
            "target_present": True,
            "cache_measured": True,
        }
    return {}


def main() -> None:
    from PIL import Image, ImageDraw, ImageFont
    from playwright.sync_api import Route, sync_playwright

    ART_DIR.mkdir(parents = True, exist_ok = True)
    variants = _backend_payload()
    variant = variants["variants"][0]
    port = _free_port()
    base_url = f"http://127.0.0.1:{port}"
    vite_log = (ART_DIR / f"{SIDE.lower()}-vite.log").open("w", encoding = "utf-8")
    vite = subprocess.Popen(
        [
            "npm",
            "run",
            "dev",
            "--",
            "--config",
            "vite.mtp-download.config.ts",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--strictPort",
        ],
        cwd = HARNESS_FRONTEND_DIR,
        env = {**os.environ, "PW_SOURCE_FRONTEND_DIR": str(FRONTEND_DIR)},
        stdout = vite_log,
        stderr = subprocess.STDOUT,
        text = True,
    )
    page_errors: list[str] = []
    try:
        _wait_for_vite(f"{base_url}/smoke-mtp-download.html", vite)
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless = True)
            context = browser.new_context(
                viewport = {"width": 1440, "height": 900},
                reduced_motion = "reduce",
                color_scheme = "dark",
            )

            def route_request(route: Route) -> None:
                parsed = urlparse(route.request.url)
                if parsed.path.startswith("/api/"):
                    route.fulfill(
                        status = 200,
                        content_type = "application/json",
                        json = _api_payload(parsed.path, parse_qs(parsed.query), variants),
                    )
                    return
                route.continue_()

            context.route("**/*", route_request)
            page = context.new_page()
            page.on("pageerror", lambda exc: page_errors.append(str(exc)))
            page.goto(
                f"{base_url}/smoke-mtp-download.html?side={SIDE}",
                wait_until = "networkidle",
            )
            page.wait_for_function("window.__mtpDownloadSmoke !== undefined")
            result = page.evaluate("window.__mtpDownloadSmoke.reproduce()")

            if SIDE == "BEFORE":
                assert result == {
                    "backendDownloaded": True,
                    "staged": False,
                    "outcome": None,
                }, result
                assert page.get_by_text("Downloading 1 item", exact = True).count() == 0
                page.get_by_role("button", name = "Downloads", exact = True).wait_for()
            else:
                assert result == {
                    "backendDownloaded": False,
                    "staged": True,
                    "outcome": "started",
                }, result
                page.get_by_text("Downloading 1 item", exact = True).wait_for()
                panel = page.locator(".hub-download-panel")
                panel.get_by_text("MTP companion", exact = False).wait_for()
                panel.get_by_text(MTP_FILENAME.rsplit("/", 1)[-1], exact = True).wait_for()
                panel.get_by_text("540 MB / 2.8 GB", exact = True).wait_for()
                jobs = page.evaluate("window.__mtpDownloadSmoke.jobs()")
                assert len(jobs["jobs"]) == 1, jobs

            assert not page_errors, page_errors
            raw_path = ART_DIR / f"{SIDE.lower()}.png"
            # The effect lives in the bottom-right overlay. A fixed clip keeps
            # its filename and byte counters legible in a PR comment instead
            # of shrinking them into a mostly empty full-page comparison.
            page.screenshot(
                path = str(raw_path),
                clip = {"x": 740, "y": 520, "width": 700, "height": 380},
            )

            raw = Image.open(raw_path).convert("RGB")
            labelled = Image.new("RGB", (raw.width, raw.height + 64), "#101114")
            labelled.paste(raw, (0, 64))
            draw = ImageDraw.Draw(labelled)
            font = ImageFont.load_default(size = 24)
            label = (
                "BEFORE - no active MTP download is shown"
                if SIDE == "BEFORE"
                else "AFTER - MTP file and its own progress are explicit"
            )
            draw.text((24, 20), label, fill = "white", font = font)
            labelled.save(ART_DIR / f"{SIDE.lower()}-labelled.png")

            facts = {
                "side": SIDE,
                "repo_dir": str(REPO_DIR),
                "backend_downloaded": variant["downloaded"],
                "download_size_bytes": variant["download_size_bytes"],
                "pending_drafter_filename": variant.get("pending_drafter_filename"),
                "pending_drafter_size_bytes": variant.get("pending_drafter_size_bytes", 0),
                "main_bytes_cached": MAIN_BYTES,
                "mtp_bytes_missing": MTP_BYTES,
                "staged": result["staged"],
                "manager_outcome": result["outcome"],
                "page_errors": page_errors,
            }
            (ART_DIR / f"{SIDE.lower()}-facts.json").write_text(
                json.dumps(facts, indent = 2) + "\n",
                encoding = "utf-8",
            )
            browser.close()
    finally:
        vite.terminate()
        try:
            vite.wait(timeout = 5)
        except subprocess.TimeoutExpired:
            vite.kill()
            vite.wait(timeout = 5)
        vite_log.close()

    print(json.dumps({"side": SIDE, "variant": variant, "artifacts": str(ART_DIR)}))


if __name__ == "__main__":
    if "--backend-payload" in sys.argv:
        print(json.dumps(_backend_payload_here()))
    else:
        main()
