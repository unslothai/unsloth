# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Isolated browser tests for the production Markdown renderer."""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
from pathlib import Path
import subprocess
import time

from PIL import Image

from _playwright_robust import stop_process

REPO = Path(__file__).resolve().parents[2]


def fixtures(output: Path) -> None:
    images = {}
    formats = {
        "png": ("PNG", "image/png"),
        "jpg": ("JPEG", "image/jpeg"),
        "jpeg": ("JPEG", "image/jpeg"),
        "gif": ("GIF", "image/gif"),
        "webp": ("WEBP", "image/webp"),
        "bmp": ("BMP", "image/bmp"),
        "avif": ("AVIF", "image/avif"),
    }
    for width in [32, 48, 64, 80, 96]:
        image = Image.new("RGB", (width, 24), (width * 2, 90, 180))
        images[str(width)] = {}
        for ext, (codec, content_type) in formats.items():
            data = io.BytesIO()
            image.save(data, codec)
            images[str(width)][ext] = {
                "type": content_type,
                "data": base64.b64encode(data.getvalue()).decode(),
            }
    (output / "images.json").write_text(json.dumps(images), encoding = "utf-8")


def launch_browser(playwright, name):
    if name in ("chrome", "msedge"):
        return playwright.chromium.launch(headless = True, channel = name)
    return getattr(playwright, name).launch(headless = True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type = Path, default = REPO / "temp/inline-image-validation")
    parser.add_argument(
        "--browsers",
        nargs = "+",
        choices = ["chromium", "firefox", "webkit", "chrome", "msedge"],
        default = ["chromium", "firefox", "webkit"],
    )
    parser.add_argument("--probe", action = "store_true")
    parser.add_argument("--baseline")
    parser.add_argument("--manual", action = "store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(REPO):
        parser.error("Output must stay inside the repository")
    output.mkdir(parents = True, exist_ok = True)
    # Keep Chromium's Unix socket path below the platform limit.
    runtime = REPO / "temp" / "p"
    runtime.mkdir(parents = True, exist_ok = True)
    os.environ.update({key: str(runtime) for key in ("TMPDIR", "TMP", "TEMP")})
    if args.probe:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as playwright:
            for name in args.browsers:
                browser = launch_browser(playwright, name)
                browser.close()
        return
    fixtures(output)
    ready = output / "ready.json"
    ready.unlink(missing_ok = True)
    command = ["node", str(REPO / "tests/studio/fixtures/inline-image-server.mjs"), str(output)]
    if args.baseline:
        command.append(args.baseline)
    with (output / "server.log").open("w", encoding = "utf-8") as log:
        server = subprocess.Popen(
            command,
            cwd = REPO,
            stdout = log,
            stderr = subprocess.STDOUT,
            start_new_session = os.name != "nt",
        )
        try:
            deadline = time.monotonic() + 60
            while not ready.exists():
                if server.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError((output / "server.log").read_text(encoding = "utf-8"))
                time.sleep(0.1)
            url = json.loads(ready.read_text(encoding = "utf-8"))["url"]
            print(url, flush = True)
            if args.manual:
                server.wait()
                return
            from playwright.sync_api import expect, sync_playwright

            reports = []
            with sync_playwright() as playwright:
                for name in args.browsers:
                    browser = None
                    page = None
                    report = {"browser": name, "platform": sys.platform, "failed": 1, "checks": []}
                    errors = []
                    try:
                        browser = launch_browser(playwright, name)
                        page = browser.new_page(
                            viewport = {"width": 1280, "height": 800}, device_scale_factor = 1
                        )
                        page.on("pageerror", lambda error: errors.append(str(error)))
                        page.set_default_timeout(30000)
                        page.goto(url, wait_until = "networkidle")
                        page.get_by_role("button", name = "Run simulations").click()
                        page.locator('#report[data-complete="true"]').wait_for(
                            state = "attached", timeout = 180000
                        )
                        report.update(json.loads(page.locator("#report").text_content()))
                        report["browser_version"] = browser.version
                        if not args.baseline and report["failed"] == 0:
                            downloads = []
                            for index, (button, filename) in enumerate(
                                [
                                    (None, "plot.png"),
                                    ("Show embedded download", "Embedded.png"),
                                    ("Show encoded download", "loss curve #1.png"),
                                ]
                            ):
                                if button:
                                    page.get_by_role("button", name = button).click()
                                ready_image = page.locator('#subject img[data-streamdown="image"]')
                                ready_image.wait_for(state = "visible")
                                expect(ready_image).to_have_js_property("naturalWidth", 32)
                                previous = page.request.get(
                                    url.replace("/inline-images", "/inline-fixture/requests")
                                ).json()
                                with page.expect_download(timeout = 10000) as downloaded:
                                    page.get_by_role("button", name = "Download image").click()
                                download = downloaded.value
                                assert download.suggested_filename == filename
                                destination = output / f"{name}-download-{index}.png"
                                download.save_as(destination)
                                with Image.open(destination) as saved:
                                    assert saved.size == (32, 24)
                                current = page.request.get(
                                    url.replace("/inline-images", "/inline-fixture/requests")
                                ).json()
                                assert (
                                    current == previous
                                ), "Download fetched the sandbox image again"
                                downloads.append(filename)
                            report["downloads"] = downloads
                        report["csp_violations"] = page.locator("#errors").text_content()
                        assert not report["csp_violations"], report["csp_violations"]
                        assert not errors, errors
                    except Exception as error:
                        report["failed"] = max(1, report["failed"])
                        report["error"] = str(error)
                    finally:
                        report["page_errors"] = errors
                        reports.append(report)
                        (output / f"{name}.json").write_text(
                            json.dumps(report, indent = 2), encoding = "utf-8"
                        )
                        print(json.dumps(report), flush = True)
                        if report["failed"] and page:
                            page.screenshot(
                                path = str(output / f"{name}-failure.png"), full_page = True
                            )
                        if browser:
                            browser.close()
            if args.baseline:
                assert all(
                    any(not c["passed"] and c["name"] == "path: plot.png" for c in r["checks"])
                    and any(c["passed"] and c["name"] == "recorded scope" for c in r["checks"])
                    for r in reports
                ), "Baseline controls did not reproduce the bug"
            else:
                assert all(
                    r["failed"] == 0 and not r["page_errors"] for r in reports
                ), "Browser simulation failed"
        finally:
            stop_process(server)


if __name__ == "__main__":
    main()
