# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Compare built before/after Studio settings with real backends and no route mocks.

Requires Python with playwright==1.56.0, httpx and pillow, installed browser engines,
and a backend Python with Studio dependencies. Build each checkout with npm run build.
Use fresh --output; private homes/logs stay outside its public evidence subdirectory.
No model is loaded: this proves settings UI/persistence, not hardware prefill speed.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import secrets
import signal
import socket
import subprocess
import time
from contextlib import contextmanager

import httpx
from PIL import Image, ImageDraw, ImageFont
from playwright.sync_api import expect, sync_playwright

ROUTE = "/api/settings/first-token-timeout"
LABEL = "First-token waiting limit"
EXPECT = "BEFORE has no first-token setting; AFTER saves custom minutes and Unlimited to the backend and retains them on reload/restart."


def revision(root):
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd = root, text = True).strip()


@contextmanager
def server(root, home, backend_python, logs):
    home.mkdir(parents = True, exist_ok = True)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    env = os.environ.copy()
    env.update(
        {
            "UNSLOTH_STUDIO_HOME": str(home),
            "HF_HOME": str(home / "hf"),
            "HF_HUB_CACHE": str(home / "hf/hub"),
            "HF_XET_CACHE": str(home / "hf/xet"),
            "XDG_CACHE_HOME": str(home / "cache"),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "UNSLOTH_STUDIO_NO_BROWSER": "1",
        }
    )
    env.pop("UNSLOTH_OPENAI_COMPAT_FIRST_TOKEN_TIMEOUT", None)
    command = [
        str(backend_python),
        str(root / "studio/backend/run.py"),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--frontend",
        str(root / "studio/frontend/dist"),
        "--no-cloudflare",
        "--silent",
    ]
    with logs.open("w") as log:
        proc = subprocess.Popen(
            command,
            cwd = root / "studio/backend",
            env = env,
            stdout = log,
            stderr = subprocess.STDOUT,
            start_new_session = True,
        )
    url = f"http://127.0.0.1:{port}"
    try:
        for _ in range(180):
            if proc.poll() is not None:
                raise RuntimeError(f"Backend stopped; inspect {logs}")
            try:
                if httpx.get(url + "/api/health", timeout = 1).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            time.sleep(0.5)
        else:
            raise RuntimeError(f"Backend health timeout; inspect {logs}")
        yield {
            "url": url,
            "pid": proc.pid,
            "port": port,
            "home": str(home),
            "commit": revision(root),
            "frontend_sha256": hashlib.sha256(
                (root / "studio/frontend/dist/index.html").read_bytes()
            ).hexdigest(),
        }
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout = 15)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout = 5)


def authenticate(state):
    home = Path(state["home"])
    private = home / ".test-auth.json"
    password = (
        json.loads(private.read_text())["password"]
        if private.exists()
        else (home / "auth/.bootstrap_password").read_text().strip()
    )
    api = httpx.Client(base_url = state["url"], timeout = 30)
    response = api.post("/api/auth/login", json = {"username": "unsloth", "password": password})
    response.raise_for_status()
    auth = response.json()
    api.headers["Authorization"] = "Bearer " + auth["access_token"]
    if auth.get("must_change_password"):
        new_password = secrets.token_urlsafe(32)
        response = api.post(
            "/api/auth/change-password",
            json = {
                "current_password": password,
                "new_password": new_password,
            },
        )
        response.raise_for_status()
        password, auth = new_password, response.json()
        api.headers["Authorization"] = "Bearer " + auth["access_token"]
    private.write_text(json.dumps({"password": password}))
    private.chmod(0o600)
    return api, auth


def open_chat_settings(page, url):
    page.goto(url + "/chat", wait_until = "domcontentloaded")
    page.get_by_role("button", name = "Settings", exact = True).click(timeout = 60000)
    page.get_by_test_id("settings-tab-chat").click()
    notice = page.get_by_role("button", name = "Remind me later", exact = True)
    if notice.is_visible():
        notice.click()
    expect(page.get_by_text("Tool output length", exact = True)).to_be_visible(timeout = 30000)


def scene(playwright, engine, label, state, api, auth, evidence):
    if label == "after":
        api.put(ROUTE, json = {"minutes": 20}).raise_for_status()
    browser = getattr(playwright, engine).launch(headless = True)
    context = browser.new_context(viewport = {"width": 1440, "height": 1080}, color_scheme = "dark")
    context.add_init_script(
        "const auth = "
        + json.dumps(auth)
        + ";"
        + "localStorage.setItem('unsloth_auth_token', auth.access_token);"
        + "localStorage.setItem('unsloth_refresh_token', auth.refresh_token);"
        + "localStorage.setItem('theme', 'dark');"
    )
    page = context.new_page()
    page.set_default_timeout(20000)
    errors = []
    page.on("pageerror", lambda err: errors.append(str(err)))
    facts = {"browser": engine, "label": label}
    try:
        open_chat_settings(page, state["url"])
        control = page.get_by_role("combobox", name = LABEL, exact = True)
        if label == "before":
            expect(control).to_have_count(0)
            facts.update(setting_count = 0, api_status = api.get(ROUTE).status_code)
            assert facts["api_status"] == 404
            page.get_by_text("Tool output length", exact = True).scroll_into_view_if_needed()
        else:
            expect(control).to_be_enabled()
            expect(control).to_have_text("20 min (default)")
            control.click()
            page.get_by_role("option", name = "Custom", exact = True).click()
            field = page.get_by_role("spinbutton", name = "First-token wait in minutes", exact = True)
            save = page.get_by_role("button", name = "Save", exact = True)
            for invalid in ("", "0", "-1", "1.5", "10081"):
                field.fill(invalid)
                expect(save).to_be_disabled()
            field.fill("60")
            with page.expect_response(
                lambda res: res.url.endswith(ROUTE) and res.request.method == "PUT"
            ) as response:
                save.click()
            assert response.value.status == 200
            expect(save).to_be_disabled()
            assert api.get(ROUTE).json()["minutes"] == 60
            page.get_by_role("dialog").screenshot(path = str(evidence / f"{engine}-after-custom.png"))
            page.reload(wait_until = "domcontentloaded")
            open_chat_settings(page, state["url"])
            expect(field).to_have_value("60")
            control.click()
            page.get_by_role("option", name = "Unlimited", exact = True).click()
            save.click()
            expect(save).to_be_disabled()
            assert api.get(ROUTE).json()["minutes"] == 0
            page.reload(wait_until = "domcontentloaded")
            open_chat_settings(page, state["url"])
            expect(control).to_have_text("Unlimited")
            control.scroll_into_view_if_needed()
            facts.update(
                setting_count = 1,
                custom_minutes = 60,
                saved_minutes = api.get(ROUTE).json()["minutes"],
                reload_persisted = True,
                invalid_inputs_rejected = 5,
            )
        page.get_by_role("dialog").screenshot(path = str(evidence / f"{engine}-{label}.png"))
        assert not errors, errors
        facts["page_errors"] = errors
        return facts
    finally:
        context.close()
        browser.close()


def composite(evidence):
    images = [
        Image.open(evidence / f"chromium-{label}.png").convert("RGB")
        for label in ("before", "after")
    ]
    width = max(image.width for image in images)
    canvas = Image.new("RGB", (width, sum(image.height + 65 for image in images)), "#151719")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 23)
    top = 0
    for image, text in zip(
        images,
        (
            "BEFORE — no first-token waiting setting",
            "AFTER — Unlimited saved and retained on reload",
        ),
    ):
        draw.text((20, top + 18), text, font = font, fill = "white")
        canvas.paste(image, (0, top + 65))
        top += image.height + 65
    canvas.save(evidence / "before-after.png")


def main():
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument("--before", type = Path, required = True)
    parser.add_argument("--after", type = Path, required = True)
    parser.add_argument("--backend-python", type = Path, required = True)
    parser.add_argument("--output", type = Path, required = True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents = True, exist_ok = False)
    evidence = output / "evidence"
    evidence.mkdir()
    meta = {
        "scene": "first-token-timeout-settings",
        "expect": EXPECT,
        "sides": {},
        "facts": [],
        "limitations": "Real settings API and production UI; no model/GPU inference. Backend timeout tests use simulated time.",
    }
    with sync_playwright() as playwright:
        for label, root in (("before", args.before.resolve()), ("after", args.after.resolve())):
            with server(
                root, output / (label + "-home"), args.backend_python, output / (label + ".log")
            ) as state:
                meta["sides"][label] = state
                api, auth = authenticate(state)
                with api:
                    for engine in ("chromium", "firefox", "webkit"):
                        meta["facts"].append(
                            scene(playwright, engine, label, state, api, auth, evidence)
                        )
                        (evidence / "meta.json").write_text(json.dumps(meta, indent = 2))
                        print(f"PASS {label} {engine}", flush = True)
        with server(
            args.after.resolve(), output / "after-home", args.backend_python, output / "restart.log"
        ) as state:
            api, _auth = authenticate(state)
            with api:
                assert api.get(ROUTE).json()["minutes"] == 0
                meta["restart_persisted"] = True
    meta["facts_diff"] = {
        "setting_count": {"before": 0, "after": 1},
        "custom_minutes": {"before": None, "after": 60},
        "unlimited_minutes": {"before": None, "after": 0},
        "reload_and_restart": {"before": None, "after": True},
    }
    (evidence / "meta.json").write_text(json.dumps(meta, indent = 2))
    composite(evidence)
    print(f"PASS persistence across backend restart. Evidence: {evidence}", flush = True)


if __name__ == "__main__":
    main()
