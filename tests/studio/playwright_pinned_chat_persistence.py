# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Chat pins survive browser lifecycle changes and synchronize between clients.

Loads the production store and lifecycle hook through Vite, with browser requests
served by the real chat-settings routes against temporary SQLite. No model or
running Unsloth installation is required, and no existing user data is touched.
The delayed-read cases put an optimistic edit behind an in-flight request; merely
adding keepalive to the eventual write cannot make those cases pass.

Prerequisites: frontend npm dependencies, Chromium installed by Playwright, and
Python packages playwright, fastapi, httpx, pyjwt, argon2-cffi, cryptography, huggingface_hub,
python-multipart, and structlog. Run from any directory:

    python tests/studio/playwright_pinned_chat_persistence.py

PLAYWRIGHT_CHROMIUM_EXECUTABLE optionally selects an already installed browser.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from urllib.parse import urlsplit

REPO = Path(__file__).resolve().parents[2]
FRONTEND = REPO / "studio" / "frontend"
BACKEND = REPO / "studio" / "backend"


def write_browser_fixture(directory: Path) -> None:
    source = "/@fs/" + (FRONTEND / "src").as_posix().lstrip("/")
    (directory / "index.html").write_text(
        """<!doctype html><html><body><div id="sync"></div><script type="module">
import React, {useEffect} from "react";
import {createRoot} from "react-dom/client";
import * as store from "__SOURCE__/features/chat/stores/pinned-chats-store.ts";
import * as runtime from "__SOURCE__/features/chat/sync-pinned-chats.ts";
import {usePinnedChatsSync} from "__SOURCE__/features/chat/hooks/use-pinned-chats-sync.ts";
import * as auth from "__SOURCE__/features/auth/session.ts";
window.pins = {...store, ...runtime};
window.auth = auth;
let root;
function Sync() {
  usePinnedChatsSync();
  useEffect(() => {
    window.syncMounted = true;
    return () => {window.syncMounted = false;};
  }, []);
  return null;
}
window.mountSync = () => {
  root = createRoot(document.getElementById("sync"));
  root.render(React.createElement(Sync));
};
window.unmountSync = () => root.unmount();
window.ready = true;
</script></body></html>""".replace("__SOURCE__", source),
        encoding = "utf-8",
    )
    (directory / "vite.mjs").write_text(
        "import {createServer} from "
        + json.dumps((FRONTEND / "node_modules/vite/dist/node/index.js").as_uri())
        + ";\nimport {writeFileSync} from 'node:fs';\n"
        + "const server = await createServer("
        + json.dumps(
            {
                "configFile": str(FRONTEND / "vite.config.ts"),
                "root": str(directory),
                "server": {"host": "127.0.0.1", "port": 0, "fs": {"allow": [str(REPO)]}},
            }
        )
        + ");\nawait server.listen();\n"
        + "writeFileSync("
        + json.dumps(str(directory / "ready.json"))
        + ", JSON.stringify({url: `http://127.0.0.1:${server.httpServer.address().port}`}));\n",
        encoding = "utf-8",
    )


def settings_client(home: Path):
    # Import the actual route file without importing routes/__init__.py, whose
    # application-wide registry eagerly loads unrelated training dependencies.
    os.environ["UNSLOTH_STUDIO_HOME"] = str(home)
    os.environ["UNSLOTH_STUDIO_DISABLE_DEVICE_PROBE"] = "1"
    sys.path.insert(0, str(BACKEND))
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from auth.authentication import get_current_subject

    spec = importlib.util.spec_from_file_location(
        "pin_persistence_routes", BACKEND / "routes/chat_history.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    app = FastAPI()
    app.include_router(module.router, prefix = "/api/chat")
    app.dependency_overrides[get_current_subject] = lambda: "pin-persistence-test"
    return TestClient(app)


async def exercise_browser(url: str, client) -> list[str]:
    from playwright.async_api import async_playwright

    passed: list[str] = []
    errors: list[str] = []

    def server_pins():
        return client.get("/api/chat/settings").json()["settings"].get("pinnedChatIds")

    def save_server_pins(ids):
        response = client.put("/api/chat/settings", json = {"pinnedChatIds": ids})
        assert response.status_code == 200, response.text

    async with async_playwright() as playwright:
        executable = os.environ.get("PLAYWRIGHT_CHROMIUM_EXECUTABLE")
        browser = await playwright.chromium.launch(
            **({"executable_path": executable} if executable else {})
        )
        try:

            async def new_client(
                *,
                context = None,
                legacy = None,
                old_backend = False,
                missing_route_status = 404,
            ):
                context = context or await browser.new_context()
                page = await context.new_page()
                state = {
                    "hold_next_get": False,
                    "held": asyncio.Event(),
                    "release": asyncio.Event(),
                    "old_backend": old_backend,
                    "old_pins": None,
                }

                async def serve(route):
                    request = route.request
                    if request.method == "GET" and state["hold_next_get"]:
                        state["hold_next_get"] = False
                        state["held"].set()
                        await state["release"].wait()
                    if state["old_backend"]:
                        if urlsplit(request.url).path.startswith("/api/chat/pins"):
                            # Reverse proxies and older servers may return plain text.
                            await route.fulfill(
                                status = missing_route_status,
                                body = "Missing endpoint",
                                content_type = "text/plain",
                            )
                            return
                        response = (
                            {
                                "settings": {}
                                if state["old_pins"] is None
                                else {"pinnedChatIds": state["old_pins"]}
                            }
                            if request.method == "GET"
                            else {
                                "detail": [
                                    {
                                        "type": "value_error",
                                        "loc": ["expectedAbsent"],
                                        "msg": "Value error, Unknown chat settings field(s): pinnedChatIds",
                                    },
                                    {
                                        "type": "extra_forbidden",
                                        "loc": ["patch", "pinnedChatIds"],
                                        "msg": "Extra inputs are not permitted",
                                    },
                                ]
                            }
                        )
                        await route.fulfill(
                            status = 200 if request.method == "GET" else 400, json = response
                        )
                        return
                    response = client.request(
                        request.method,
                        urlsplit(request.url).path,
                        content = request.post_data,
                        headers = {"Content-Type": "application/json"},
                    )
                    # Reload/signout can abort the intentionally delayed read.
                    try:
                        await route.fulfill(
                            status = response.status_code,
                            body = response.content,
                            headers = {"Content-Type": "application/json"},
                        )
                    except Exception:
                        if not request.failure:
                            raise

                await page.route("**/api/chat/settings**", serve)
                await page.route("**/api/chat/pins**", serve)
                page.on("pageerror", lambda error: errors.append(str(error)))
                # Supported older WebKit versions lack these static helpers.
                # Keep every lifecycle scenario on that API feature floor.
                await page.add_init_script("""
                  Object.defineProperty(AbortSignal, 'any', {value: undefined, configurable: true});
                  Object.defineProperty(AbortSignal, 'timeout', {value: undefined, configurable: true});
                """)
                await page.add_init_script("""if (!localStorage.getItem('unsloth_auth_token')) {
                  localStorage.setItem('unsloth_auth_token', 'pin-test-access');
                  localStorage.setItem('unsloth_auth_refresh_token', 'pin-test-refresh');
                  localStorage.setItem('unsloth_auth_session_mark', 'pin-test-session');
                }""")
                if legacy is not None:
                    await page.add_init_script(
                        "localStorage.setItem('unsloth_pinned_chats', "
                        + json.dumps(json.dumps({"state": {"pinnedIds": legacy}, "version": 0}))
                        + ")"
                    )
                await page.goto(url)
                await page.wait_for_function("window.ready === true")
                assert await page.evaluate(
                    "AbortSignal.any === undefined && AbortSignal.timeout === undefined"
                )
                return context, page, state

            async def mount(page):
                await page.evaluate("window.mountSync()")
                await page.wait_for_function("window.syncMounted === true")
                await page.evaluate("window.pins.refreshPinnedChats()")

            async def pins(page):
                return await page.evaluate("window.pins.usePinnedChatsStore.getState().pinnedIds")

            async def delay_focus_read(page, state):
                state.update(hold_next_get = True, held = asyncio.Event(), release = asyncio.Event())
                await page.evaluate("window.dispatchEvent(new Event('focus'))")
                await asyncio.wait_for(state["held"].wait(), timeout = 5)

            for chat_id in (
                "survives-reload",
                "survives-remount",
                "server-before-signout",
                "old-session-only",
                "from-second-browser",
                "local-before-upgrade",
                "after-server-upgrade",
                "downgrade-a",
                "downgrade-b",
                "downgrade-offline",
                "downgrade-concurrent",
                "legacy-token-pending",
            ):
                response = client.post(
                    "/api/chat/threads",
                    json = {
                        "id": chat_id,
                        "title": chat_id,
                        "modelType": "base",
                        "createdAt": 1,
                    },
                )
                assert response.status_code == 200, response.text
            save_server_pins([])
            context, page, state = await new_client()
            await mount(page)
            for pinned in (True, False):
                await delay_focus_read(page, state)
                await page.evaluate(
                    "pinned => window.pins.usePinnedChatsStore.getState().setPinned(['survives-reload'], pinned)",
                    pinned,
                )
                expected = ["survives-reload"] if pinned else []
                assert await pins(page) == expected
                await page.reload()
                await page.wait_for_function("window.ready === true")
                state["release"].set()
                await mount(page)
                assert await pins(page) == expected
                assert server_pins() == expected
            passed.append("queued pin and unpin survive reload")

            # A component cleanup is not itself a session change (StrictMode
            # and remounts also run it), so an unconfirmed edit must survive.
            await delay_focus_read(page, state)
            await page.evaluate(
                "window.pins.usePinnedChatsStore.getState().togglePin('survives-remount'); window.unmountSync()"
            )
            state["release"].set()
            await mount(page)
            assert await pins(page) == ["survives-remount"]
            assert server_pins() == ["survives-remount"]
            passed.append("same-session hook remount retains pending intent")

            save_server_pins(["server-before-signout"])
            await page.evaluate("window.pins.refreshPinnedChats()")
            await delay_focus_read(page, state)
            await page.evaluate(
                "window.pins.usePinnedChatsStore.getState().togglePin('old-session-only'); window.auth.clearAuthTokens()"
            )
            assert await pins(page) == []
            state["release"].set()
            await page.evaluate(
                "window.auth.storeAuthTokens('new-session-access', 'new-session-refresh')"
            )
            await page.evaluate("window.pins.refreshPinnedChats()")
            assert await pins(page) == ["server-before-signout"]
            assert server_pins() == ["server-before-signout"]
            passed.append("signout never replays old pending intent into a new session")

            _, other, _ = await new_client()
            await mount(other)
            await other.evaluate(
                "window.pins.usePinnedChatsStore.getState().togglePin('from-second-browser')"
            )
            await other.evaluate("window.pins.refreshPinnedChats()")
            await page.evaluate("window.dispatchEvent(new Event('focus'))")
            await page.wait_for_function(
                "window.pins.usePinnedChatsStore.getState().pinnedIds[0] === 'from-second-browser'"
            )
            passed.append("focus refresh synchronizes isolated browser clients")

            # Same-origin session changes use storage events instead of the
            # in-document auth events emitted by clearAuthTokens.
            sibling = await context.new_page()
            await sibling.goto(url)
            await sibling.wait_for_function("window.ready === true")
            await sibling.evaluate(
                "localStorage.removeItem('unsloth_auth_token'); localStorage.removeItem('unsloth_auth_session_mark')"
            )
            await page.wait_for_function(
                "window.pins.usePinnedChatsStore.getState().pinnedIds.length === 0"
            )
            await sibling.evaluate(
                "localStorage.setItem('unsloth_auth_token','cross-tab-access'); localStorage.setItem('unsloth_auth_session_mark','cross-tab-session')"
            )
            await page.wait_for_function(
                "window.pins.usePinnedChatsStore.getState().pinnedIds[0] === 'from-second-browser'"
            )
            passed.append("cross-tab session changes reset and rehydrate pins")

            # A supported old backend initially preserves local pins. Once it
            # is upgraded, the open page must resume shared persistence.
            _, compatible, compatibility = await new_client(
                legacy = ["local-before-upgrade"], old_backend = True, missing_route_status = 405
            )
            await mount(compatible)
            assert await pins(compatible) == ["local-before-upgrade"]
            compatibility["old_backend"] = False
            await compatible.evaluate("window.pins.refreshPinnedChats()")
            await compatible.evaluate(
                "window.pins.usePinnedChatsStore.getState().togglePin('after-server-upgrade')"
            )
            await compatible.evaluate("window.pins.refreshPinnedChats()")
            assert "after-server-upgrade" in server_pins()
            passed.append("old-server fallback resumes synchronization after upgrade")

            # Downgrading after reading server pins must preserve that baseline.
            # Keep explicit local changes across reload, then replay against the
            # upgraded server's newer list without replacing another client edit.
            save_server_pins(["downgrade-a", "downgrade-b"])
            _, downgraded, downgrade = await new_client()
            await mount(downgraded)
            downgrade.update(old_backend = True, old_pins = ["downgrade-a", "downgrade-b"])
            await downgraded.evaluate("window.pins.refreshPinnedChats()")
            await downgraded.evaluate(
                "window.pins.usePinnedChatsStore.getState().setPinned(['downgrade-offline'], true)"
            )
            await downgraded.evaluate("window.pins.refreshPinnedChats()")
            assert await pins(downgraded) == ["downgrade-offline", "downgrade-a", "downgrade-b"]
            await downgraded.evaluate(
                "window.pins.usePinnedChatsStore.getState().unpin('downgrade-a')"
            )
            await downgraded.evaluate("window.pins.refreshPinnedChats()")
            await downgraded.reload()
            await downgraded.wait_for_function("window.ready === true")
            await mount(downgraded)
            assert await pins(downgraded) == ["downgrade-offline", "downgrade-b"]
            save_server_pins(["downgrade-concurrent", "downgrade-a", "downgrade-b"])
            downgrade["old_backend"] = False
            await downgraded.evaluate("window.pins.refreshPinnedChats()")
            expected = ["downgrade-offline", "downgrade-concurrent", "downgrade-b"]
            assert await pins(downgraded) == expected
            assert server_pins() == expected
            passed.append(
                "downgrade preserves server pins and reload replays ordered offline changes after upgrade"
            )

            # Browser pin migration can beat the independent legacy thread
            # import. Retry the original IDs after import, while tombstones keep
            # an explicit unpin from being undone by later refreshes or reloads.
            save_server_pins([])
            _, delayed_import, _ = await new_client(legacy = ["legacy-imported-later"])
            await mount(delayed_import)
            assert await pins(delayed_import) == []
            response = client.post(
                "/api/chat/threads",
                json = {
                    "id": "legacy-imported-later",
                    "title": "Imported later",
                    "modelType": "base",
                    "createdAt": 1,
                },
            )
            assert response.status_code == 200, response.text
            await delayed_import.evaluate("window.dispatchEvent(new Event('focus'))")
            await delayed_import.wait_for_function(
                "window.pins.usePinnedChatsStore.getState().pinnedIds.includes('legacy-imported-later')"
            )
            assert server_pins() == ["legacy-imported-later"]
            await delayed_import.evaluate(
                "window.pins.usePinnedChatsStore.getState().unpin('legacy-imported-later')"
            )
            await delayed_import.evaluate("window.pins.refreshPinnedChats()")
            await delayed_import.evaluate("window.dispatchEvent(new Event('focus'))")
            await delayed_import.evaluate("window.pins.refreshPinnedChats()")
            assert await pins(delayed_import) == []
            await delayed_import.reload()
            await delayed_import.wait_for_function("window.ready === true")
            await mount(delayed_import)
            assert await pins(delayed_import) == []
            assert server_pins() == []
            passed.append(
                "legacy pins migrate after delayed thread import and explicit unpins do not resurrect"
            )

            # Pre-session-mark auth storage must gain a stable scope before a
            # durable intent is saved. Hourly token rotation preserves that
            # scope, including while navigation interrupts a pending request.
            _, old_auth, old_auth_state = await new_client()
            await old_auth.evaluate("localStorage.removeItem('unsloth_auth_session_mark')")
            await mount(old_auth)
            mark = await old_auth.evaluate("localStorage.getItem('unsloth_auth_session_mark')")
            assert mark
            await delay_focus_read(old_auth, old_auth_state)
            await old_auth.evaluate(
                "window.pins.usePinnedChatsStore.getState().setPinned(['legacy-token-pending'], true)"
            )
            await old_auth.evaluate(
                "window.auth.storeAuthTokens('rotated-access', 'rotated-refresh')"
            )
            assert (
                await old_auth.evaluate("localStorage.getItem('unsloth_auth_session_mark')") == mark
            )
            assert await pins(old_auth) == ["legacy-token-pending"]
            assert server_pins() == []
            await old_auth.reload()
            await old_auth.wait_for_function("window.ready === true")
            old_auth_state["release"].set()
            await mount(old_auth)
            assert (
                await old_auth.evaluate("localStorage.getItem('unsloth_auth_session_mark')") == mark
            )
            assert await pins(old_auth) == ["legacy-token-pending"]
            assert server_pins() == ["legacy-token-pending"]
            passed.append(
                "legacy auth gains a stable mark and token rotation preserves queued pins across reload"
            )
            assert not errors, errors
        finally:
            await browser.close()
    return passed


def main() -> None:
    if not (FRONTEND / "node_modules").is_dir():
        raise RuntimeError(
            "Install frontend dependencies before running the pin persistence browser test"
        )
    with tempfile.TemporaryDirectory(prefix = ".pin-persistence-test-", dir = FRONTEND) as directory:
        directory = Path(directory)
        write_browser_fixture(directory)
        client = settings_client(directory / "home")
        with (directory / "vite.log").open("w", encoding = "utf-8") as log:
            server = subprocess.Popen(
                ["node", str(directory / "vite.mjs")],
                cwd = FRONTEND,
                stdout = log,
                stderr = subprocess.STDOUT,
            )
            try:
                deadline = time.monotonic() + 60
                while not (directory / "ready.json").exists():
                    if server.poll() is not None or time.monotonic() >= deadline:
                        raise RuntimeError((directory / "vite.log").read_text(encoding = "utf-8"))
                    time.sleep(0.05)
                url = json.loads((directory / "ready.json").read_text(encoding = "utf-8"))["url"]
                passed = asyncio.run(exercise_browser(url, client))
                print(json.dumps({"passed": passed, "count": len(passed)}, indent = 2))
            finally:
                server.terminate()
                try:
                    server.wait(timeout = 5)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait()
                client.close()


if __name__ == "__main__":
    main()
