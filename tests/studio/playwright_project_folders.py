# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Drive real project dialogs with stubbed native IPC and HTTP.

Starts its own vite server unless --base-url names a running frontend. --before-url
optionally captures the same dialog from a second frontend. No model, real native
picker, or host filesystem grant is involved.
"""

import asyncio
import argparse
import json
import os
import re
import sys
from pathlib import Path
from playwright.async_api import async_playwright, expect

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _playwright_robust import start_vite, stop_process, wait_for_smoke_page  # noqa: E402

parser = argparse.ArgumentParser(description = __doc__)
parser.add_argument("--base-url")
parser.add_argument("--before-url")
parser.add_argument("--out", type = Path, default = Path("work/project-folders-review"))
ARGS = parser.parse_args()
OUT = ARGS.out.resolve()
MANAGED_FOLDER = "Unsloth-managed folder"  # working-folder-picker.tsx with no folder set

# The picker button's three labels, read from the component rather than copied: #15080 dropped the
# ellipses ("Choose folder…" -> "Choose folder", "Change…" -> "Change folder") and every lookup here
# then timed out on a dialog that worked. The flow below is what this suite checks, not the wording.
_PICKER = (
    Path(__file__).resolve().parents[2]
    / "studio/frontend/src/features/chat/components/working-folder-picker.tsx"
)
_LABELS = re.search(
    r'\{picking \? "([^"]+)" : value \? "([^"]+)" : "([^"]+)"\}',
    _PICKER.read_text(encoding = "utf-8"),
)
assert (
    _LABELS
), f"{_PICKER.name} no longer spells the picker button as picking ? ... : value ? ... : ..."
CHOOSING, CHANGE, CHOOSE = _LABELS.groups()


async def scene(
    browser,
    label,
    base_url,
    mode,
    *,
    exercise = False,
    reject = False,
    cancel = False,
    reset = False,
    pending = False,
    narrow = False,
    legacy = False,
    lost = False,
    lost_put = False,
):
    context = await browser.new_context(
        viewport = {"width": 390, "height": 750} if narrow else {"width": 1100, "height": 950},
        device_scale_factor = 1,
    )
    await context.add_init_script("""
      localStorage.setItem('unsloth_auth_token', 'review.'+btoa(JSON.stringify({sub:'unsloth', role:'owner'}))+'.review');
      window.__picks = 0;
    """)
    await context.add_init_script(
        f"window.__cancelPick={str(cancel).lower()}; window.__delayPick={str(pending).lower()};"
    )
    if mode == "guest":
        await context.add_init_script("""
          localStorage.setItem('unsloth_auth_token', 'review.'+btoa(JSON.stringify({sub:'member',role:'member'}))+'.review');
        """)
    if mode != "browser":
        await context.add_init_script("""
          window.__TAURI_INTERNALS__ = {
            metadata: {currentWindow:{label:'main'}, currentWebview:{label:'main'}},
            transformCallback: () => 1, unregisterCallback: () => {},
            invoke: async (command) => {
              if(command === 'native_path_leases_usable') return true;
              if(command === 'pick_native_document_folder') {
                window.__picks++;
                if(window.__delayPick) return new Promise(resolve => {window.__resolvePick=resolve;});
                if(window.__cancelPick) return null;
                return {token:'native-grant-'+window.__picks, displayName:'Existing project', expiresAtMs:Date.now()+120000};
              }
              if(command.endsWith('scale_factor')) return 1;
              if(command.endsWith('listen')) return 1;
              return null;
            }
          };
        """)
    requests = []
    folder_requests = []
    stored = {}
    failures = int(reject)
    spent = set()

    async def api(route):
        nonlocal failures
        req = route.request
        path = req.url.split("/api/", 1)[-1]
        if path == "health":
            return await route.fulfill(
                json = {"native_path_leases_supported": mode in ("desktop", "guest")}
            )
        if path == "chat/projects" and req.method == "POST":
            data = req.post_data_json
            requests.append(data)
            if data.get("nativePathLease") in spent:
                return await route.fulfill(status = 400, json = {"detail": "Grant already used"})
            if failures:
                failures -= 1
                return await route.fulfill(status = 400, json = {"detail": "Folder selection expired"})
            # An older backend ignores the lease field.
            granted = data.get("nativePathLease") and not legacy
            if granted:
                spent.add(data["nativePathLease"])
            stored.update(
                {**data, "workingFolder": "/review/existing-project" if granted else None}
            )
            if lost and len(requests) == 1:
                # Committed, but the response never arrives.
                return await route.abort("connectionreset")
            return await route.fulfill(json = stored)
        if path.endswith("/working-folder"):
            folder_requests.append(
                {"method": req.method, "body": req.post_data_json if req.method == "PUT" else None}
            )
            put_lease = (
                (req.post_data_json or {}).get("nativePathLease") if req.method == "PUT" else None
            )
            if put_lease in spent:
                return await route.fulfill(status = 400, json = {"detail": "Grant already used"})
            if put_lease and lost_put:
                spent.add(put_lease)
                stored["workingFolder"] = "/review/alternate-project"
                return await route.abort("connectionreset")
            if await page.evaluate("Boolean(window.__rejectUpdate)"):
                await page.evaluate("window.__rejectUpdate=false")
                return await route.fulfill(status = 400, json = {"detail": "Folder selection expired"})
            if await page.evaluate("Boolean(window.__delayUpdate)"):
                await page.wait_for_function("!window.__delayUpdate")
            stored["workingFolder"] = "/review/alternate-project" if req.method == "PUT" else None
            return await route.fulfill(json = stored)
        if path.startswith("chat/projects/") and req.method == "GET":
            if path.split("/")[2] != stored.get("id"):
                return await route.fulfill(status = 404, json = {"detail": "Project not found"})
            return await route.fulfill(json = stored)
        return await route.fulfill(json = {})

    await context.route(f"{base_url}/api/**", api)
    page = await context.new_page()
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.on("pageerror", lambda error: print("PAGE ERROR", label, str(error), flush = True))
    print("SCENE", label, flush = True)
    await page.goto(f"{base_url}/smoke-project-folders.html")
    dialog = page.get_by_role("dialog")
    await expect(dialog).to_be_visible(timeout = 90000)
    await page.get_by_label("Project name", exact = True).fill("My project")
    observations = {
        "workingFolderControls": await dialog.get_by_role(
            "button", name = CHOOSE, exact = True
        ).count(),
        "sourceFolderLabels": await dialog.get_by_text("Source folders", exact = True).count(),
        "technicalBackendCopy": "managed desktop backend" in await dialog.inner_text(),
    }
    await page.evaluate("document.fonts.ready")
    await page.screenshot(path = str(OUT / f"{label}.png"), animations = "disabled")
    await dialog.screenshot(path = str(OUT / f"{label}-dialog.png"), animations = "disabled")
    if exercise:
        if mode == "desktop":
            choose = dialog.get_by_role("button", name = CHOOSE, exact = True)
            await expect(choose).to_be_enabled()
            await choose.click()
            if pending:
                await expect(
                    dialog.get_by_role("button", name = CHOOSING, exact = True)
                ).to_be_disabled()
                await page.keyboard.press("Escape")
                await expect(dialog).to_be_visible()
                # The button flips before the native pick is invoked.
                await page.wait_for_function("typeof window.__resolvePick === 'function'")
                await page.evaluate("window.__resolvePick(null)")
            if cancel or pending:
                await expect(dialog.get_by_text(MANAGED_FOLDER, exact = True)).to_be_visible()
                await expect(choose).to_be_enabled()
            else:
                await expect(dialog.get_by_text("Existing project", exact = True)).to_be_visible()
            if reset:
                await dialog.get_by_role("button", name = "Use managed folder", exact = True).click()
                await expect(dialog.get_by_text(MANAGED_FOLDER, exact = True)).to_be_visible()
            if narrow:
                assert await page.evaluate("document.documentElement.scrollWidth <= innerWidth")
            await page.screenshot(path = str(OUT / f"{label}-selected.png"), animations = "disabled")
            await dialog.screenshot(
                path = str(OUT / f"{label}-selected-dialog.png"), animations = "disabled"
            )
            await dialog.get_by_role("button", name = "Create project", exact = True).click()
            if reject:
                await expect(dialog.get_by_role("alert")).to_contain_text("Choose the folder again")
                assert len(requests) == 1
                await expect(
                    dialog.get_by_role("button", name = "Create project", exact = True)
                ).to_be_disabled()
                await page.screenshot(path = str(OUT / f"{label}-expired.png"), animations = "disabled")
                await dialog.get_by_role("button", name = CHANGE, exact = True).click()
                await dialog.get_by_role("button", name = "Create project", exact = True).click()
            if not (lost or lost_put):
                await expect(page.get_by_test_id("created")).not_to_be_empty()
            if cancel or pending or reset:
                assert "nativePathLease" not in requests[-1]
            else:
                assert requests[-1]["nativePathLease"].startswith("native-grant-")
            assert requests[-1]["name"] == "My project"
            if lost:
                await expect(dialog.get_by_role("alert")).to_have_count(0)
                await dialog.get_by_role("button", name = "Create project", exact = True).click()
                await expect(page.get_by_test_id("created")).to_have_text(requests[0]["id"])
                assert {r["id"] for r in requests} == {requests[0]["id"]}
                assert not folder_requests
            if lost_put:
                await expect(dialog.get_by_role("alert")).to_have_count(0)
                await dialog.get_by_role("button", name = "Create project", exact = True).click()
                await expect(page.get_by_test_id("created")).to_have_text(requests[0]["id"])
                assert len(requests) == 1
                assert [r["method"] for r in folder_requests] == ["PUT", "PUT"]
            if legacy and not lost_put:
                assert folder_requests == [
                    {"method": "PUT", "body": {"nativePathLease": requests[-1]["nativePathLease"]}}
                ]
            if label == "after-desktop":
                await page.get_by_role("button", name = "Edit project", exact = True).click()
                await expect(
                    dialog.get_by_text("/review/existing-project", exact = True)
                ).to_be_visible()
                await expect(
                    dialog.get_by_text(
                        "Where code and file tools run for every chat in this project.", exact = True
                    )
                ).to_be_visible()
                await page.evaluate("window.__cancelPick=true")
                await dialog.get_by_role("button", name = CHANGE, exact = True).click()
                await expect(dialog.get_by_role("button", name = CHANGE, exact = True)).to_be_enabled()
                assert not folder_requests
                await page.evaluate("window.__cancelPick=false; window.__rejectUpdate=true")
                await dialog.get_by_role("button", name = CHANGE, exact = True).click()
                await expect(
                    page.get_by_text("Could not change the working folder", exact = True)
                ).to_be_visible()
                await expect(
                    dialog.get_by_text("/review/existing-project", exact = True)
                ).to_be_visible()
                await page.evaluate("window.__delayUpdate=true")
                await dialog.get_by_role("button", name = CHANGE, exact = True).click()
                await expect(dialog.get_by_role("button", name = "Save", exact = True)).to_be_disabled()
                await expect(
                    dialog.get_by_role("button", name = "Cancel", exact = True)
                ).to_be_disabled()
                await expect(
                    dialog.get_by_role("button", name = "Delete project", exact = True)
                ).to_be_disabled()
                await page.keyboard.press("Escape")
                await expect(dialog).to_be_visible()
                await page.evaluate("window.__delayUpdate=false")
                await expect(
                    dialog.get_by_text("/review/alternate-project", exact = True)
                ).to_be_visible()
                await dialog.screenshot(
                    path = str(OUT / "after-edit-selected-dialog.png"), animations = "disabled"
                )
                await dialog.get_by_role("button", name = "Cancel", exact = True).click()
                await page.get_by_role("button", name = "Edit project", exact = True).click()
                await expect(
                    dialog.get_by_text("/review/alternate-project", exact = True)
                ).to_be_visible()
                await dialog.get_by_role("button", name = "Use managed folder", exact = True).click()
                await expect(dialog.get_by_text(MANAGED_FOLDER, exact = True)).to_be_visible()
                await dialog.get_by_role("button", name = "Cancel", exact = True).click()
                await page.get_by_role("button", name = "Edit project", exact = True).click()
                await expect(dialog.get_by_text(MANAGED_FOLDER, exact = True)).to_be_visible()
                assert [r["method"] for r in folder_requests] == ["PUT", "PUT", "DELETE"]
        else:
            if mode in ("unsupported", "guest"):
                await expect(dialog.get_by_role("button", name = CHOOSE, exact = True)).to_be_disabled()
            await dialog.get_by_role("button", name = "Create project", exact = True).click()
            await expect(page.get_by_test_id("created")).not_to_be_empty()
            assert "nativePathLease" not in requests[-1]
    assert not errors, errors
    facts = {
        "mode": mode,
        "requests": requests,
        "folderRequests": folder_requests,
        "pageErrors": errors,
        "rejectedGrant": reject,
        "cancelledPicker": cancel,
        "resetToDefault": reset,
        "pendingPicker": pending,
        "narrowViewport": narrow,
        "olderBackend": legacy,
        "lostResponse": lost,
        "lostFolderResponse": lost_put,
        **observations,
    }
    await context.close()
    return facts


async def ambiguous_create(browser, label, base_url, choice):
    """A lease-carrying create commits but its response and the recovery read are both lost."""
    context = await browser.new_context(viewport = {"width": 1100, "height": 950})
    await context.add_init_script("""
      localStorage.setItem('unsloth_auth_token', 'review.'+btoa(JSON.stringify({sub:'unsloth', role:'owner'}))+'.review');
      window.__picks = 0; window.__failGet = false;
      window.__TAURI_INTERNALS__ = {
        metadata: {currentWindow:{label:'main'}, currentWebview:{label:'main'}},
        transformCallback: () => 1, unregisterCallback: () => {},
        invoke: async (command) => {
          if(command === 'native_path_leases_usable') return true;
          if(command === 'pick_native_document_folder') {
            window.__picks++;
            return {token:'native-grant-'+window.__picks, displayName:'Folder '+window.__picks, expiresAtMs:Date.now()+120000};
          }
          if(command.endsWith('scale_factor')) return 1;
          if(command.endsWith('listen')) return 1;
          return null;
        }
      };
    """)
    posts, folder_requests, spent, stored = [], [], set(), {}

    def grant(lease):
        # The backend spends a grant before anything else can fail.
        if lease in spent:
            return False
        spent.add(lease)
        return True

    async def api(route):
        req = route.request
        path = req.url.split("/api/", 1)[-1]
        if path == "health":
            return await route.fulfill(json = {"native_path_leases_supported": True})
        if path == "chat/projects" and req.method == "POST":
            data = req.post_data_json
            posts.append(data)
            lease = data.get("nativePathLease")
            if lease and not grant(lease):
                return await route.fulfill(status = 400, json = {"detail": "Grant already used"})
            if lease and stored:
                return await route.fulfill(
                    status = 409, json = {"detail": "This project already exists."}
                )
            folder = "/review/" + lease if lease else stored.get("workingFolder")
            stored.update({**data, "workingFolder": folder})
            if len(posts) == 1:
                return await route.abort("connectionreset")
            return await route.fulfill(json = stored)
        if path.endswith("/working-folder"):
            body = req.post_data_json if req.method == "PUT" else None
            folder_requests.append({"method": req.method, "body": body})
            if body and not grant(body["nativePathLease"]):
                return await route.fulfill(status = 400, json = {"detail": "Grant already used"})
            stored["workingFolder"] = "/review/" + body["nativePathLease"] if body else None
            return await route.fulfill(json = stored)
        if path.startswith("chat/projects/") and req.method == "GET":
            if await page.evaluate("window.__failGet"):
                return await route.abort("connectionreset")
            if path.split("/")[2] != stored.get("id"):
                return await route.fulfill(status = 404, json = {"detail": "Project not found"})
            return await route.fulfill(json = stored)
        return await route.fulfill(json = {})

    await context.route(f"{base_url}/api/**", api)
    page = await context.new_page()
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    print("SCENE", label, flush = True)
    await page.goto(f"{base_url}/smoke-project-folders.html")
    dialog = page.get_by_role("dialog")
    await expect(dialog).to_be_visible(timeout = 90000)
    await page.get_by_label("Project name", exact = True).fill("My project")
    create = dialog.get_by_role("button", name = "Create project", exact = True)
    await dialog.get_by_role("button", name = CHOOSE, exact = True).click()
    await page.evaluate("window.__failGet = true")
    await create.click()
    await expect(dialog.get_by_role("alert")).to_contain_text(
        "Choose the folder again", timeout = 60000
    )
    await page.evaluate("window.__failGet = false")
    if choice == "managed":
        await dialog.get_by_role("button", name = "Use managed folder", exact = True).click()
        await create.click()
        await expect(page.get_by_test_id("created")).to_have_text(posts[0]["id"])
        assert [r["method"] for r in folder_requests] == ["DELETE"]
        assert stored["workingFolder"] is None
    else:
        await dialog.get_by_role("button", name = CHANGE, exact = True).click()
        await create.click()
        # The second grant was spent on the conflict, and the saved folder is the first one.
        await expect(dialog.get_by_role("alert")).to_contain_text("Choose the folder again")
        await dialog.get_by_role("button", name = CHANGE, exact = True).click()
        await create.click()
        await expect(page.get_by_test_id("created")).to_have_text(posts[0]["id"])
        assert folder_requests == [{"method": "PUT", "body": {"nativePathLease": "native-grant-3"}}]
        assert stored["workingFolder"] == "/review/native-grant-3"
    assert {p["id"] for p in posts} == {posts[0]["id"]}
    assert not errors, errors
    await context.close()
    return {"posts": posts, "folderRequests": folder_requests, "saved": stored.get("workingFolder")}


async def main():
    OUT.mkdir(parents = True, exist_ok = True)
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless = True)
        facts = {}
        if ARGS.before_url:
            facts["before-browser"] = await scene(
                browser, "before-browser", ARGS.before_url, "browser"
            )
        facts["after-browser"] = await scene(
            browser, "after-browser", ARGS.base_url, "browser", exercise = True
        )
        facts["after-desktop-unavailable"] = await scene(
            browser, "after-desktop-unavailable", ARGS.base_url, "unsupported", exercise = True
        )
        facts["after-desktop"] = await scene(
            browser, "after-desktop", ARGS.base_url, "desktop", exercise = True
        )
        facts["after-retry"] = await scene(
            browser, "after-retry", ARGS.base_url, "desktop", exercise = True, reject = True
        )
        facts["after-picker-cancel"] = await scene(
            browser, "after-picker-cancel", ARGS.base_url, "desktop", exercise = True, cancel = True
        )
        facts["after-use-default"] = await scene(
            browser, "after-use-default", ARGS.base_url, "desktop", exercise = True, reset = True
        )
        facts["after-picker-pending"] = await scene(
            browser, "after-picker-pending", ARGS.base_url, "desktop", exercise = True, pending = True
        )
        facts["after-managed-account"] = await scene(
            browser, "after-managed-account", ARGS.base_url, "guest", exercise = True
        )
        facts["after-older-backend"] = await scene(
            browser, "after-older-backend", ARGS.base_url, "desktop", exercise = True, legacy = True
        )
        facts["after-lost-response"] = await scene(
            browser, "after-lost-response", ARGS.base_url, "desktop", exercise = True, lost = True
        )
        facts["after-older-backend-lost-put"] = await scene(
            browser,
            "after-older-backend-lost-put",
            ARGS.base_url,
            "desktop",
            exercise = True,
            legacy = True,
            lost_put = True,
        )
        facts["after-ambiguous-then-other"] = await ambiguous_create(
            browser, "after-ambiguous-then-other", ARGS.base_url, "other"
        )
        facts["after-ambiguous-then-managed"] = await ambiguous_create(
            browser, "after-ambiguous-then-managed", ARGS.base_url, "managed"
        )
        facts["after-narrow"] = await scene(
            browser, "after-narrow", ARGS.base_url, "desktop", exercise = True, narrow = True
        )
        await browser.close()
    (OUT / "facts.json").write_text(json.dumps(facts, indent = 2), encoding = "utf-8")
    print(json.dumps({"scenes": len(facts), "output": str(OUT)}, indent = 2))


if ARGS.base_url:
    asyncio.run(main())
else:
    port = int(os.environ.get("SMOKE_PORT", "5433"))
    ARGS.base_url = f"http://127.0.0.1:{port}"
    server = start_vite(port)
    try:
        wait_for_smoke_page(
            f"{ARGS.base_url}/smoke-project-folders.html",
            "smoke-project-folders-main.tsx",
            proc = server,
        )
        asyncio.run(main())
    finally:
        stop_process(server)
