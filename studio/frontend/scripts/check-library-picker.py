# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import json
import os
import re
import time
import uuid
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from playwright.sync_api import expect, sync_playwright


def main():
    base = os.environ["BASE_URL"].rstrip("/")
    out = Path(os.environ["PW_ART_DIR"])
    out.mkdir(parents = True, exist_ok = True)
    token = ""

    def api(
        path,
        body = None,
        method = "POST",
        content_type = "application/json",
    ):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        request = Request(base + path, data = None if method == "GET" else data, method = method)
        request.add_header("Content-Type", content_type)
        if token:
            request.add_header("Authorization", "Bearer " + token)
        with urlopen(request, timeout = 30) as response:
            return json.load(response)

    auth = api("/api/auth/login", {"username": "unsloth", "password": os.environ["STUDIO_PW"]})
    token = auth["access_token"]
    ident = uuid.uuid4().hex
    thread = "library-picker-" + ident
    name = f"picker-note-{ident[:8]}.txt"
    note = b"Saved Library report: revenue grew by 12 percent.\n"
    boundary = "picker-" + ident
    multipart = (
        (
            f'--{boundary}\r\nContent-Disposition: form-data; name="files"; filename="{name}"\r\n'
            "Content-Type: text/plain\r\n\r\n"
        ).encode()
        + note
        + f"\r\n--{boundary}--\r\n".encode()
    )
    api("/api/library/uploads", multipart, content_type = "multipart/form-data; boundary=" + boundary)
    saved = next(
        item for item in api("/api/library", method = "GET")["items"] if item["name"] == name
    )
    now = int(time.time() * 1000)
    api(
        "/api/chat/threads",
        {
            "id": thread,
            "title": "Library picker regression",
            "modelType": "base",
            "modelId": "",
            "createdAt": now,
            "updatedAt": now,
        },
    )
    parent = None
    for role, text in [
        ("user", "Please review a saved file."),
        ("assistant", "Attach your Library report."),
    ]:
        message = ident + "-" + role
        api(
            f"/api/chat/threads/{thread}/messages/{message}",
            {
                "id": message,
                "threadId": thread,
                "parentId": parent,
                "role": role,
                "content": [{"type": "text", "text": text}],
                "createdAt": now,
            },
            method = "PUT",
        )
        parent = message
    seed = {
        "unsloth_auth_token": token,
        "unsloth_refresh_token": auth.get("refresh_token", ""),
        "unsloth_chat_connections_enabled": "false",
        "unsloth_chat_external_providers": "[]",
        "unsloth_chat_external_provider_keys": "{}",
    }
    facts = {"passed": False, "attachment_counts": [], "pointer_events": []}
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        context = browser.new_context(viewport = {"width": 1280, "height": 900})
        context.add_init_script(
            "for (const [k,v] of Object.entries("
            + json.dumps(seed)
            + ")) localStorage.setItem(k,v);"
        )
        page = context.new_page()
        page.set_default_timeout(15_000)
        try:
            page.goto(f"{base}/chat?thread={thread}", wait_until = "domcontentloaded")
            expect(page.get_by_text("Attach your Library report.", exact = True)).to_be_visible(
                timeout = 60_000
            )
            composer = page.locator("form:has(textarea)").filter(visible = True).first
            text = composer.locator("textarea").first
            draft = "Summarize the attached report without leaving this conversation."
            text.fill(draft)
            original_url = page.url
            plus = composer.get_by_role("button", name = "Tools and attachments", exact = True)
            dialog = page.get_by_role("dialog", name = "Attach from Library", exact = True)
            remove = composer.get_by_role("button", name = "Remove file", exact = True)
            for attempt in range(2):
                plus.click()
                expect(
                    page.get_by_role("menuitem", name = "Attach from Library", exact = True)
                ).to_have_count(0)
                page.get_by_role("menuitem", name = "More", exact = True).click()
                page.get_by_role("menuitem", name = "Attach from Library", exact = True).click()
                expect(dialog).to_be_visible()
                dialog.get_by_placeholder("Search Library files...", exact = True).fill(name)
                with page.expect_response(
                    lambda response: urlsplit(response.url).path == urlsplit(saved["fileUrl"]).path
                    and response.request.method == "GET"
                ) as response:
                    dialog.get_by_role("option").filter(has_text = name).click()
                assert response.value.ok and response.value.body() == note, "Library bytes changed"
                expect(dialog).to_be_hidden()
                expect(remove).to_have_count(1)
                expect(text).to_have_value(draft)
                assert page.url == original_url, "Attaching changed the conversation"
                page.wait_for_function("getComputedStyle(document.body).pointerEvents !== 'none'")
                facts["attachment_counts"].append(remove.count())
                facts["pointer_events"].append(
                    page.evaluate("getComputedStyle(document.body).pointerEvents")
                )
                if attempt == 0:
                    remove.click()
                    expect(remove).to_have_count(0)
            page.screenshot(path = str(out / "library-picker-attached.png"), animations = "disabled")
            page.get_by_test_id("nav-row-library").click(timeout = 5_000)
            heading = page.locator("h1").filter(has_text = re.compile(r"^Library$"))
            expect(heading).to_be_visible()
            facts["sidebar_click_succeeded"] = True
            page.locator(f'[data-testid="recent-thread"][data-thread-id="{thread}"]').click()
            page.wait_for_url(
                lambda url: urlsplit(url).path == "/chat" and thread in urlsplit(url).query
            )
            expect(text).to_be_visible()
            marker = uuid.uuid4().hex
            page.evaluate("value => { window.__libraryPickerDocument = value; }", marker)
            retained = composer.element_handle()
            plus.click()
            page.get_by_role("menuitem", name = "More", exact = True).click()
            page.get_by_role("menuitem", name = "Attach from Library", exact = True).click()
            expect(dialog).to_be_visible()
            page.go_back(wait_until = "domcontentloaded")
            page.wait_for_url(lambda url: urlsplit(url).path == "/library")
            expect(heading).to_be_visible()
            expect(dialog).to_be_hidden()
            facts["same_document"] = page.evaluate(
                "value => window.__libraryPickerDocument === value", marker
            )
            facts["composer_retained"] = page.evaluate("node => node.isConnected", retained)
            assert facts["same_document"] and facts["composer_retained"], "Back reloaded the app"
            facts["dialog_hidden_after_back"] = True
            facts["draft_preserved"] = True
            facts["passed"] = True
        finally:
            page.screenshot(path = str(out / "library-picker-final.png"), animations = "disabled")
            (out / "library-picker-facts.json").write_text(
                json.dumps(facts, indent = 2), encoding = "utf-8"
            )
            context.close()
            browser.close()
    print("Library picker browser checks passed.")


if __name__ == "__main__":
    main()
