# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import json
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from playwright.sync_api import expect, sync_playwright

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _playwright_robust import (  # noqa: E402
    chromium_launch_args,
    install_view_transition_killer,
    install_wall_clock_watchdog,
    wait_for_health,
)


BASE = os.environ["BASE_URL"].rstrip("/")
OLD = os.environ["STUDIO_OLD_PW"]
NEW = os.environ["STUDIO_NEW_PW"]
ART = Path(os.environ.get("PW_ART_DIR", "logs/playwright-mcp-arguments"))
BROWSER = os.environ.get("STUDIO_PLAYWRIGHT_BROWSER", "chromium").lower()
CHANNEL = os.environ.get("STUDIO_PLAYWRIGHT_CHANNEL") or None
WALL_TIMEOUT_S = float(os.environ.get("STUDIO_UI_WALL_TIMEOUT_S", "420"))
SOURCE_FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "studio"
    / "backend"
    / "tests"
    / "fixtures"
    / "mcp_argument_echo_server.py"
)
MCP_PYTHON = os.environ.get("STUDIO_MCP_PYTHON", sys.executable)


def info(message: str) -> None:
    print(f"[mcp-ui] {message}", flush = True)


def api(
    path: str,
    payload: dict | None = None,
    token: str | None = None,
    method: str = "POST",
) -> dict | list:
    request = urllib.request.Request(
        BASE + path,
        data = json.dumps(payload).encode() if payload is not None else None,
        headers = {
            "Content-Type": "application/json",
            **({"Authorization": f"Bearer {token}"} if token else {}),
        },
        method = method,
    )
    with urllib.request.urlopen(request, timeout = 30) as response:
        body = response.read()
        return json.loads(body) if body else {}


def authenticate() -> dict:
    try:
        session = api("/api/auth/login", {"username": "unsloth", "password": NEW})
        if not session.get("must_change_password"):
            return session
    except urllib.error.HTTPError:
        pass
    initial = api("/api/auth/login", {"username": "unsloth", "password": OLD})
    try:
        api(
            "/api/auth/change-password",
            {"current_password": OLD, "new_password": NEW},
            initial["access_token"],
        )
    except urllib.error.HTTPError as exc:
        if exc.code not in (400, 401, 403):
            raise
    return api("/api/auth/login", {"username": "unsloth", "password": NEW})


def remove_prior_test_servers(token: str) -> None:
    servers = api("/api/mcp/servers/", token = token, method = "GET")
    assert isinstance(servers, list)
    for server in servers:
        if server.get("display_name") not in (
            "Playwright argument echo",
            "Playwright imported echo",
            "Unsloth Docs",
        ):
            continue
        api(
            f"/api/mcp/servers/{server['id']}",
            token = token,
            method = "DELETE",
        )


def open_dialog(page):
    page.get_by_role("button", name = "MCP servers").click()
    page.get_by_role("menuitem", name = "Manage MCP servers").press("Enter")
    dialog = page.get_by_role("dialog", name = "MCP Servers")
    expect(dialog).to_be_visible()
    return dialog


def row_for(dialog, name: str):
    row = dialog.locator("li").filter(has_text = name)
    expect(row).to_have_count(1)
    return row


def fill_arguments(dialog, arguments: list[str]) -> None:
    for index, argument in enumerate(arguments, start = 1):
        dialog.get_by_role("button", name = "Add argument").click()
        dialog.get_by_role("textbox", name = f"Argument {index}", exact = True).fill(argument)


def fill_environment(dialog, values: dict[str, str]) -> None:
    for key, value in values.items():
        dialog.get_by_role("button", name = "Add variable").click()
        keys = dialog.get_by_placeholder("Variable name")
        vals = dialog.get_by_placeholder("Variable value")
        keys.nth(keys.count() - 1).fill(key)
        vals.nth(vals.count() - 1).fill(value)


def assert_arguments(dialog, expected: list[str]) -> None:
    inputs = dialog.locator('[aria-label^="Argument "]')
    expect(inputs).to_have_count(len(expected))
    for index, value in enumerate(expected):
        expect(inputs.nth(index)).to_have_value(value)


def screenshot(dialog, name: str) -> None:
    dialog.screenshot(path = str(ART / f"{name}-{BROWSER}.png"))


def delay_real_response(seconds: float):
    def handler(route):
        response = route.fetch()
        time.sleep(seconds)
        route.fulfill(response = response)

    return handler


def hold_next_request(page, pattern: str) -> list:
    held: list = []
    page.route(pattern, lambda route: held.append(route), times = 1)
    return held


def release_request(page, held: list) -> None:
    page.wait_for_timeout(50)
    assert len(held) == 1
    held[0].continue_()


def run(page, launch_log: Path, fixture: Path) -> None:
    page.goto(BASE + "/hub", wait_until = "domcontentloaded")
    page.goto(BASE + "/chat", wait_until = "domcontentloaded")
    page.wait_for_timeout(1000)
    if page.url.startswith(BASE + "/login") or page.url.startswith(BASE + "/change-password"):
        raise AssertionError(f"not authenticated: {page.url}")

    dialog = open_dialog(page)
    dialog.get_by_role("button", name = "Add server").click()
    dialog.locator("#mcp-display-name").fill("Playwright argument echo")
    dialog.locator("#mcp-url").fill(MCP_PYTHON)
    arguments = [
        str(fixture),
        "--flag",
        "",
        "a b",
        'quote"inside',
        "single'quote",
        "trailing\\",
        "https://example.com/value?q=a%20b",
        "  keep outer spaces  ",
    ]
    fill_arguments(dialog, arguments)
    environment = {
        "UNSLOTH_MCP_ARGUMENT_MARKER": "playwright create",
        "UNSLOTH_MCP_ARGUMENT_LOG": str(launch_log),
    }
    fill_environment(dialog, environment)
    screenshot(dialog, "fixed-create-filled")

    dialog.get_by_role("button", name = "Test connection").click()
    expect(page.get_by_text("Connected (1 tool)")).to_be_visible(timeout = 30_000)
    dialog.get_by_role("button", name = "Add server").click()
    row = row_for(dialog, "Playwright argument echo")
    expect(row).to_contain_text("--flag")

    page.route("**/api/mcp/servers/stdio/decode", delay_real_response(1.5))
    row.get_by_role("button", name = "Edit server").click(no_wait_after = True)
    loading = dialog.get_by_text("Reading local command…", exact = True)
    expect(loading).to_be_visible()
    screenshot(dialog, "fixed-edit-loading")
    page.unroute("**/api/mcp/servers/stdio/decode")
    expect(loading).to_be_hidden(timeout = 10_000)
    assert_arguments(dialog, arguments)
    screenshot(dialog, "fixed-edit-hydrated")

    dialog.get_by_role("button", name = "Cancel").click()
    dialog.get_by_role("button", name = "Close").click()
    page.reload(wait_until = "domcontentloaded")
    page.wait_for_timeout(500)
    dialog = open_dialog(page)
    row = row_for(dialog, "Playwright argument echo")
    row.get_by_role("button", name = "Edit server").click()
    assert_arguments(dialog, arguments)

    edited = [str(fixture), "second", "", "first", "a&b", "x|y", "%TOKEN%"]
    inputs = dialog.locator('[aria-label^="Argument "]')
    while inputs.count() > len(edited):
        dialog.get_by_role("button", name = f"Remove argument {inputs.count()}").click()
    for index, value in enumerate(edited):
        dialog.get_by_role("textbox", name = f"Argument {index + 1}", exact = True).fill(value)
    dialog.get_by_placeholder("Variable value").first.fill("playwright edited")
    dialog.get_by_role("button", name = "Save changes").click()
    expect(page.get_by_text("MCP server updated")).to_be_visible()
    row = row_for(dialog, "Playwright argument echo")
    row.get_by_role("button", name = "Edit server").click()
    assert_arguments(dialog, edited)
    screenshot(dialog, "fixed-edit-persisted")

    dialog.get_by_role("button", name = "Cancel").click()
    row = row_for(dialog, "Playwright argument echo")
    launches_before_refresh = len(launch_log.read_text(encoding = "utf-8").splitlines())
    held_refresh = hold_next_request(page, "**/api/mcp/servers/*/refresh")
    row.get_by_role("button", name = "Refresh tools").click(no_wait_after = True)
    expect(row.get_by_role("switch", name = "Enable server")).to_be_disabled()
    expect(row.get_by_role("button", name = "Refresh tools")).to_be_disabled()
    expect(row.get_by_role("button", name = "Edit server")).to_be_disabled()
    expect(row.get_by_role("button", name = "Delete server")).to_be_disabled()
    expect(dialog.get_by_role("button", name = "Close")).to_have_count(0)
    screenshot(dialog, "fixed-row-busy")
    release_request(page, held_refresh)
    expect(page.get_by_text('Refreshed "Playwright argument echo" (1 tool)')).to_be_visible(
        timeout = 30_000
    )
    launches_after_refresh = len(launch_log.read_text(encoding = "utf-8").splitlines())
    assert launches_after_refresh == launches_before_refresh + 1

    imported = {
        "mcpServers": {
            "Playwright imported echo": {
                "command": MCP_PYTHON,
                "args": [str(fixture), "imported", "", "with spaces", 'say "hello"'],
                "env": {
                    "UNSLOTH_MCP_ARGUMENT_MARKER": "playwright import",
                    "UNSLOTH_MCP_ARGUMENT_LOG": str(launch_log),
                },
            }
        }
    }
    held_import = hold_next_request(page, "**/api/mcp/servers/import")
    dialog.locator('input[type="file"]').set_input_files(
        {
            "name": "mcp-config.json",
            "mimeType": "application/json",
            "buffer": json.dumps(imported).encode(),
        }
    )
    expect(dialog.get_by_role("button", name = "Import")).to_be_disabled()
    expect(dialog.get_by_role("button", name = "Add server")).to_be_disabled()
    dialog.get_by_role("button", name = "Close").click()
    composer = page.get_by_role("button", name = "MCP servers")
    composer.click()
    expect(page.get_by_role("menuitem", name = "Unsloth Docs")).to_be_disabled()
    page.get_by_role("menu").screenshot(
        path = str(ART / f"fixed-composer-import-reconciliation-busy-{BROWSER}.png")
    )
    page.keyboard.press("Escape")
    dialog = open_dialog(page)
    expect(dialog.get_by_role("button", name = "Import")).to_be_disabled()
    expect(dialog.get_by_role("button", name = "Add server")).to_be_disabled()
    screenshot(dialog, "fixed-import-reopened-busy")
    release_request(page, held_import)
    expect(dialog.get_by_role("button", name = "Import")).to_be_enabled(timeout = 10_000)
    imported_row = row_for(dialog, "Playwright imported echo")
    imported_row.get_by_role("button", name = "Edit server").click()
    assert_arguments(dialog, [str(fixture), "imported", "", "with spaces", 'say "hello"'])
    dialog.get_by_role("button", name = "Test connection").click()
    expect(page.get_by_text("Connected (1 tool)")).to_be_visible(timeout = 30_000)
    dialog.get_by_role("button", name = "Cancel").click()
    imported_row = row_for(dialog, "Playwright imported echo")
    imported_row.get_by_role("button", name = "Refresh tools").click()
    expect(page.get_by_text('Refreshed "Playwright imported echo" (1 tool)')).to_be_visible(
        timeout = 30_000
    )

    row = row_for(dialog, "Playwright argument echo")
    page.route(
        "**/api/mcp/servers/stdio/decode",
        lambda route: route.fulfill(
            status = 500,
            content_type = "application/json",
            body = json.dumps({"detail": "forced decode failure"}),
        ),
        times = 1,
    )
    row.get_by_role("button", name = "Edit server").click()
    expect(dialog.get_by_role("alert")).to_contain_text("forced decode failure")
    expect(dialog.locator("#mcp-url")).not_to_have_value("")
    screenshot(dialog, "fixed-edit-error")
    dialog.get_by_role("button", name = "Retry").click()
    assert_arguments(dialog, edited)

    expect(dialog.get_by_placeholder("Variable name")).to_have_count(2)
    dialog.locator("#mcp-url").fill("https://example.com/mcp")
    expect(dialog.get_by_text("Custom headers", exact = True)).to_be_visible()
    expect(dialog.get_by_placeholder("Header name")).to_have_count(0)
    dialog.get_by_role("button", name = "Add header").click()
    dialog.get_by_placeholder("Header name").fill("Authorization")
    dialog.get_by_placeholder("Header value").fill("Bearer remote-secret")
    oauth = dialog.get_by_role("switch", name = "Use OAuth sign-in")
    oauth.click()
    expect(oauth).to_be_checked()
    address = dialog.locator("#mcp-url")
    address.fill("")
    address.press_sequentially("https://example.org/mcp")
    expect(dialog.get_by_placeholder("Header name")).to_have_count(1)
    expect(dialog.get_by_placeholder("Header name")).to_have_value("Authorization")
    expect(oauth).to_be_checked()
    screenshot(dialog, "fixed-http-typing-preserves-credentials")
    address.fill("https:/")
    expect(dialog.get_by_role("button", name = "Save changes")).to_be_disabled()
    expect(dialog.get_by_role("button", name = "Test connection")).to_be_disabled()
    screenshot(dialog, "fixed-partial-http-save-disabled")
    address.fill("http")
    expect(dialog.get_by_role("button", name = "Save changes")).to_be_disabled()
    expect(dialog.get_by_role("button", name = "Test connection")).to_be_disabled()
    screenshot(dialog, "fixed-ambiguous-http-save-disabled")
    address.blur()
    expect(dialog.get_by_text("Environment variables", exact = True)).to_be_visible()
    expect(dialog.get_by_placeholder("Variable name")).to_have_count(0)
    expect(dialog.get_by_role("switch", name = "Use OAuth sign-in")).to_have_count(0)
    screenshot(dialog, "fixed-ambiguous-http-command-clears-credentials")
    dialog.locator("#mcp-url").fill(MCP_PYTHON)
    expect(dialog.get_by_text("Environment variables", exact = True)).to_be_visible()
    expect(dialog.get_by_placeholder("Variable name")).to_have_count(0)
    screenshot(dialog, "fixed-transport-credentials-cleared")
    dialog.get_by_role("button", name = "Cancel").click()

    dialog.get_by_role("button", name = "Close").click()
    composer = page.get_by_role("button", name = "MCP servers")
    expect(composer).to_be_visible()
    composer.click()
    preset = page.get_by_role("menuitem", name = "Unsloth Docs")
    expect(preset).to_be_enabled()
    page.wait_for_timeout(250)
    held_list: list = []

    def hold_first_list(route):
        if route.request.method == "GET" and not held_list:
            held_list.append(route)
        else:
            route.continue_()

    writes = []
    page.on(
        "request",
        lambda request: (
            writes.append(request.method)
            if "/api/mcp/servers/" in request.url and request.method in ("POST", "PUT")
            else None
        ),
    )
    page.route("**/api/mcp/servers/", hold_first_list)
    preset.click()
    expect(preset).to_be_disabled(timeout = 10_000)
    deadline = time.monotonic() + 10
    while (not held_list or writes.count("POST") != 1) and time.monotonic() < deadline:
        page.wait_for_timeout(50)
    assert held_list and writes.count("POST") == 1
    page.get_by_role("menu").screenshot(
        path = str(ART / f"fixed-composer-preset-waits-for-refresh-{BROWSER}.png")
    )
    release_request(page, held_list)
    page.unroute("**/api/mcp/servers/", hold_first_list)
    expect(preset).to_be_enabled(timeout = 10_000)
    preset.click()
    page.wait_for_timeout(500)
    assert writes.count("POST") == 1
    assert writes.count("PUT") == 1
    failed_list = []

    def fail_next_list(route):
        if route.request.method == "GET" and not failed_list:
            failed_list.append(route.request.url)
            route.fulfill(status = 503, json = {"detail": "temporary list failure"})
        else:
            route.continue_()

    page.route("**/api/mcp/servers/", fail_next_list)
    preset.click()
    deadline = time.monotonic() + 10
    while (not failed_list or writes.count("PUT") != 2) and time.monotonic() < deadline:
        page.wait_for_timeout(50)
    assert failed_list and writes.count("PUT") == 2
    expect(preset).to_be_enabled()
    page.get_by_role("menu").screenshot(
        path = str(ART / f"fixed-composer-recovers-after-refresh-error-{BROWSER}.png")
    )
    page.unroute("**/api/mcp/servers/", fail_next_list)
    page.get_by_role("menuitem", name = "Manage MCP servers").press("Enter")
    dialog = page.get_by_role("dialog", name = "MCP Servers")
    expect(dialog).to_be_visible()
    expect(dialog.locator("li").filter(has_text = "Unsloth Docs")).to_have_count(1)
    screenshot(dialog, "fixed-composer-preset-no-duplicate")

    row = row_for(dialog, "Playwright argument echo")
    row.get_by_role("button", name = "Delete server").click()
    expect(page.get_by_role("alertdialog", name = "Delete MCP server")).to_be_visible()
    page.goto(BASE + "/hub", wait_until = "domcontentloaded")
    expect(page).to_have_url(BASE + "/hub")
    expect(page.get_by_role("alertdialog", name = "Delete MCP server")).to_have_count(0)
    page.screenshot(path = str(ART / f"fixed-delete-route-closed-{BROWSER}.png"))

    records = [json.loads(line) for line in launch_log.read_text(encoding = "utf-8").splitlines()]
    expected_runs = [record for record in records if record["marker"] == "playwright edited"]
    assert expected_runs
    assert all(record["arguments"] == edited[1:] for record in expected_runs)
    imported_runs = [record for record in records if record["marker"] == "playwright import"]
    assert imported_runs
    assert all(
        record["arguments"] == ["imported", "", "with spaces", 'say "hello"']
        for record in imported_runs
    )


def main() -> int:
    ART.mkdir(parents = True, exist_ok = True)
    fixture_dir = ART / "fixture directory with spaces"
    fixture_dir.mkdir(parents = True, exist_ok = True)
    fixture = fixture_dir / SOURCE_FIXTURE.name
    shutil.copyfile(SOURCE_FIXTURE, fixture)
    launch_log = ART / "argument launches.jsonl"
    launch_log.unlink(missing_ok = True)
    wait_for_health(BASE, timeout = 60, info = info)
    session = authenticate()
    remove_prior_test_servers(session["access_token"])
    seed_js = (
        "(() => {"
        f"localStorage.setItem('unsloth_auth_token', {json.dumps(session['access_token'])});"
        f"localStorage.setItem('unsloth_refresh_token', {json.dumps(session.get('refresh_token', ''))});"
        "localStorage.setItem('unsloth_chat_mcp_enabled', 'true');"
        "})();"
    )
    install_wall_clock_watchdog(WALL_TIMEOUT_S, label = "mcp-arguments", info = info)
    with sync_playwright() as playwright:
        if BROWSER not in ("chromium", "firefox", "webkit"):
            raise AssertionError(f"unsupported browser: {BROWSER}")
        browser_type = getattr(playwright, BROWSER)
        launch_kwargs: dict = {"headless": True}
        if BROWSER == "chromium":
            launch_kwargs["args"] = chromium_launch_args()
            if CHANNEL:
                launch_kwargs["channel"] = CHANNEL
        browser = browser_type.launch(**launch_kwargs)
        context = browser.new_context(
            viewport = {"width": 1280, "height": 900},
            reduced_motion = "reduce",
        )
        install_view_transition_killer(context)
        context.add_init_script(seed_js)
        page = context.new_page()
        page.set_default_timeout(15_000)
        try:
            run(page, launch_log, fixture)
        finally:
            page.screenshot(path = str(ART / f"fixed-final-{BROWSER}.png"), full_page = True)
            context.close()
            browser.close()
            remove_prior_test_servers(session["access_token"])
    info("PASS real stdio create, edit, import, probe, persistence, reconnect, and launch")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
