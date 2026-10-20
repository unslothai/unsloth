// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Run against a built, running Unsloth Studio with STUDIO_BASE_URL and STUDIO_PASSWORD_FILE.
import assert from "node:assert/strict";
import { readFileSync, mkdirSync } from "node:fs";
import { createRequire } from "node:module";
import path from "node:path";

const { chromium, firefox, webkit, expect } = createRequire(import.meta.url)("playwright/test");
const base = process.env.STUDIO_BASE_URL ?? "http://127.0.0.1:8888";
const artifacts = process.env.PW_ART_DIR ?? "logs/browser-bookmarks";
const login = await fetch(`${base}/api/auth/login`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    username: "unsloth",
    password: readFileSync(process.env.STUDIO_PASSWORD_FILE, "utf8").trim(),
  }),
});
assert.equal(login.status, 200, "test account login succeeds");
const tokens = await login.json();
assert.equal(tokens.must_change_password, false, "use a test account that has completed setup");
mkdirSync(artifacts, { recursive: true });
const first = "https://bookmark.test/first";
const second = "https://bookmark.test/followed";
let failures = 0;

for (const name of (process.env.PW_BROWSERS ?? "chromium,firefox,webkit").split(",")) {
  const browser = await ({ chromium, firefox, webkit })[name].launch();
  try {
    for (const action of ["Save", "Enter"]) {
      for (const followed of [false, true]) {
        const context = await browser.newContext({ viewport: { width: 1440, height: 900 }, locale: "en-US" });
        const page = await context.newPage();
        const label = `${name}-${action}-${followed ? "followed" : "typed"}`;
        try {
          await context.addInitScript(({ access_token, refresh_token }) => {
            localStorage.setItem("unsloth_auth_token", access_token);
            localStorage.setItem("unsloth_refresh_token", refresh_token);
          }, tokens);
          await page.route("**/api/browser/fetch", async (route) => {
            const { url } = route.request().postDataJSON();
            const title = url === first ? "Typed page" : "Followed page";
            await route.fulfill({
              headers: { "Content-Type": "application/json", "X-Unsloth-Browser-Kind": "html" },
              body: JSON.stringify({
                url, base: url, refresh: null,
                html: `<title>${title}</title><a href="${second}">Follow link</a><div style="height:600px"></div><label>Page input<input aria-label="Page input"></label><div style="height:3000px"></div>`,
              }),
            });
          });
          await page.goto(`${base}/chat`, { waitUntil: "domcontentloaded" });
          await page.getByRole("button", { name: "New tab", exact: true }).click();
          const panel = page.getByRole("region", { name: "Browser", exact: true });
          const address = panel.getByRole("textbox", { name: "Search or enter a URL" });
          await address.fill(first);
          await address.press("Enter");
          const content = panel.frameLocator('iframe[src*="/api/browser/frame"]').frameLocator("iframe");
          await expect(content.getByRole("link", { name: "Follow link" })).toBeVisible();
          if (followed) await content.getByRole("link", { name: "Follow link" }).click();
          const current = followed ? second : first;
          await expect(address).toHaveValue(current);
          await expect(panel.getByRole("tab")).toContainText(followed ? "Followed page" : "Typed page");
          await content.getByRole("textbox", { name: "Page input" }).fill("keep this input");
          const frame = page.frames().find((candidate) => candidate.url() === "about:srcdoc");
          await frame.evaluate(() => window.scrollTo(0, 500));
          const scroll = await frame.evaluate(() => window.scrollY);
          assert.ok(scroll > 0);
          const shell = await panel.locator('iframe[src*="/api/browser/frame"]').getAttribute("src");
          let loads = 0;
          page.on("framenavigated", (navigated) => {
            if (navigated.parentFrame()) loads++;
          });
          await panel.getByRole("button", { name: "Bookmark this page", exact: true }).click();
          const editor = page.getByRole("dialog", { name: "", exact: true }).filter({ has: page.locator("[data-bookmark-name]") });
          await expect(editor.getByRole("textbox", { name: "Name", exact: true })).toHaveValue(followed ? "Followed page" : "Typed page");
          if (action === "Enter") await editor.getByRole("textbox", { name: "Name", exact: true }).press("Enter");
          else await editor.getByRole("button", { name: "Save", exact: true }).click();
          await expect(editor).toBeHidden();
          await expect(address).toHaveValue(current);
          await expect(content.getByRole("textbox", { name: "Page input" })).toHaveValue("keep this input");
          assert.equal(await frame.evaluate(() => window.scrollY), scroll);
          assert.equal(await panel.locator('iframe[src*="/api/browser/frame"]').getAttribute("src"), shell);
          assert.equal(loads, 0, "bookmarking must not load another frame");
          const bookmarks = await page.evaluate(() => Object.keys(localStorage)
            .filter((key) => key.startsWith("unsloth_browser_bookmarks"))
            .flatMap((key) => JSON.parse(localStorage.getItem(key)).state.bookmarks));
          assert.equal(bookmarks.length, 1);
          assert.equal(bookmarks[0].url, current);
          assert.equal(bookmarks[0].title, followed ? "Followed page" : "Typed page");
          if (followed && action === "Save") {
            await panel.getByRole("button", { name: "Edit bookmark", exact: true }).click();
            const name = editor.getByRole("textbox", { name: "Name", exact: true });
            await name.fill("Renamed page");
            for (const folder of ["Other bookmarks", "Bookmarks toolbar"]) {
              await editor.getByRole("combobox").click();
              await page.getByRole("option", { name: folder, exact: true }).click();
              await expect(address).toHaveValue(current);
            }
            const checkbox = editor.getByRole("checkbox", { name: "Show editor when saving" });
            await checkbox.uncheck();
            await checkbox.check();
            await name.press("Escape");
            await expect(editor).toBeHidden();
            await panel.getByRole("button", { name: "Edit bookmark", exact: true }).click();
            await expect(name).toHaveValue("Renamed page");
            await editor.getByRole("button", { name: "Remove bookmark", exact: true }).click();
            await expect(editor).toBeHidden();
            await panel.getByRole("button", { name: "Bookmark this page", exact: true }).click();
            await editor.getByRole("button", { name: "Cancel", exact: true }).click();
            await expect(editor).toBeHidden();
            for (const shortcut of ["Control+d", "Meta+d"]) {
              await panel.getByRole("button", { name: "Bookmark this page", exact: true }).press(shortcut);
              await expect(name).toHaveValue("Followed page");
              await editor.getByRole("button", { name: "Save", exact: true }).click();
              await panel.getByRole("button", { name: "Edit bookmark", exact: true }).click();
              await editor.getByRole("button", { name: "Remove bookmark", exact: true }).click();
            }
            await panel.getByRole("button", { name: "Site information", exact: true }).click();
            await page.getByRole("button", { name: "Connection secure", exact: true }).click();
            await page.getByRole("button", { name: "Back", exact: true }).last().click();
            await page.getByRole("button", { name: "Clear browsing data", exact: true }).click();
            await page.getByRole("button", { name: "Cancel", exact: true }).click();
            await expect(address).toHaveValue(current);
            await expect(content.getByRole("textbox", { name: "Page input" })).toHaveValue("keep this input");
            assert.equal(await frame.evaluate(() => window.scrollY), scroll);
            assert.equal(loads, 0, "editor and site popover controls must not navigate");
          }
          await page.screenshot({ path: path.join(artifacts, `${label}.png`) });
          if (followed) {
            await panel.getByRole("button", { name: "Back", exact: true }).click();
            await expect(address).toHaveValue(first);
            await panel.getByRole("button", { name: "Forward", exact: true }).click();
            await expect(address).toHaveValue(second);
          }
          console.log(`PASS ${label}`);
        } catch (error) {
          failures++;
          await page.screenshot({ path: path.join(artifacts, `${label}-failed.png`) });
          console.error(`FAIL ${label}: ${error.message}`);
        } finally {
          await context.close();
        }
      }
    }
  } finally {
    await browser.close();
  }
}
assert.equal(failures, 0, "bookmark browser regressions");
