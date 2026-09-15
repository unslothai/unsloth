// SPDX-License-Identifier: AGPL-3.0-only
// Start the frontend Vite server, then run: node tests/studio/playwright_research_review.mjs
// Requires playwright; SMOKE_BASE_URL and SMOKE_BROWSER (chromium/webkit) are optional.
import assert from "node:assert/strict";
import { createRequire } from "node:module";
const { chromium, webkit } = createRequire(import.meta.url)("playwright");
const browser = await (process.env.SMOKE_BROWSER === "webkit" ? webkit : chromium).launch();
try {
  const page = await browser.newPage();
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  const requests = [];
  await page.route((url) => url.pathname.startsWith("/api/"), async (route) => {
    requests.push(route.request().url());
    await route.fulfill({ status: 500, body: "Unexpected API request" });
  });
  await page.goto(`${process.env.SMOKE_BASE_URL || "http://127.0.0.1:5184"}/smoke-research-review.html`);
  const dialog = page.getByRole("dialog", { name: "Review the research plan" });
  await dialog.waitFor({ state: "visible" });
  await page.keyboard.press("Escape");
  await dialog.waitFor({ state: "hidden" });
  await page.evaluate(() => { window.__review.setDraft(); window.__review.setError(); });
  const before = await page.evaluate(() => window.__review.state());
  // With the panel already open, the old handler is a complete no-op after dismissal.
  await page.getByTestId("research-message").getByRole("button", { name: "Review plan", exact: true }).click();
  await dialog.waitFor({ state: "visible", timeout: 5000 });
  const after = await page.evaluate(() => window.__review.state());
  assert.deepEqual(after.sessions, before.sessions, "Opening review must not change connection/run state");
  assert.deepEqual(after.planReviewByRunId["review-run"], { ...before.planReviewByRunId["review-run"], open: true });
  // Also reopen after the activity panel has been unmounted.
  await page.keyboard.press("Escape");
  await dialog.waitFor({ state: "hidden" });
  await page.evaluate(() => window.__review.closePanel());
  await page.getByTestId("research-message").getByRole("button", { name: "Review plan", exact: true }).click();
  await dialog.waitFor({ state: "visible" });
  await page.keyboard.press("Escape");
  await dialog.waitFor({ state: "hidden" });
  // Ordinary View activity keeps the review dismissed and preserves the session.
  await page.evaluate(() => { window.__review.complete(); window.__review.closePanel(); });
  const completed = await page.evaluate(() => window.__review.state());
  await page.getByTestId("research-message").getByRole("button", { name: /View activity/ }).click();
  const viewed = await page.evaluate(() => window.__review.state());
  assert.equal(viewed.openRunId, "review-run");
  assert.deepEqual(viewed.sessions, completed.sessions);
  assert.deepEqual(viewed.planReviewByRunId, completed.planReviewByRunId);
  assert.equal(await dialog.count(), 0);
  assert.deepEqual(requests, [], "Reviewing a plan must not make API requests");
  assert.deepEqual(errors, []);
  console.log("PASS: dismissed plan reopens; draft, connection and activity behavior preserved");
} finally {
  await browser.close();
}
