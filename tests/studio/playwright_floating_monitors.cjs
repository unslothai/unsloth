// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Run against Vite: BASE_URL=http://127.0.0.1:5173 node tests/studio/playwright_floating_monitors.cjs
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const { chromium } = require("playwright");

const base = process.env.BASE_URL || "http://127.0.0.1:5173";
const out = process.env.PW_ART_DIR || "logs/floating-monitors";
const near = (a, b, label) => assert.ok(Math.abs(a - b) < 2, `${label}: ${a} vs ${b}`);
const frames = (page) => page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));

async function drag(page, handle, dx, dy) {
  const b = await handle.boundingBox();
  await page.mouse.move(b.x + b.width / 2, b.y + b.height / 2);
  await page.mouse.down();
  await page.mouse.move(b.x + b.width / 2 + dx, b.y + b.height / 2 + dy, { steps: 12 });
  await page.mouse.up();
  await frames(page);
}

async function resize(page, panel, dx, dy) {
  const b = await panel.boundingBox();
  await page.mouse.move(b.x + b.width - 3, b.y + b.height - 3);
  await page.mouse.down();
  await page.mouse.move(b.x + b.width - 3 + dx, b.y + b.height - 3 + dy, { steps: 12 });
  await page.mouse.up();
  await frames(page);
  return panel.boundingBox();
}

(async () => {
  fs.mkdirSync(out, { recursive: true });
  const browser = await chromium.launch({ headless: true, ...(process.env.PW_CHANNEL ? { channel: process.env.PW_CHANNEL } : {}) });
  try {
    const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
    const errors = [];
    page.on("pageerror", error => errors.push(error.message));
    let rows = 1;
    await page.route("**/api/**", route => {
      const url = new URL(route.request().url());
      if (!url.pathname.startsWith("/api/")) return route.continue();
      const data = url.pathname === "/api/system" ? {
        platform: "Windows", memory: { total_gb: 32, available_gb: 16, percent_used: 50 },
        gpu: { available: true, backend: "cuda", devices: [{ name: "Test GPU", memory_total_gb: 16, vram_used_gb: 8 }] },
      } : {
        status: "ready", active_model: "Test model", active_requests: 0,
        entries: Array.from({ length: rows }, (_, i) => ({
          id: String(i), endpoint: "/v1/chat/completions", method: "POST", model: "Test model",
          via_api_key: false, status: "completed", started_at: 1, updated_at: 2, duration_ms: 1000,
          prompt_preview: "", reply_preview: "", prompt_truncated: false, reply_truncated: false,
        })),
      };
      return route.fulfill({ json: data });
    });
    await page.goto(`${base}/smoke-floating-monitors.html`);
    await page.getByRole("button", { name: "Open API", exact: true }).click();
    // Semantic anchor also works against the unmodified panel for negative proof.
    const api = page.locator(".menu-soft-surface").filter({ has: page.getByRole("button", { name: "Close API monitor" }) });
    const handle = api.locator(".cursor-grab");
    await api.getByText("Test model", { exact: true }).first().waitFor();
    await page.waitForTimeout(400);
    const initial = await api.boundingBox();
    const firstResize = await resize(page, api, 80, 60);
    fs.writeFileSync(path.join(out, "first-resize.json"), JSON.stringify({ initial, firstResize }, null, 2));
    near(firstResize.x, initial.x, "unmoved resize X");
    near(firstResize.y, initial.y, "unmoved resize Y");
    await drag(page, handle, -500, -400);
    const moved = await api.boundingBox();
    near(moved.x, initial.x - 500, "drag X");
    near(moved.y, initial.y - 400, "drag Y");
    const grown = await resize(page, api, 170, 110);
    near(grown.x, moved.x, "resize keeps X");
    near(grown.y, moved.y, "resize keeps Y");
    assert.ok(grown.width > moved.width + 100 && grown.height > moved.height + 60, "native resize must actually grow");
    rows = 4;
    await api.getByText("/chat/completions", { exact: true }).nth(3).waitFor();
    near((await api.boundingBox()).y, grown.y, "polling keeps user position");
    const constrained = await resize(page, api, 2000, 2000);
    near(constrained.x, grown.x, "edge resize keeps X");
    near(constrained.y, grown.y, "edge resize keeps Y");
    assert.ok(constrained.x + constrained.width <= 1425 && constrained.y + constrained.height <= 985);
    await drag(page, handle, -2000, -2000);
    const corner = await api.boundingBox();
    near(corner.x, 16, "left boundary");
    near(corner.y, 64, "top boundary");
    await page.setViewportSize({ width: 390, height: 400 });
    await frames(page);
    const compact = await api.boundingBox();
    assert.ok(compact.width <= 358 && compact.height <= 320);
    const scroll = api.locator(".overflow-y-auto");
    assert.ok(await scroll.evaluate(el => el.scrollHeight > el.clientHeight), "short panel must scroll");
    assert.ok((await api.getByRole("button", { name: "Expand to full monitor" }).boundingBox()).height >= 25);
    await page.screenshot({ path: path.join(out, "api-compact.png") });
    await api.getByRole("button", { name: "Close API monitor" }).click();
    await api.waitFor({ state: "detached" });
    await page.setViewportSize({ width: 1440, height: 1000 });
    await page.getByRole("button", { name: "Open API", exact: true }).click();
    await api.waitFor();
    await page.waitForTimeout(400);
    const reopened = await api.boundingBox();
    near(reopened.width, initial.width, "reopen resets native resize");
    await drag(page, handle, -400, -250);
    await resize(page, api, 120, 80);
    await api.getByRole("button", { name: "Close API monitor" }).click();
    await page.getByRole("button", { name: "Open API", exact: true }).click();
    await page.waitForTimeout(400);
    assert.equal(await api.count(), 1);
    near((await api.boundingBox()).width, initial.width, "rapid reopen resets resize");
    await api.getByRole("button", { name: "Close API monitor" }).click();
    await api.waitFor({ state: "detached" });
    await page.getByRole("button", { name: "Open hardware", exact: true }).click();
    const hardware = page.getByTestId("floating-monitor");
    await hardware.getByText("Test GPU", { exact: false }).waitFor();
    await page.waitForTimeout(400);
    await drag(page, page.getByTestId("floating-monitor-drag-handle"), -500, -400);
    const hardwareBefore = await hardware.boundingBox();
    const hardwareAfter = await resize(page, hardware, 200, 150);
    near(hardwareBefore.x, hardwareAfter.x, "hardware X");
    near(hardwareBefore.y, hardwareAfter.y, "hardware Y");
    assert.ok(hardwareAfter.width > hardwareBefore.width + 100, JSON.stringify({ hardwareBefore, hardwareAfter }));
    await page.getByRole("button", { name: "Open API", exact: true }).click();
    await api.waitFor();
    await page.waitForTimeout(400);
    const bothApi = await api.boundingBox();
    const overlap = Math.min(bothApi.x + bothApi.width, hardwareAfter.x + hardwareAfter.width) > Math.max(bothApi.x, hardwareAfter.x)
      && Math.min(bothApi.y + bothApi.height, hardwareAfter.y + hardwareAfter.height) > Math.max(bothApi.y, hardwareAfter.y);
    assert.equal(overlap, false, "initial placement avoids hardware");
    await page.screenshot({ path: path.join(out, "both-monitors.png") });
    await drag(page, page.getByTestId("floating-monitor-drag-handle"), -600, 0);
    const afterObstacleMove = await api.boundingBox();
    near(afterObstacleMove.x, bothApi.x, "moving hardware must not relocate API X");
    near(afterObstacleMove.y, bothApi.y, "moving hardware must not relocate API Y");
    assert.deepEqual(errors, []);
    const facts = { initial, firstResize, moved, grown, constrained, corner, compact, reopened, hardwareBefore, hardwareAfter, bothApi };
    fs.writeFileSync(path.join(out, "facts.json"), JSON.stringify(facts, null, 2));
    console.log(JSON.stringify(facts, null, 2));
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
