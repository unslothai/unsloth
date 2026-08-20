// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { once } from "node:events";
import path from "node:path";
import { fileURLToPath } from "node:url";
import puppeteer from "puppeteer-core";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const frontendRoot = process.env.FRONTEND_ROOT
  ? path.resolve(process.env.FRONTEND_ROOT)
  : path.resolve(__dirname, "../..");
const mockServerPath = process.env.MOCK_SERVER_PATH
  ? path.resolve(process.env.MOCK_SERVER_PATH)
  : path.join(__dirname, "profile-stats-mock-server.mjs");
const vitePort = Number(process.env.VITE_PORT ?? 8000);
const mockPort = Number(process.env.MOCK_API_PORT ?? 8888);
const baseUrl = `http://127.0.0.1:${vitePort}`;
const chromiumPath =
  process.env.CHROMIUM_PATH ?? "/snap/bin/chromium";

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function waitForUrl(url, attempts = 60) {
  for (let index = 0; index < attempts; index += 1) {
    try {
      const response = await fetch(url);
      if (response.ok || response.status === 404) return;
    } catch {
      // retry
    }
    await sleep(500);
  }
  throw new Error(`Timed out waiting for ${url}`);
}

function spawnLogged(command, args, options = {}) {
  const child = spawn(command, args, {
    stdio: ["ignore", "pipe", "pipe"],
    ...options,
  });
  child.stdout.on("data", (chunk) => process.stdout.write(chunk));
  child.stderr.on("data", (chunk) => process.stderr.write(chunk));
  return child;
}

async function main() {
  const mock = spawnLogged("node", [mockServerPath], {
    cwd: frontendRoot,
    env: { ...process.env, MOCK_API_PORT: String(mockPort) },
  });

  const vite = spawnLogged(
    "npm",
    ["run", "dev", "--", "--port", String(vitePort), "--strictPort"],
    { cwd: frontendRoot },
  );

  try {
    await waitForUrl(`${baseUrl}/`);
    await waitForUrl(`http://127.0.0.1:${mockPort}/api/health`);

    const browser = await puppeteer.launch({
      executablePath: chromiumPath,
      headless: true,
      args: ["--no-sandbox", "--disable-setuid-sandbox", "--disable-dev-shm-usage"],
    });
    const page = await browser.newPage();

    await page.evaluateOnNewDocument(() => {
      localStorage.setItem("unsloth_auth_token", "test-access-token");
      localStorage.setItem("unsloth_auth_refresh_token", "test-refresh-token");
      localStorage.setItem("unsloth_onboarding_done", "1");
      localStorage.setItem("unsloth_settings_active_tab", "profile");
    });

    await page.goto(`${baseUrl}/settings`, { waitUntil: "networkidle0", timeout: 60_000 });

    await page.waitForFunction(
      () => document.querySelector('[data-testid="settings-tab-profile"]') !== null,
      { timeout: 30_000 },
    );
    await page.click('[data-testid="settings-tab-profile"]');
    await page.waitForFunction(
      () => document.body.textContent?.includes("Token activity") ?? false,
      { timeout: 30_000 },
    );

    const readDescription = async () => {
      return page.evaluate(() => {
        const heading = [...document.querySelectorAll("h3")].find((node) =>
          node.textContent?.includes("Token activity"),
        );
        const description = heading?.parentElement?.querySelector("p");
        return description?.textContent?.trim() ?? "";
      });
    };

    const clickMode = async (label) => {
      const clicked = await page.evaluate((modeLabel) => {
        const button = [...document.querySelectorAll("button")].find(
          (node) => node.textContent?.trim() === modeLabel,
        );
        if (!button) return false;
        button.click();
        return true;
      }, label);
      assert.equal(clicked, true, `missing ${label} mode button`);
    };

    const dailyText = await readDescription();
    assert.match(dailyText, /over the last/i);
    assert.match(dailyText, /token/i);

    await clickMode("Weekly");
    const weeklyText = await readDescription();
    assert.match(weeklyText, /Peak week/i);
    assert.notEqual(weeklyText, dailyText);

    await clickMode("Cumulative");
    const cumulativeText = await readDescription();
    assert.match(cumulativeText, /accumulated/i);
    assert.notEqual(cumulativeText, dailyText);
    assert.notEqual(cumulativeText, weeklyText);

    await clickMode("Daily");
    const dailyAgain = await readDescription();
    assert.equal(dailyAgain, dailyText);

    console.log("profile stats UI verification passed");
    console.log(`  daily: ${dailyText}`);
    console.log(`  weekly: ${weeklyText}`);
    console.log(`  cumulative: ${cumulativeText}`);

    await browser.close();
  } finally {
    vite.kill("SIGTERM");
    mock.kill("SIGTERM");
    await Promise.allSettled([once(vite, "exit"), once(mock, "exit")]);
  }
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
