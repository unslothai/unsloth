// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { defineConfig, devices } from "@playwright/test";

const frontendPort = Number(process.env.E2E_FRONTEND_PORT ?? 5199);
const backendPort = Number(process.env.E2E_BACKEND_PORT ?? 8888);
const frontendBaseUrl =
  process.env.E2E_BASE_URL ?? `http://127.0.0.1:${frontendPort}`;

export default defineConfig({
  testDir: ".",
  testMatch: "**/*.spec.ts",
  fullyParallel: false,
  forbidOnly: Boolean(process.env.CI),
  retries: process.env.CI ? 1 : 0,
  workers: 1,
  reporter: [["list"], ["html", { open: "never" }]],
  timeout: 10 * 60 * 1000,
  expect: {
    timeout: 120_000,
  },
  use: {
    baseURL: frontendBaseUrl,
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
    video: "retain-on-failure",
    ...devices["Desktop Chrome"],
  },
  globalSetup: "./global-setup.ts",
  webServer: [
    {
      command: `npm run dev -- --host 127.0.0.1 --port ${frontendPort} --strictPort`,
      url: frontendBaseUrl,
      reuseExistingServer: true,
      timeout: 180_000,
      cwd: "..",
    },
  ],
  metadata: {
    backendUrl: `http://127.0.0.1:${backendPort}`,
  },
});
