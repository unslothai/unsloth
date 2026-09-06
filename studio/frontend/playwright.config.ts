// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { defineConfig, devices } from "@playwright/test";

// Browser journeys against a disposable Studio server. The specs read
// STUDIO_E2E_URL (and STUDIO_E2E_OWNER_PASSWORD where a login is needed) and
// skip when the server is not there, so `npm test` never depends on this file.
export default defineConfig({
  testDir: "./e2e",
  outputDir: process.env.PLAYWRIGHT_OUTPUT_DIR ?? "test-results/playwright",
  fullyParallel: false,
  workers: 1,
  retries: process.env.CI ? 1 : 0,
  reporter: [["list"]],
  use: {
    baseURL: process.env.STUDIO_E2E_URL ?? "http://127.0.0.1:8000",
    ...devices["Desktop Chrome"],
    // Bundled Chromium unless a branded channel ("chrome", "msedge") is named;
    // the branded builds differ in codecs, policies and storage partitioning.
    channel: process.env.PLAYWRIGHT_CHANNEL || undefined,
    viewport: { width: 1440, height: 900 },
    colorScheme: "light",
    screenshot: "only-on-failure",
    trace: "retain-on-failure",
  },
});
