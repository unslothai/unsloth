// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

test("Settings > Browser > Browser use links to its docs page in a new tab", () => {
  const settings = readSrc("features/settings/tabs/browser-agent-settings.tsx");
  assert.match(settings, /BROWSER_USE_DOCS_URL = "https:\/\/unsloth\.ai\/docs\/basics\/browser-use"/);
  assert.match(settings, /href=\{BROWSER_USE_DOCS_URL\}\s+target="_blank"\s+rel="noreferrer"/);
});

test("the browser suggests the docs at unsloth.ai/docs, separate from the home page", () => {
  const page = readSrc("features/browser/new-tab-page.tsx");
  assert.doesNotMatch(page, /url: "https:\/\/docs\.unsloth\.ai"/);
  assert.equal(page.match(/url: "https:\/\/unsloth\.ai\/docs"/g)?.length, 2);
  // Both live on unsloth.ai, so suggestions are keyed per site, not per host.
  assert.match(
    page,
    /if \(host === "docs\.unsloth\.ai"\) return "unsloth\.ai\/docs";/,
  );
  assert.doesNotMatch(page, /hosts\.has\(hostOf\(/);
  assert.match(page, /hideSuggestion\(siteKey\(site\.url\)\)/);
});
