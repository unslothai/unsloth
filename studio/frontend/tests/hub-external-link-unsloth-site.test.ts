// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Links to unsloth.ai open without the external-link warning; every other site still asks.

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc, registerStoreStubResolver } from "./helpers/kit.ts";

registerStoreStubResolver();

const { isUnslothSiteUrl } = await import(
  "../src/features/hub/stores/external-link-confirm.ts"
);

test("unsloth.ai and its subdomains are Unsloth's own site", () => {
  for (const url of [
    "https://unsloth.ai/docs/models/nemotron-3.5-bolt",
    "https://docs.unsloth.ai/get-started",
    "http://UNSLOTH.AI",
  ]) {
    assert.equal(isUnslothSiteUrl(url), true, url);
  }
});

test("look-alikes and other schemes still get the warning", () => {
  for (const url of [
    "https://unsloth.ai.evil.example/docs",
    "https://evil.example/unsloth.ai",
    "https://unsloth.ai@evil.example/",
    "https://notunsloth.ai/",
    "javascript:alert('unsloth.ai')",
    "mailto:hi@unsloth.ai",
    "not a url",
  ]) {
    assert.equal(isUnslothSiteUrl(url), false, url);
  }
});

test("the trusted path opens through openLink, the rest through the dialog", () => {
  const store = readSrc("features/hub/stores/external-link-confirm.ts");
  assert.match(store, /if \(isUnslothSiteUrl\(trimmed\)\) \{\s*return openLink\(trimmed\);\s*\}\s*useExternalLinkConfirm\.getState\(\)\.request\(trimmed\);/);
});
