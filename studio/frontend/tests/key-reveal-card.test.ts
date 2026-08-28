// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// #9899: on plain HTTP the clipboard API is blocked, so a newly minted API token must
// stay manually selectable instead of living inside a click-to-copy button.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const source = readFileSync(
  new URL(
    "../src/features/settings/components/key-reveal-card.tsx",
    import.meta.url,
  ),
  "utf8",
);
const en = readFileSync(
  new URL("../src/i18n/locales/en.ts", import.meta.url),
  "utf8",
);

test("the revealed API token is a read-only input, not a button-wrapped code block", () => {
  assert.match(source, /<Input[\s\S]*readOnly[\s\S]*value=\{rawKey\}/);
  assert.doesNotMatch(source, /<button[\s\S]*<code[\s\S]*\{rawKey\}/);
});

test("the token field selects itself on focus and mount for manual copy", () => {
  assert.match(source, /onFocus=\{\(event\) => event\.currentTarget\.select\(\)\}/);
  assert.match(source, /input\.focus\(\{ preventScroll: true \}\);\s*input\.select\(\);/);
});

test("a failed automatic copy toasts and re-selects the token field", () => {
  assert.match(
    source,
    /toast\.error\(t\("settings\.apiKeys\.copyAccessTokenFailed"\)\)/,
  );
  assert.match(
    source,
    /inputRef\.current\?\.focus\(\{ preventScroll: true \}\);\s*inputRef\.current\?\.select\(\);/,
  );
  assert.match(en, /copyAccessTokenFailed:/);
});
