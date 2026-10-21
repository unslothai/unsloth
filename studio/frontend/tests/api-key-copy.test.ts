// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const ROW = readSrc("features/settings/components/api-key-row.tsx");
const REVEAL = readSrc("features/settings/components/key-reveal-card.tsx");

test("a copied key prefix carries no display ellipsis", () => {
  // The row used to copy "sk-unsloth-abcd1234…", which no client accepts.
  assert.match(ROW, /const prefix = `sk-unsloth-\$\{apiKey\.key_prefix\}`;/);
  assert.match(
    ROW,
    /\{prefix\}…\s*<\/code>/,
    "the row no longer shows the ellipsis",
  );
  assert.match(ROW, /copyToClipboard\(prefix\)/);
});

test("copying a key prefix says whether it worked", () => {
  // The menu closes on select, so a toast is the only feedback there is.
  assert.match(ROW, /toast\.success\(t\("settings\.apiKeys\.copied"\)\)/);
  assert.match(ROW, /toast\.error\(t\("settings\.apiKeys\.copyFailed"\)\)/);
});

test("a new key opens showing its start, fully selected", () => {
  // select() scrolls a long key to its end, hiding the sk-unsloth- start.
  assert.match(
    REVEAL,
    /input\.select\(\);[\s\S]*?input\.scrollLeft = 0;/,
    "the key opens scrolled to its end",
  );
  // Refocusing the field selects it the same way.
  assert.match(
    REVEAL,
    /onFocus=\{\(event\) => selectToken\(event\.currentTarget\)\}/,
  );
});
