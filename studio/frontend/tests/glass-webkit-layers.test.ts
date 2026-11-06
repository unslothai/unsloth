// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const MATERIAL = readSrc("features/chat-background/material.ts");
const CSS = readSrc("index.css");

test("glass surfaces get child layers only on WebKit and only while glass is on", () => {
  assert.match(
    MATERIAL,
    /const WEBKIT_LAYERS =\s*typeof navigator !== "undefined" &&\s*\/AppleWebKit\/\.test\(navigator\.userAgent\) &&\s*!\/Chrom\|Edg\|Android\/i\.test\(navigator\.userAgent\);/,
    "Chromium and WebView2 composite these already; only WebKit re-blurs a surface when its contents repaint",
  );
  assert.match(
    MATERIAL,
    /toggleAttribute\(\s*"data-glass-layers",\s*WEBKIT_LAYERS && material !== "solid" && material !== "transparent" && here,\s*\)/,
    "solid, Transparent and out-of-scope routes never carry the flag, so turning glass off restores the exact previous tree",
  );
});

test("the layer rule keeps containing blocks and never uses :has()", () => {
  const at = CSS.indexOf("html[data-glass][data-glass-layers]");
  assert.ok(at > 0, "the rule exists");
  const rule = CSS.slice(at, CSS.indexOf("}", at) + 1);
  assert.match(rule, /will-change: opacity;/);
  assert.doesNotMatch(rule, /transform/, "a transform would re-anchor absolute and fixed descendants");
  assert.doesNotMatch(rule, /:has\(/);
});
