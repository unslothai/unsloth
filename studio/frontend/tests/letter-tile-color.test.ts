// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

// The module needs Vite env and React, so mirror its hash here and pin the source to it.
const src = readSrc("features/chat/api-provider-logo.tsx");
function color(name: string): { l: number; h: number } {
  let hash = 0x811c9dc5;
  for (const ch of name.toLowerCase()) {
    hash = Math.imul(hash ^ ch.charCodeAt(0), 0x01000193);
  }
  hash ^= hash >>> 16;
  hash = Math.imul(hash, 0x85ebca6b);
  hash ^= hash >>> 13;
  hash = Math.imul(hash, 0xc2b2ae35);
  hash ^= hash >>> 16;
  const h = hash >>> 0;
  return { l: 0.52 + ((h >>> 20) % 4) * 0.04, h: h % 360 };
}

test("letter tiles take a hue anywhere on the wheel, stable per name", () => {
  assert.match(src, /hash = Math\.imul\(hash, 0x85ebca6b\);/);
  assert.match(
    src,
    /const lightness = 0\.52 \+ \(\(h >>> 20\) % 4\) \* 0\.04;/,
  );
  assert.match(
    src,
    /return `oklch\(\$\{lightness\.toFixed\(2\)\} 0\.15 \$\{h % 360\}\)`;/,
  );
  // Even over the wheel: 600 model-like names fill each 60-degree band within a third of its share.
  const counts = new Array(6).fill(0);
  for (let i = 0; i < 600; i++) {
    const { l, h } = color(`model-${i}-GGUF`);
    counts[Math.floor(h / 60)]++;
    assert.ok(l >= 0.52 && l <= 0.64, `lightness ${l}`);
  }
  for (const c of counts) {
    assert.ok(c > 66 && c < 134, `uneven bands: ${counts.join(", ")}`);
  }
  assert.deepEqual(color("flint-GGUF"), color("FLINT-gguf"));
});

test("the tile redraws once fonts load and when it resizes", () => {
  assert.match(src, /document\.fonts\?\.ready\.then\(draw\)/);
  assert.match(src, /new ResizeObserver\(draw\)/);
});
