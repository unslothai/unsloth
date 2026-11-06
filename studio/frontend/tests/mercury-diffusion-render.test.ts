// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Streamdown memoises strong, headings, list items and code by source position, so a diffusion
// snapshot that rewrites a bold heading in place (same span) left the noisy heading on screen.
// The renderer remounts on any non-append rewrite of the reply text.

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc, registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { TextRewriteCounter } = await import(
  "../src/components/assistant-ui/streaming-render-schedule.ts"
);

test("appends keep the generation; an in-place rewrite of an earlier heading bumps it", () => {
  const counter = new TextRewriteCounter();
  const noisy = "**Paragraph y~ vrade-offs and thy >Pnl Ftead** Despite";
  const clean = "**Paragraph 6: Trade-offs and the Path Ahead** Despite";
  assert.equal(noisy.length, clean.length);
  assert.equal(counter.update(noisy.slice(0, 20)), 0);
  assert.equal(counter.update(noisy), 0);
  assert.equal(counter.update(noisy), 0);
  assert.equal(counter.update(clean), 1);
  assert.equal(counter.update(`${clean} these advantages`), 1);
});

test("the rewrite generation is part of the Streamdown key", () => {
  const src = readSrc("components/assistant-ui/markdown-text.tsx");
  assert.match(src, /const rewriteGeneration = incrementalCacheRef\.current\.rewrites\.update\(displayText\);/);
  assert.match(src, /key=\{`\$\{messageId\}:\$\{incrementalCache\.renderGeneration\}:\$\{rewriteGeneration\}:/);
});
