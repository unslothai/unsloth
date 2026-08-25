// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * The split Thinking pill puts the options behind the caret, and the narrow
 * layout hides that caret. Without the swap below, the menu (Preserve thinking,
 * and the effort rows) becomes unreachable there.
 */

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const css = readFileSync(new URL("../src/index.css", import.meta.url), "utf8");

function block(source: string, opener: string): string {
  const start = source.indexOf(opener);
  assert.notEqual(start, -1, `missing ${opener}`);
  let depth = 0;
  for (let i = start; i < source.length; i++) {
    if (source[i] === "{") depth++;
    else if (source[i] === "}" && --depth === 0) return source.slice(start, i);
  }
  throw new Error(`unterminated ${opener}`);
}

test("the narrow layout swaps the split pill back to a single trigger", () => {
  const narrow = block(css, "@container (max-width: 36rem)");
  assert.match(narrow, /\.unsloth-thinking-caret\s*{[^}]*display:\s*none/);
  assert.match(
    block(narrow, ".unsloth-thinking-split-toggle {"),
    /display:\s*none/,
  );
  assert.match(
    block(narrow, ".unsloth-thinking-split-icon {"),
    /display:\s*block/,
  );
});

test("both composers render the icon that narrow layout reveals", () => {
  for (const path of [
    "../src/components/assistant-ui/thread.tsx",
    "../src/features/chat/shared-composer.tsx",
  ]) {
    const source = readFileSync(new URL(path, import.meta.url), "utf8");
    assert.ok(
      source.includes("unsloth-thinking-split-icon"),
      `${path} drops the narrow-layout icon`,
    );
  }
});
