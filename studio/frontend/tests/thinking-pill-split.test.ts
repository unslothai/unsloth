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

const COMPOSERS = [
  { path: "../src/components/assistant-ui/thread.tsx", flag: "splitPill" },
  { path: "../src/features/chat/shared-composer.tsx", flag: "splitThinkingPill" },
];

function read(path: string): string {
  return readFileSync(new URL(path, import.meta.url), "utf8");
}

test("both composers render the icon that narrow layout reveals", () => {
  for (const { path } of COMPOSERS) {
    assert.ok(
      read(path).includes("unsloth-thinking-split-icon"),
      `${path} drops the narrow-layout icon`,
    );
  }
});

test("the toggle half still answers clicks while the menu is open", () => {
  // Radix's modal menu drops pointer events outside its content. The open
  // trigger opts back in, so the toggle half has to as well, or half of one
  // pill goes dead exactly while it is lit as one pill.
  assert.match(
    block(
      css,
      '.unsloth-thinking-split:has([data-state="open"]) .unsloth-thinking-split-toggle {',
    ),
    /pointer-events:\s*auto/,
  );
});

test("only a split pill gets the wrapper that paints the background", () => {
  for (const { path, flag } of COMPOSERS) {
    const source = read(path);
    const at = source.indexOf('className="unsloth-thinking-split"');
    assert.notEqual(at, -1, `${path} drops the split wrapper`);
    // The wrapper holds the hover background, and it is not the button, so
    // disabled:opacity-40 cannot dim it. Wrapping a lone trigger would light
    // an unloaded model's pill at full strength.
    assert.match(
      source.slice(Math.max(0, at - 160), at),
      new RegExp(`${flag} \\?`),
      `${path} wraps the trigger when the pill is not split`,
    );
  }
});

test("the toggle half is wired to the thinking toggle", () => {
  for (const { path } of COMPOSERS) {
    const source = read(path);
    const at = source.indexOf("unsloth-thinking-split-toggle");
    assert.notEqual(at, -1, `${path} drops the toggle half`);
    const button = source.slice(at, source.indexOf("</button>", at));
    assert.match(
      button,
      /onClick=\{toggleThinking\}/,
      `${path} left half no longer toggles thinking`,
    );
  }
});
