// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Chromium reads an oklch() scrollbar-color back as oklab() once a transition has run on it, so the
// next restyle sees a "changed" value and starts another one. Reduce motion (on by default) gives
// every element a 0.01ms `all` transition, and `* { scrollbar-color }` covers every element, so in a
// long chat each restyle started thousands of transitions (typing in Liquid Glass at 200 turns:
// about 13,000 elements restyled instead of 1,300). Legacy rgb()/hex values round-trip unchanged.

import assert from "node:assert/strict";
import { readdirSync, readFileSync } from "node:fs";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

const SRC = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../src");

function cssFiles(dir: string): string[] {
  const out: string[] = [];
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) out.push(...cssFiles(full));
    else if (entry.name.endsWith(".css")) out.push(full);
  }
  return out;
}

const LEGACY = /^(?:auto|transparent|inherit|initial|unset|revert|currentcolor|#[0-9a-f]{3,8}|rgba?\([^()]*\))$/i;

export function nonLegacyScrollbarColors(css: string): string[] {
  const bad: string[] = [];
  for (const m of css.replace(/\/\*[\s\S]*?\*\//g, "").matchAll(/scrollbar-color\s*:\s*([^;}]+)/g)) {
    const value = m[1].replace(/!important/i, "").trim();
    const tokens = value.match(/[a-z-]+\([^()]*\)|#[0-9a-f]+|[a-z-]+/gi) ?? [];
    if (tokens.length === 0 || tokens.some((t) => !LEGACY.test(t))) bad.push(value);
  }
  return bad;
}

test("the checker flags oklch, var() and color-mix, and accepts rgb, hex and keywords", () => {
  assert.deepEqual(nonLegacyScrollbarColors("* { scrollbar-color: oklch(0.5 0 0 / 0.54) transparent; }"), [
    "oklch(0.5 0 0 / 0.54) transparent",
  ]);
  assert.equal(nonLegacyScrollbarColors("a { scrollbar-color: var(--x) transparent }").length, 1);
  assert.equal(nonLegacyScrollbarColors("a { scrollbar-color: color-mix(in srgb, red 5%, blue) red }").length, 1);
  assert.deepEqual(
    nonLegacyScrollbarColors(
      "a { scrollbar-color: rgb(99 99 99 / 0.4) transparent !important; } b { scrollbar-color: auto } c { scrollbar-color: #636363 rgba(0,0,0,0) }",
    ),
    [],
  );
});

test("every scrollbar-color in the Studio stylesheets uses legacy rgb, hex or a keyword", () => {
  const files = cssFiles(SRC);
  assert.ok(files.some((f) => f.endsWith(`${path.sep}index.css`)), "index.css not found under src");
  const offenders = files.flatMap((f) =>
    nonLegacyScrollbarColors(readFileSync(f, "utf8")).map((v) => `${path.relative(SRC, f)}: ${v}`),
  );
  assert.deepEqual(offenders, []);
});
