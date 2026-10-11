// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Radix writes to <body> on every modal open. Each rule pinned here turned
 * one of those writes into a full page restyle (~160-185ms at ~15K elements).
 */

import assert from "node:assert/strict";
import { existsSync } from "node:fs";
import test from "node:test";

import { readSrcAsync } from "./helpers/kit.ts";

/** [selector, declarations] of each innermost rule. */
function rules(css: string): [string, string][] {
  const bare = css.replace(/\/\*[\s\S]*?\*\//g, "");
  return [...bare.matchAll(/([^{}]+)\{([^{}]*)\}/g)].map((m) => [
    m[1].replace(/\s+/g, " ").trim(),
    m[2],
  ]);
}

test("dark menus find their dialog or picker by sibling, not body:has()", async () => {
  const css = rules(await readSrcAsync("index.css"));
  const context = css.filter(
    ([, body]) =>
      /--dropdown-glow:/.test(body) || /var\(--dropdown-lift-surface\)/.test(body),
  );
  assert.ok(context.length >= 3, "the glow and lift context rules are present");
  for (const [selector] of context) {
    assert.doesNotMatch(selector, /body:has\(/, selector);
    assert.match(selector, /~ \*/, selector);
  }
});

test("no stylesheet uses a universal sibling selector", async () => {
  for (const file of ["index.css", "features/hub/hub.css"]) {
    for (const [selector] of rules(await readSrcAsync(file))) {
      assert.doesNotMatch(selector, /(^|[\s>(,])\*\s*[+~]\s*\*/, `${file}: ${selector}`);
    }
  }
});

test("body already holds the value react-remove-scroll writes", async () => {
  const css = rules(await readSrcAsync("index.css"));
  assert.ok(
    css.some(
      ([selector, body]) =>
        selector === "body" && /--removed-body-scroll-bar-size:\s*0px;/.test(body),
    ),
  );
});

test("the dark glow is static, with no per-menu probe", async () => {
  assert.equal(
    existsSync(new URL("../src/lib/dropdown-surround.ts", import.meta.url)),
    false,
  );
  assert.doesNotMatch(await readSrcAsync("main.tsx"), /dropdown-surround|watchDropdownSurround/);
});
