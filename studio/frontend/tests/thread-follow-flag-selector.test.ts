// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The autoscroll owner flips `data-aui-following` at the start and end of every follow window (any
// width change: run settings, browser pane, sidebar). WebKit restyles every descendant of the
// flipped element's later siblings when a rule reaches its subject through `+`/`~` and a descendant
// combinator, which on a 150-turn thread cost 40-70 ms per flip. The flag therefore sits on the
// viewport and is matched only as the subject.

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const HOOK = readSrc("components/assistant-ui/use-intent-aware-autoscroll.tsx");
const THREAD = readSrc("components/assistant-ui/thread.tsx");
const CSS = readSrc("index.css");

function followSelectors(): string[] {
  const out: string[] = [];
  const re = /([^{}]*\[data-aui-following\][^{}]*)\{/g;
  for (const m of CSS.matchAll(re)) {
    const prelude = m[1].replace(/\/\*[\s\S]*?\*\//g, "").trim();
    for (const sel of prelude.split(",")) out.push(sel.trim());
  }
  return out;
}

test("the follow flag is matched on the viewport itself", () => {
  const selectors = followSelectors();
  assert.ok(selectors.length > 0, "no data-aui-following rule left; the overflow-anchor guard is gone");
  for (const sel of selectors) {
    const at = sel.indexOf("[data-aui-following]");
    const subject = sel.slice(at + "[data-aui-following]".length);
    assert.doesNotMatch(subject, /[\s+~>]/, `${sel}: the flag must be on the subject compound, not an ancestor or sibling`);
    assert.match(sel, /\.aui-thread-viewport\[data-aui-following\]$/, `${sel}: the flag belongs to the viewport`);
  }
});

test("the hook flips the flag on the viewport, never on the portal host", () => {
  assert.match(HOOK, /el\.toggleAttribute\("data-aui-following", on\)/);
  assert.doesNotMatch(HOOK, /markerElRef|markerRef/);
  assert.doesNotMatch(THREAD, /markerRef/);
  // Every write goes through the one setter, teardown included.
  const writes = [...HOOK.matchAll(/data-aui-following/g)].length;
  assert.equal(writes, 1, "only the setter may name the flag");
  assert.match(HOOK, /setFollowing\(true\);/);
  assert.ok(
    [...HOOK.matchAll(/setFollowing\(false\);/g)].length >= 2,
    "the idle frame and teardown both clear the flag",
  );
});

test("no thread rule reaches the viewport through a flagged sibling", () => {
  // `.x[attr] + .aui-thread-root ...` invalidates the whole thread in WebKit on every flip.
  const rules = CSS.replace(/\/\*[\s\S]*?\*\//g, "").match(/[^{}]+\{/g) ?? [];
  for (const rule of rules) {
    if (!/aui-thread-(root|viewport)/.test(rule)) continue;
    assert.doesNotMatch(
      rule,
      /\[[^\]]+\]\s*[+~]/,
      `${rule.trim()}: an attribute on a sibling restyles every row when it changes`,
    );
  }
});
