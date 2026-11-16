// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The Reduce motion `*` blankets in index.css must use transition-duration 0s: transition-property
// defaults to all, so a nonzero duration transitions every property of every element, and Chromium
// starts a no-op scrollbar-color transition on each `* { scrollbar-color }` element per restyle
// (2,550 per body-level restyle at 40 turns in Liquid Glass; thread scroll at 200 turns 0.6-1.2 s
// of main-thread CPU against 0.17-0.21 s). Only class-declared transitions and the overlays
// native-view.ts follows keep the 0.01ms that still fires transitionend.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const read = (rel: string) => readFileSync(path.join(ROOT, rel), "utf8");

const BLANKETS = ["html.force-reduced-motion *", "html:not(.force-motion) *"];

type Rule = { selectors: string[]; body: string };

function rules(css: string): Rule[] {
  const out: Rule[] = [];
  const flat = css.replace(/\/\*[\s\S]*?\*\//g, "");
  for (const m of flat.matchAll(/([^{}@;]+)\{([^{}]*)\}/g)) {
    const selectors = m[1].split(",").map((s) => s.trim().replace(/\s+/g, " ")).filter(Boolean);
    out.push({ selectors, body: m[2] });
  }
  return out;
}

function duration(body: string): string | null {
  const m = body.match(/transition-duration\s*:\s*([^;!]+)/);
  return m ? m[1].trim() : null;
}

/** Reduce motion `*` blankets that give every element a nonzero transition-duration. */
export function implicitTransitionBlankets(css: string): string[] {
  const bad: string[] = [];
  for (const rule of rules(css)) {
    const blanket = rule.selectors.find((s) => BLANKETS.includes(s));
    if (!blanket) continue;
    const value = duration(rule.body);
    if (value !== null && !/^0m?s$/.test(value)) bad.push(`${blanket} { transition-duration: ${value} }`);
  }
  return bad;
}

/** The selector lists of the rules that hand class-declared transitions back their 0.01ms. */
function exemptions(css: string): string[] {
  return rules(css)
    .filter((r) => duration(r.body) === "0.01ms" && r.selectors.join(",").includes('[class*="transition"]'))
    .map((r) => r.selectors.join(", "));
}

test("the checker flags a nonzero blanket duration and accepts 0s", () => {
  const bad = "html.force-reduced-motion *,\nhtml.force-reduced-motion *::before { transition-duration: 0.01ms !important; }";
  assert.deepEqual(implicitTransitionBlankets(bad), ["html.force-reduced-motion * { transition-duration: 0.01ms }"]);
  const media = "@media (prefers-reduced-motion: reduce) { html:not(.force-motion) * { transition-duration: 1ms !important; } }";
  assert.deepEqual(implicitTransitionBlankets(media), ["html:not(.force-motion) * { transition-duration: 1ms }"]);
  const good = "html.force-reduced-motion * { transition-duration: 0s !important; }\n.x * { transition-duration: 1s; }";
  assert.deepEqual(implicitTransitionBlankets(good), []);
});

test("neither Reduce motion blanket turns on transitions for every element", () => {
  const css = read("src/index.css");
  for (const blanket of BLANKETS) {
    assert.ok(rules(css).some((r) => r.selectors.includes(blanket)), `${blanket} block not found`);
  }
  assert.deepEqual(implicitTransitionBlankets(css), []);
});

test("both blocks keep 0.01ms for class-declared transitions and every overlay native-view.ts follows", () => {
  const css = read("src/index.css");
  const found = exemptions(css);
  assert.equal(found.length, 2, `one exemption per Reduce motion block, found: ${JSON.stringify(found)}`);
  assert.ok(found.some((s) => s.startsWith("html.force-reduced-motion ")));
  assert.ok(found.some((s) => s.startsWith("html:not(.force-motion) ")));
  // native-view.ts measures these overlays while their transitions run; at 0s none would fire.
  const nativeView = read("src/features/browser/native-view.ts");
  const clickable = nativeView.match(/const CLICKABLE_SELECTOR =\s*"([^"]+)"/);
  assert.ok(clickable, "CLICKABLE_SELECTOR not found in native-view.ts");
  for (const part of clickable[1].split(",").map((s) => s.trim())) {
    for (const s of found) assert.ok(s.includes(part), `${part} missing from: ${s}`);
  }
});
