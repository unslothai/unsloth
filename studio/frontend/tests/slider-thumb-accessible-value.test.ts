// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// What the slider says to a screen reader. Two gaps a sighted reviewer cannot see:
//
// 1. Radix puts role="slider" on the Thumb while the wrapper spreads props onto
//    Root, so an aria-label reached a plain div and the control was unnamed. Radix
//    only synthesises its own label for multi-thumb ranges.
// 2. Radix does not synthesise aria-valuetext, so the Auto position announced as
//    "0" -- the one stop on that track whose number is not a context length.
//
// Asserted on source: the failure is a missing attribute on an element node:test
// cannot render.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import path from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const read = (relative: string) =>
  readFileSync(path.join(here, "..", relative), "utf8").replace(/\s+/g, " ");

const slider = read("src/components/ui/slider.tsx");
const panel = read(
  "src/features/model-picker/components/model-config-page.tsx",
);

test("the thumb carries the accessible name, not just the root", () => {
  // Both spellings: a caller may name the slider either way.
  assert.match(
    slider,
    /<SliderPrimitive\.Thumb[^>]*aria-label=\{props\["aria-label"\]\}/,
  );
  assert.match(
    slider,
    /<SliderPrimitive\.Thumb[^>]*aria-labelledby=\{props\["aria-labelledby"\]\}/,
  );
});

test("the wrapper can express a spoken value for a position", () => {
  assert.match(
    slider,
    /thumbValueText\?:\s*\(value: number, index: number\) => string/,
  );
  assert.match(slider, /aria-valuetext=\{thumbValueText\?\.\(/);
});

test("the context slider says Auto rather than zero", () => {
  // aria-valuenow carries the raw position, so Auto needs saying in words.
  assert.match(panel, /thumbValueText=\{\(v\) =>/);
  assert.match(panel, /v !== 0 \? `\$\{v\.toLocaleString\(\)\} tokens`/);
  assert.match(panel, /: "Auto"/);
});

test("Auto announces a current value only once one exists", () => {
  // Before a load contextInputValue is the offload fallback that seeds the input,
  // not a selection: Auto may still fit the model's native context. Announcing it
  // as "currently N" tells a screen-reader user a number no other user is shown.
  assert.match(
    panel,
    /activeLoadedContext != null \? `Auto, currently \$\{contextInputValue\.toLocaleString\(\)\} tokens` : "Auto"/,
  );
});

test("no slider is left announcing a bare number for a named position", () => {
  // If the wrapper stops forwarding value text, every such caller regresses at once.
  assert.ok(
    slider.includes("aria-valuetext"),
    "the shared Slider must keep forwarding aria-valuetext to the thumb",
  );
});
