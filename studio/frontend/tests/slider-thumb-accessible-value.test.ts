// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// What a slider says to a screen reader, which is not what it shows on screen.
//
// Two separate gaps, both invisible to a sighted reviewer:
//
// 1. Radix puts role="slider" on the Thumb, but the wrapper spreads its props
//    onto Root. An aria-label passed to <Slider> therefore landed on a plain div
//    and the actual control had no accessible name at all, since Radix only
//    synthesises a label of its own for multi-thumb ranges.
//
// 2. Radix does not synthesise aria-valuetext. A slider whose positions mean
//    something other than their number announces the number and nothing else.
//    The context slider's leftmost position means Auto, so without value text it
//    reads as "0" -- a context length no model has, and the one position on that
//    track whose number is not a context at all.
//
// Source-level assertions, because the failure is a missing attribute on an
// element node:test cannot render.

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
  // The number Auto landed on is the point of the setting, so it has to be
  // spoken too; aria-valuenow can only carry the raw slider position.
  assert.match(panel, /thumbValueText=\{\(v\) =>/);
  assert.match(
    panel,
    /v === 0 \? `Auto, currently \$\{contextInputValue\.toLocaleString\(\)\} tokens`/,
  );
  assert.match(panel, /: `\$\{v\.toLocaleString\(\)\} tokens`/);
});

test("no slider is left announcing a bare number for a named position", () => {
  // A guard against the next slider that gives a position a meaning: if the
  // wrapper ever stops forwarding value text, every such caller regresses at
  // once and silently.
  assert.ok(
    slider.includes("aria-valuetext"),
    "the shared Slider must keep forwarding aria-valuetext to the thumb",
  );
});
