// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Radix puts role=slider on the Thumb, not Root, and never synthesises aria-valuetext,
// so both must be forwarded explicitly. Asserted on source: node:test cannot render.

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
  assert.match(panel, /!\(isMaxPosition\(v\) && value === 0\)/);
  assert.match(panel, /: "Auto"/);
});

test("Auto announces a current value only once one exists", () => {
  // Pre-load autoValue is a fallback seed, not a selection; announcing it misleads users.
  assert.match(
    panel,
    /activeLoadedContext != null \? `Auto, currently \$\{autoValue\.toLocaleString\(\)\} tokens` : "Auto"/,
  );
});

test("no slider is left announcing a bare number for a named position", () => {
  assert.ok(
    slider.includes("aria-valuetext"),
    "the shared Slider must keep forwarding aria-valuetext to the thumb",
  );
});
