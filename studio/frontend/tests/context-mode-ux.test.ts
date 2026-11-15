// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import path from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const panel = readFileSync(
  path.join(
    here,
    "..",
    "src/features/model-picker/components/model-config-page.tsx",
  ),
  "utf8",
);
const panelCode = panel.replace(/\s+/g, " ");

test("the far-right slider position is Auto, a full bar", () => {
  assert.match(
    panelCode,
    /const contextIsAuto = config\.customContextLength == null;/,
  );
  assert.match(
    panelCode,
    /<GgufContextLengthSetting value=\{contextIsAuto \? 0 : contextValue\}/,
  );
  assert.match(panelCode, /<Slider min=\{min\} max=\{max\}/);
  // Auto (0) sits at max on the slider; within one step of max reads as Auto.
  assert.match(panelCode, /const toSlider = \(v: number\) => \(v === 0 \? max : v\);/);
  assert.match(panelCode, /const isAutoPosition = \(v: number\) => v > max - CONTEXT_SLIDER_STEP;/);
  assert.match(panelCode, /customContextLength: v === 0 \? null : v/);
  // The value box names Auto; the slider shares its row.
  assert.match(panelCode, /displayValue=\{isAuto \? "Auto" : undefined\}/);
});

test("moving left or typing a number creates a custom context", () => {
  assert.match(
    panelCode,
    /onChange=\{\(v\) => update\(\{ customContextLength: v === 0 \? null : v \}\)\}/,
  );
  assert.match(
    panelCode,
    /useSliderDraft\(toSlider\(value\), \(v\) => onChange\(fromSlider\(v\)\), \); const shown = draft == null \? value : fromSlider\(draft\); const isAuto = shown === 0;/,
  );
  assert.match(
    panelCode,
    /<NumericValueInput ref=\{inputRef\} value=\{isAuto \? autoValue : shown\} min=\{min\} max=\{max\} step=\{1\} onChange=\{onChange\} displayValue=\{isAuto \? "Auto" : undefined\}/,
  );
  assert.doesNotMatch(panelCode, /ariaLabel="Context length mode"/);
});

test("the Auto number editor opens at the fitted or safe offload context", () => {
  assert.match(
    panelCode,
    /activeLoadedContext \?\? AUTO_OFFLOAD_CONTEXT_LENGTH/,
  );
  assert.match(panelCode, /const AUTO_OFFLOAD_CONTEXT_LENGTH = 8192;/);
});

test("the existing info hint owns the mode explanation and fitted result", () => {
  assert.match(
    panelCode,
    /<InfoHint> The model&apos;s context window\. Drag all the way right for Auto,[\s\S]*Anything short of that requests an exact context window,[\s\S]*Auto currently selected \$\{activeLoadedContext\.toLocaleString\(\)\} tokens\.[\s\S]*<\/InfoHint>/,
  );
  assert.match(
    panelCode,
    /renderWarning=\{\(v\) => resolvedIsDiffusion \? \( <DiffusionContextNote [^)]*\) : isActiveModel && loadedMaxContextLength != null && v > loadedMaxContextLength && \(/,
  );
  assert.match(panelCode, /\{renderWarning\(shown\)\}/);
  assert.doesNotMatch(
    panelCode,
    /<p className="text-ui-11 leading-relaxed text-muted-foreground">/,
  );
});
