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
  assert.match(panelCode, /const toSlider = \(v: number\) => \(v === 0 \? max : v\);/);
  assert.match(panelCode, /const isMaxPosition = \(v: number\) => v > max - CONTEXT_SLIDER_STEP;/);
  assert.match(panelCode, /customContextLength: v === 0 \? null : v/);
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
    /autoLoadedContext \?\? AUTO_OFFLOAD_CONTEXT_LENGTH/,
  );
  assert.match(panelCode, /const AUTO_OFFLOAD_CONTEXT_LENGTH = 8192;/);
});

test("the existing info hint owns the mode explanation and fitted result", () => {
  assert.match(
    panelCode,
    /<InfoHint> The model&apos;s context window\. Auto \(a full bar\) picks one that fits[\s\S]*the reset button returns to it\. A number requests an exact context window,[\s\S]*Auto currently selected \$\{activeLoadedContext\.toLocaleString\(\)\} tokens\.[\s\S]*<\/InfoHint>/,
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

test("once a number is set, the far right is the real maximum and only reset returns to Auto", () => {
  assert.match(
    panelCode,
    /const fromSlider = \(v: number\) => isMaxPosition\(v\) && !\(keyed\.current && v < max\) \? value === 0 \? 0 : max : v;/,
  );
  assert.match(
    panelCode,
    /onKeyDownCapture=\{\(\) => \{ keyed\.current = true; \}\} onPointerDownCapture=\{\(\) => \{ keyed\.current = false; \}\}/,
  );
  assert.match(
    panelCode,
    /const reset = isAuto \? null : \( <button type="button" onClick=\{\(\) => onChange\(0\)\}[^>]*aria-label="Reset context to Auto"/,
  );
});

test("the sidebar puts the context reset beside the number; the picker keeps it by the label", () => {
  assert.match(panelCode, /<\/InfoHint> \{stacked \? null : reset\} <\/span>/);
  assert.match(panelCode, /\{stacked \? reset : null\} <NumericValueInput/);
  assert.match(panelCode, /stacked=\{variant === "sidebar"\}/);
});

test("the context slider's left gap counts from the label text, with a tighter right gap", () => {
  assert.match(
    panelCode,
    /const label = \( \/\/ .*? <div className="relative flex shrink-0 items-center"> <span className=\{LABEL_CLASS\}>Context<\/span> <span className="absolute left-full ml-1\.5 flex items-center gap-1">/,
  );
  assert.match(
    panelCode,
    /<div className="config-row flex min-h-8 items-center"> \{label\} \{slider\} \{input\} <\/div>/,
  );
  assert.match(
    panelCode,
    /className=\{stacked \? undefined : "ml-12 mr-7 min-w-0 flex-1"\} aria-label="Context"/,
  );
});

test("the Run settings sidebar puts the context slider on its own row", () => {
  assert.match(panelCode, /stacked=\{variant === "sidebar"\}/);
  assert.match(
    panelCode,
    /<div className="space-y-2"> <div className="config-row flex min-h-8 items-center"> \{label\} \{input\} <\/div> \{slider\} <\/div>/,
  );
});

test("Reset all reads in the same color as Save", () => {
  assert.doesNotMatch(
    panelCode,
    /className="text-muted-foreground" disabled=\{ \(pinnedSet !== null/,
  );
});
