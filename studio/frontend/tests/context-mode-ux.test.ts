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

test("the far-left slider position is Auto", () => {
  assert.match(
    panelCode,
    /const contextIsAuto = config\.customContextLength == null;/,
  );
  assert.match(
    panelCode,
    /const contextSliderValue = contextIsAuto \? 0 : contextValue/,
  );
  assert.match(panelCode, /<Slider min=\{0\}/);
  assert.match(panelCode, /customContextLength: v === 0 \? null : v/);
  assert.match(panelCode, /<span>Auto<\/span>/);
});

test("moving right or typing a number creates a custom context", () => {
  assert.match(
    panelCode,
    /onValueChange=\{\(\[v\]\) => setContextSliderValue\(v\)\}/,
  );
  assert.match(
    panelCode,
    /const setContextLength = \(v: number\) => update\(\{ customContextLength: v \}\)/,
  );
  assert.match(
    panelCode,
    /displayValue=\{contextIsAuto \? "Auto" : undefined\}/,
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
    /<InfoHint> Drag all the way left for Auto,[\s\S]*Custom values request an exact context;[\s\S]*Auto currently selected \$\{activeLoadedContext\.toLocaleString\(\)\} tokens\.[\s\S]*<\/InfoHint>/,
  );
  assert.match(panelCode, /\{!contextIsAuto && isActiveModel/);
  assert.doesNotMatch(
    panelCode,
    /<p className="text-ui-11 leading-relaxed text-muted-foreground">/,
  );
});
