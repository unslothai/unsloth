// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  GLASS_ELEMENT_OUTLINE,
  outlineStrength,
} from "../src/features/settings/lib/element-outline-default.ts";
import { readSrc } from "./helpers/kit.ts";

const hex = (value: string): [number, number, number, number] => [
  Number.parseInt(value.slice(1, 3), 16),
  Number.parseInt(value.slice(3, 5), 16),
  Number.parseInt(value.slice(5, 7), 16),
  1,
];

test("theme borders map onto the slider's foreground mix", () => {
  // Standard light #e5e5e5 on #fefefd, dark #3a3a3a on #181818.
  assert.equal(
    outlineStrength(hex("#e5e5e5"), hex("#fefefd"), hex("#0f1114")),
    10,
  );
  assert.equal(
    outlineStrength(hex("#3a3a3a"), hex("#181818"), hex("#fcfcfc")),
    15,
  );
  // Blueberry light: a tinted border still lands on its strength.
  assert.equal(
    outlineStrength(hex("#abb0c5"), hex("#f6f7fc"), hex("#1c2140")),
    33,
  );
  // A translucent border is composited over the background first.
  assert.equal(
    outlineStrength([15, 17, 20, 0.09], hex("#fefefd"), hex("#0f1114")),
    9,
  );
  assert.equal(
    outlineStrength(hex("#fefefd"), hex("#fefefd"), hex("#fefefd")),
    0,
  );
});

test("the glass defaults match the ring fallbacks in index.css", () => {
  const css = readSrc("index.css");
  assert.match(
    css,
    new RegExp(`var\\(--element-outline, ${GLASS_ELEMENT_OUTLINE.light}%\\)`),
  );
  assert.match(
    css,
    new RegExp(`var\\(--element-outline, ${GLASS_ELEMENT_OUTLINE.dark}%\\)`),
  );
});
