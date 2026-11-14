// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

test("the allowance switch saves the visible draft and can always turn it off", () => {
  const SECTION = readSrc("features/settings/components/token-limit-section.tsx");
  assert.match(SECTION, /onCheckedChange=\{\(next\) => void save\(\{ enabled: next \}\)\}/);
  assert.match(SECTION, /disabled=\{busy \|\| \(!enabled && cap === null\)\}/);
  assert.match(SECTION, /\{\s*period,\s*\.\.\.\(cap === null \? \{\} : \{ limit: cap \}\)/);
});
