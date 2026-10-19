// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { placeTitleTooltip } from "../src/lib/title-tooltip-placement.ts";

const viewport = { width: 1000, height: 800 };
const size = { width: 100, height: 30 };

test("a title tooltip sits centred below its element", () => {
  assert.deepEqual(placeTitleTooltip({ left: 400, top: 100, bottom: 120, width: 40 }, size, viewport), { left: 370, top: 126 });
});

test("near the bottom edge it flips above", () => {
  assert.deepEqual(placeTitleTooltip({ left: 400, top: 760, bottom: 780, width: 40 }, size, viewport), { left: 370, top: 724 });
});

test("it stays inside the left and right edges", () => {
  assert.equal(placeTitleTooltip({ left: 0, top: 100, bottom: 120, width: 20 }, size, viewport).left, 8);
  assert.equal(placeTitleTooltip({ left: 980, top: 100, bottom: 120, width: 20 }, size, viewport).left, 892);
});
