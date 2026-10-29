// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  placementPosition,
  stopPlacement,
  togglePlacement,
  usePlacement,
} from "../src/features/flows/editor/placement.ts";

const armed = () => usePlacement.getState().armed;

test("clicking an item arms it, clicking it again disarms, another item switches", () => {
  stopPlacement();
  togglePlacement("text.input");
  assert.equal(armed(), "text.input");
  togglePlacement("llm.generate");
  assert.equal(armed(), "llm.generate");
  togglePlacement("llm.generate");
  assert.equal(armed(), null);
});

test("Esc, right-click and leaving the editor all stop placement", () => {
  togglePlacement("text.input");
  stopPlacement();
  assert.equal(armed(), null);
  stopPlacement();
  assert.equal(armed(), null);
  const src = readFileSync(
    new URL("../src/features/flows/editor/canvas-controls.tsx", import.meta.url),
    "utf8",
  );
  assert.match(src, /useEffect\(\(\) => stopPlacement, \[\]\);/);
  assert.match(src, /e\.key === "Escape"\) stopPlacement\(\)/);
  assert.match(src, /const menu = [\s\S]*?stopPlacement\(\);/);
});

test("a placed node lands with its header under the pointer, on the grid when snap is on", () => {
  assert.deepEqual(placementPosition({ x: 500, y: 300 }, false), [370, 280]);
  assert.deepEqual(placementPosition({ x: 507.4, y: 301 }, true), [380, 280]);
  const [x, y] = placementPosition({ x: -45, y: 13 }, true);
  assert.equal(Math.abs(x % 20), 0);
  assert.equal(Math.abs(y % 20), 0);
});

test("only empty canvas places; clicks on nodes and groups do not", () => {
  const src = readFileSync(
    new URL("../src/features/flows/editor/canvas-controls.tsx", import.meta.url),
    "utf8",
  );
  const click = src.slice(src.indexOf("const click ="), src.indexOf("const dblclick"));
  assert.match(click, /!onPane\(e\)\) return;/);
  assert.match(src, /classList\.contains\("react-flow__pane"\)/);
});
