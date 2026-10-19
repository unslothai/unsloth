// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  ORB_WIDGET_HEIGHT,
  ORB_WIDGET_WIDTH,
  isOrbWidgetAction,
  parseSavedPosition,
  placeOrbWidget,
} from "../src/features/chat/voice/orb-widget.ts";

const laptop = { x: 0, y: 0, width: 1512, height: 982 };
const external = { x: 1512, y: -200, width: 2560, height: 1440 };

test("a saved position round-trips and junk falls back to nothing", () => {
  assert.deepEqual(parseSavedPosition('{"x":100.4,"y":-20.6}'), { x: 100, y: -21 });
  assert.equal(parseSavedPosition(null), null);
  assert.equal(parseSavedPosition("not json"), null);
  assert.equal(parseSavedPosition('{"x":"1","y":2}'), null);
  assert.equal(parseSavedPosition('{"x":null,"y":2}'), null);
});

test("the widget reopens where it was left while that spot is on a monitor", () => {
  assert.deepEqual(placeOrbWidget({ x: 2000, y: 100 }, [laptop, external], laptop), { x: 2000, y: 100 });
  // Mostly off the right edge of the only monitor still counts while its centre is on it.
  const nearEdge = { x: laptop.width - ORB_WIDGET_WIDTH / 2 - 1, y: 10 };
  assert.deepEqual(placeOrbWidget(nearEdge, [laptop], laptop), nearEdge);
});

test("a spot on an unplugged monitor falls back to the bottom right corner", () => {
  const placed = placeOrbWidget({ x: 2000, y: 100 }, [laptop], laptop);
  assert.deepEqual(placed, {
    x: laptop.width - ORB_WIDGET_WIDTH - 24,
    y: laptop.height - ORB_WIDGET_HEIGHT - 72,
  });
  assert.deepEqual(placeOrbWidget(null, [laptop, external], external), {
    x: external.x + external.width - ORB_WIDGET_WIDTH - 24,
    y: external.y + external.height - ORB_WIDGET_HEIGHT - 72,
  });
});

test("no monitor information leaves placement to the window manager", () => {
  assert.equal(placeOrbWidget(null, [], null), null);
  assert.equal(placeOrbWidget({ x: 5, y: 5 }, [], null), null);
});

test("only the control bar's actions are accepted from the widget", () => {
  for (const action of ["toggle-pause", "stop", "back"]) assert.ok(isOrbWidgetAction(action));
  for (const action of ["quit", "", null, 1, { action: "back" }]) assert.ok(!isOrbWidgetAction(action));
});
