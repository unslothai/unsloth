// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  ORB_WIDGET_BALL,
  ORB_WIDGET_CENTER,
  ORB_WIDGET_HEIGHT,
  ORB_WIDGET_WIDTH,
  clampToScreens,
  isOrbWidgetAction,
  packSpectrum,
  parseSavedPosition,
  placeOrbWidget,
  unpackSpectrum,
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

test("the ball, its glow at the speaking peak and the bar fit inside the window", () => {
  // voice-orb.tsx at 140 px: a 70 px disc with a 32 px blur and 8 px spread, pulsing to 1.1x.
  const glow = (70 + 8 + 32) * (ORB_WIDGET_BALL / 140) * 1.1;
  assert.equal(ORB_WIDGET_CENTER.x, ORB_WIDGET_WIDTH / 2);
  assert.ok(ORB_WIDGET_CENTER.x - glow >= 0 && ORB_WIDGET_CENTER.x + glow <= ORB_WIDGET_WIDTH);
  assert.ok(ORB_WIDGET_CENTER.y - glow >= 0 && ORB_WIDGET_CENTER.y + glow <= ORB_WIDGET_HEIGHT);
  // The bar: top 26 px below the centre, three 28 px buttons plus padding, ~30 px tall.
  assert.ok(ORB_WIDGET_CENTER.y + 26 < ORB_WIDGET_CENTER.y + ORB_WIDGET_BALL / 2, "bar overlaps the ball");
  assert.ok(ORB_WIDGET_CENTER.y + 26 + 30 <= ORB_WIDGET_HEIGHT);
  assert.ok(3 * 28 + 16 <= ORB_WIDGET_WIDTH);
});

test("a drag may reach the very top of the screen, menu bar included", () => {
  const top = { x: 600, y: -ORB_WIDGET_CENTER.y };
  assert.deepEqual(clampToScreens(top, [laptop]), top);
  assert.deepEqual(clampToScreens({ x: 600, y: 0 }, [laptop]), { x: 600, y: 0 });
});

test("a drag past an edge keeps the ball's centre on the screen", () => {
  assert.deepEqual(clampToScreens({ x: 600, y: -500 }, [laptop]), { x: 600, y: -ORB_WIDGET_CENTER.y });
  assert.deepEqual(clampToScreens({ x: -900, y: 300 }, [laptop]), { x: -ORB_WIDGET_CENTER.x, y: 300 });
  const farRight = clampToScreens({ x: 9000, y: 9000 }, [laptop, external]);
  assert.equal(farRight.x + ORB_WIDGET_CENTER.x, external.x + external.width - 1);
  assert.equal(farRight.y + ORB_WIDGET_CENTER.y, external.y + external.height - 1);
  // Crossing onto the second screen is free.
  assert.deepEqual(clampToScreens({ x: 2000, y: -150 }, [laptop, external]), { x: 2000, y: -150 });
  assert.deepEqual(clampToScreens({ x: 5, y: 5 }, []), { x: 5, y: 5 });
});

test("spectrum frames round-trip and junk reads as silence", () => {
  const bands = Float32Array.from({ length: 32 }, (_, i) => i / 31);
  const frame = packSpectrum("output", 0.123456, bands);
  assert.equal(frame.bands.length, 32);
  const out = new Float32Array(32);
  assert.equal(unpackSpectrum(JSON.parse(JSON.stringify(frame)), out), 0.123);
  for (let i = 0; i < 32; i++) assert.ok(Math.abs(out[i] - bands[i]) <= 0.005);
  out.fill(1);
  assert.equal(unpackSpectrum({ side: "input", level: 0.5, bands: [1, 2] }, out), 0);
  assert.ok(out.every((v) => v === 0));
  assert.equal(unpackSpectrum(null, out), 0);
  assert.equal(unpackSpectrum({ level: 3, bands: new Array(32).fill(7) }, out), 1);
  assert.ok(out.every((v) => v === 1));
});
