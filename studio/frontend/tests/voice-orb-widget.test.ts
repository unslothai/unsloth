// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  ORB_WIDGET_BALL,
  ORB_WIDGET_BAR_HEIGHT,
  ORB_WIDGET_BAR_TOP,
  ORB_WIDGET_DRAG_THRESHOLD,
  ORB_WIDGET_CENTER,
  ORB_WIDGET_EDGE_REACH,
  ORB_WIDGET_GLOW_REACH,
  ORB_WIDGET_HEIGHT,
  ORB_WIDGET_WIDTH,
  clampToScreens,
  dragTarget,
  isOrbWidgetAction,
  packSpectrum,
  parseSavedPosition,
  placeAtCursor,
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

test("the ball, its widest wave, its glow and the bar fit inside the window", () => {
  // voice-orb.tsx at 140 px: the edge reaches 70 + 3 + 16; the glow's visible tail
  // 70 + 8 + 1.5 * 32; the speaking pulse scales both by 1.1. Measured in a browser at
  // 72 px: edge 50.3, glow 71.3 from the centre.
  const k = ORB_WIDGET_BALL / 140;
  const wave = (70 + 3 + 16) * k * 1.1;
  const glow = (70 + 8 + 48) * k * 1.1;
  assert.ok(Math.abs(ORB_WIDGET_EDGE_REACH - wave) < 1e-9 && Math.abs(ORB_WIDGET_GLOW_REACH - glow) < 1e-9);
  const reach = Math.max(wave, glow);
  const { x, y } = ORB_WIDGET_CENTER;
  assert.equal(x, ORB_WIDGET_WIDTH / 2);
  for (const room of [x - reach, ORB_WIDGET_WIDTH - x - reach, y - reach, ORB_WIDGET_HEIGHT - y - reach]) {
    assert.ok(room >= 24, `only ${room} px of margin`);
  }
  // The bar sits below the ball's rim (half its height lower than flush), so only the
  // pulse and the wave, drawn in front, cross it; it still ends inside the window.
  assert.ok(ORB_WIDGET_BAR_TOP > y + ORB_WIDGET_BALL / 2, "bar clears the ball at rest");
  assert.ok(ORB_WIDGET_BAR_TOP < y + wave, "the wave can cross the bar");
  assert.ok(ORB_WIDGET_BAR_TOP + ORB_WIDGET_BAR_HEIGHT + 8 <= ORB_WIDGET_HEIGHT);
  assert.ok(3 * 28 + 16 <= ORB_WIDGET_WIDTH);
  assert.deepEqual([ORB_WIDGET_WIDTH, ORB_WIDGET_HEIGHT], [192, 192]);
});

test("the widget opens centred on the cursor, kept on the cursor's screen", () => {
  const c = ORB_WIDGET_CENTER;
  assert.deepEqual(placeAtCursor({ x: 700, y: 400 }, [laptop, external]), { x: 700 - c.x, y: 400 - c.y });
  // On the second, taller screen, above the first one's top.
  assert.deepEqual(placeAtCursor({ x: 2000, y: -150 }, [laptop, external]), { x: 2000 - c.x, y: -150 - c.y });
  // In a corner, the whole ball stays on that screen.
  const r = ORB_WIDGET_BALL / 2;
  assert.deepEqual(placeAtCursor({ x: 2, y: 1 }, [laptop, external]), { x: r - c.x, y: r - c.y });
  assert.deepEqual(placeAtCursor({ x: 1511, y: 981 }, [laptop, external]), {
    x: laptop.width - r - c.x,
    y: laptop.height - r - c.y,
  });
  assert.equal(placeAtCursor({ x: -50, y: 10 }, [laptop]), null);
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

test("a click or a jitter inside the slop never moves the window", () => {
  const start = { screen: { x: 500, y: 300 }, grab: { x: 70, y: 60 } };
  assert.equal(dragTarget(start, { x: 500, y: 300 }, false), null);
  assert.equal(dragTarget(start, { x: 503, y: 302 }, false), null);
  assert.equal(dragTarget(start, { x: 500 + ORB_WIDGET_DRAG_THRESHOLD, y: 300 }, false), null);
});

test("a drag keeps the grabbed point under the pointer, with no jump at the start", () => {
  const start = { screen: { x: 500, y: 300 }, grab: { x: 70, y: 60 } };
  // The window's top left at press is screen - grab, so the first move lands next to it.
  assert.deepEqual(dragTarget(start, { x: 506, y: 300 }, false), { x: 436, y: 240 });
  // Once moving, small steps follow the pointer exactly, back to the start included.
  assert.deepEqual(dragTarget(start, { x: 501, y: 301 }, true), { x: 431, y: 241 });
  assert.deepEqual(dragTarget(start, { x: 500, y: 300 }, true), { x: 430, y: 240 });
  // Fractional points (a 2x screen reports halves) round to whole logical pixels.
  assert.deepEqual(dragTarget(start, { x: 600.5, y: 10.5 }, true), { x: 531, y: -49 });
});

test("past any edge, on any screen, at least half the ball stays visible", () => {
  const r = ORB_WIDGET_BALL / 2;
  const screens = [laptop, external];
  const visibleWidth = (p: { x: number; y: number }) => {
    // The ball's horizontal and vertical spans clipped to whichever screen holds its centre.
    const cx = p.x + ORB_WIDGET_CENTER.x;
    const cy = p.y + ORB_WIDGET_CENTER.y;
    const m = screens.find((s) => cx >= s.x && cx < s.x + s.width && cy >= s.y && cy < s.y + s.height);
    assert.ok(m, "centre on a screen");
    const w = Math.min(cx + r, m.x + m.width) - Math.max(cx - r, m.x);
    const h = Math.min(cy + r, m.y + m.height) - Math.max(cy - r, m.y);
    return Math.min(w, h);
  };
  for (const far of [
    { x: -5000, y: 300 },
    { x: 400, y: -5000 },
    { x: 400, y: 5000 },
    { x: 9000, y: 300 },
    { x: -5000, y: -5000 },
    { x: 1700, y: -5000 },
  ]) {
    assert.ok(visibleWidth(clampToScreens(far, screens)) >= r - 1, JSON.stringify(far));
  }
});
