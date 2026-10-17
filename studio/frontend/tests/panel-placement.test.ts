// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The API monitor panel and Live resource monitor both default to bottom right.

import assert from "node:assert/strict";
import test from "node:test";

import {
  clampPanelToViewport,
  isFullyCovered,
  PANEL_GAP,
  PANEL_MARGIN,
  PANEL_TOP_MARGIN,
  type PanelRect,
  type PanelSize,
  placeFloatingPanel,
} from "../src/features/api-monitor/panel-placement.ts";

const W = 1440;
const H = 900;
const VIEWPORT = { width: W, height: H };
const PANEL: PanelSize = { width: 400, height: 300 };

function monitorCorner(height = 300): PanelRect {
  return {
    left: W - PANEL_MARGIN - 256,
    top: H - PANEL_MARGIN - height,
    right: W - PANEL_MARGIN,
    bottom: H - PANEL_MARGIN,
  };
}

function overlaps(
  anchor: { left: number; top: number },
  size: PanelSize,
  box: PanelRect,
): boolean {
  return (
    anchor.left < box.right &&
    anchor.left + size.width > box.left &&
    anchor.top < box.bottom &&
    anchor.top + size.height > box.top
  );
}

test("with nothing in the way the panel keeps the corner it shipped in", () => {
  const anchor = placeFloatingPanel(PANEL, [], VIEWPORT);
  assert.deepEqual(anchor, {
    left: W - PANEL_MARGIN - PANEL.width,
    top: H - PANEL_MARGIN - PANEL.height,
  });
});

test("a monitor in the shared corner is stepped over, not sat on", () => {
  const monitor = monitorCorner();
  const anchor = placeFloatingPanel(PANEL, [monitor], VIEWPORT);
  assert.ok(
    !overlaps(anchor, PANEL, monitor),
    `the panel still covers the monitor: ${JSON.stringify(anchor)}`,
  );
  // Same column, directly above it, which is the move the notification stack
  // makes: staying put horizontally is the smallest surprise.
  assert.equal(anchor.left, W - PANEL_MARGIN - PANEL.width);
  assert.equal(anchor.top + PANEL.height + PANEL_GAP, monitor.top);
});

test("a monitor dragged out of the corner leaves the panel where it was", () => {
  const monitor: PanelRect = {
    left: PANEL_MARGIN,
    top: PANEL_MARGIN,
    right: PANEL_MARGIN + 256,
    bottom: PANEL_MARGIN + 300,
  };
  assert.deepEqual(placeFloatingPanel(PANEL, [monitor], VIEWPORT), {
    left: W - PANEL_MARGIN - PANEL.width,
    top: H - PANEL_MARGIN - PANEL.height,
  });
});

test("a monitor too tall to step over sends the panel to the other side", () => {
  const monitor: PanelRect = {
    left: W - PANEL_MARGIN - 500,
    top: PANEL_MARGIN,
    right: W - PANEL_MARGIN,
    bottom: H - PANEL_MARGIN,
  };
  const anchor = placeFloatingPanel(PANEL, [monitor], VIEWPORT);
  assert.ok(!overlaps(anchor, PANEL, monitor), "the panel must clear it");
  assert.equal(anchor.left, PANEL_MARGIN);
});

test("every published box is dodged, not just the first", () => {
  const monitor = monitorCorner(200);
  const composer: PanelRect = {
    left: 200,
    top: H - 260,
    right: W - 200,
    bottom: H - 24,
  };
  const anchor = placeFloatingPanel(PANEL, [monitor, composer], VIEWPORT);
  assert.ok(!overlaps(anchor, PANEL, monitor), "monitor still covered");
  assert.ok(!overlaps(anchor, PANEL, composer), "composer still covered");
});

// Fallback order differs from preference: the monitor's close and resize grip on its right
// edge must stay clickable.
test("a monitor filling the viewport pushes the panel off the right edge", () => {
  const monitor: PanelRect = {
    left: PANEL_MARGIN,
    top: PANEL_MARGIN,
    right: W - PANEL_MARGIN,
    bottom: H - PANEL_MARGIN,
  };
  const anchor = placeFloatingPanel(PANEL, [monitor], VIEWPORT);
  assert.equal(anchor.left, PANEL_MARGIN, "must not stay in the right column");
  assert.equal(anchor.top, H - PANEL_MARGIN - PANEL.height);
});

// Narrow window: the right anchor's left coord falls left of midpoint, so side is not
// derived from coordinates.
test("a narrow window still sends the panel off the right edge", () => {
  for (const width of [768, 800, 832]) {
    const viewport = { width, height: 700 };
    const monitor: PanelRect = {
      left: PANEL_MARGIN,
      top: PANEL_MARGIN,
      right: width - PANEL_MARGIN,
      bottom: 700 - PANEL_MARGIN,
    };
    const anchor = placeFloatingPanel(PANEL, [monitor], viewport);
    assert.equal(
      anchor.left,
      PANEL_MARGIN,
      `at ${width}px the panel stayed in the right column, over Close and the grip`,
    );
  }
});

test("the panel always lands fully on screen", () => {
  const monitor = monitorCorner(600);
  for (const viewport of [
    { width: 1440, height: 900 },
    { width: 420, height: 700 },
  ]) {
    const size = {
      width: Math.min(PANEL.width, viewport.width - 2 * PANEL_MARGIN),
      height: PANEL.height,
    };
    const anchor = placeFloatingPanel(size, [monitor], viewport);
    assert.ok(anchor.left >= PANEL_MARGIN, `left off screen: ${anchor.left}`);
    assert.ok(anchor.top >= PANEL_MARGIN, `top off screen: ${anchor.top}`);
    assert.ok(
      anchor.left + size.width <= viewport.width - PANEL_MARGIN,
      `right off screen: ${anchor.left + size.width}`,
    );
    assert.ok(
      anchor.top + size.height <= viewport.height - PANEL_MARGIN,
      `bottom off screen: ${anchor.top + size.height}`,
    );
  }
});

// The header must survive: it holds the drag handle and Close.
test("a viewport too small for the panel keeps its header on screen", () => {
  const anchor = placeFloatingPanel(PANEL, [], { width: 320, height: 200 });
  assert.deepEqual(anchor, { left: PANEL_MARGIN, top: PANEL_TOP_MARGIN });
});

// The top chrome publishes no box and sits under this layer.
test("stepping over a tall box never reaches the top chrome", () => {
  const tall: PanelRect = {
    left: 400,
    top: 100,
    right: W - PANEL_MARGIN,
    bottom: H - PANEL_MARGIN,
  };
  const anchor = placeFloatingPanel(PANEL, [tall], VIEWPORT);
  assert.ok(
    anchor.top >= PANEL_TOP_MARGIN,
    `the panel climbed into the top chrome: ${anchor.top}`,
  );
});

test("nothing published leaves the panel uncovered", () => {
  const anchor = { left: 100, top: 100 };
  assert.equal(isFullyCovered(anchor, PANEL, []), false);
});

test("a box that swallows the panel whole is reported", () => {
  const anchor = { left: 100, top: 100 };
  const whole: PanelRect = { left: 0, top: 0, right: W, bottom: H };
  assert.equal(isFullyCovered(anchor, PANEL, [whole]), true);
});

// A visible sliver is clickable; one case per edge since dropping one edge passes the rest.
test("a box that leaves any edge of the panel showing is not reported", () => {
  const anchor = { left: 100, top: 100 };
  const whole: PanelRect = {
    left: 0,
    top: 0,
    right: 100 + PANEL.width,
    bottom: 100 + PANEL.height,
  };
  const shy = {
    left: { ...whole, left: anchor.left + 1 },
    top: { ...whole, top: anchor.top + 1 },
    right: { ...whole, right: whole.right - 1 },
    bottom: { ...whole, bottom: whole.bottom - 1 },
  };
  assert.equal(isFullyCovered(anchor, PANEL, [whole]), true);
  for (const [edge, box] of Object.entries(shy)) {
    assert.equal(
      isFullyCovered(anchor, PANEL, [box]),
      false,
      `a strip showing on the ${edge} still counted as covered`,
    );
  }
});

// The panel keeps no position across reloads, so one stranded off-screen is unreachable.
test("a hand-placed panel is pulled back onto a shrunken viewport", () => {
  const landed = clampPanelToViewport({ left: 1200, top: 700 }, PANEL, {
    width: 800,
    height: 600,
  });
  assert.deepEqual(landed, {
    left: 800 - PANEL_MARGIN - PANEL.width,
    top: 600 - PANEL_MARGIN - PANEL.height,
  });
});

test("a hand-placed panel inside the viewport is left alone", () => {
  const placed = { left: 300, top: 300 };
  assert.deepEqual(clampPanelToViewport(placed, PANEL, VIEWPORT), placed);
});
