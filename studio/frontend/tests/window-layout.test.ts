// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  DEFAULT_APP_WINDOW_SIZE_BOUNDS,
  PREFERRED_SETUP_WINDOW_SIZE,
  calculateCenteredPosition,
  calculateFirstAppWindowSize,
  calculateWindowSizeBounds,
  constrainWindowSize,
  fitWindowSize,
} from "../src/app/window-layout.ts";
import {
  finalizeAppWindowLayout,
  measureWindowLayout,
} from "../src/app/window-layout-lifecycle.ts";

// A work area is the panel minus the taskbar, in logical pixels.
function workArea(
  width: number,
  height: number,
  taskbar: number,
  scaleFactor: number,
) {
  return {
    width: width / scaleFactor,
    height: (height - taskbar) / scaleFactor,
  };
}

test("keeps the nominal minimum and preferred size on a roomy work area", () => {
  const bounds = calculateWindowSizeBounds({ width: 1920, height: 1040 });

  assert.deepEqual(bounds.minimum, { width: 900, height: 600 });
  assert.deepEqual(calculateFirstAppWindowSize(bounds), {
    width: 1440,
    height: 884,
  });
});

test("fits a 1366x768 panel at 125% scaling above the taskbar", () => {
  const bounds = calculateWindowSizeBounds(workArea(1366, 768, 40, 1.25));

  assert.deepEqual(bounds.maximum, { width: 1092, height: 582 });
  const size = calculateFirstAppWindowSize(bounds);
  assert.deepEqual(size, { width: 900, height: 556 });
  // The window clears the taskbar.
  assert.ok(size.height <= (768 - 40) / 1.25);
});

test("fits a 1080p panel at 175% scaling above the taskbar", () => {
  const bounds = calculateWindowSizeBounds(workArea(1920, 1080, 48, 1.75));

  assert.deepEqual(bounds.maximum, { width: 1097, height: 589 });
  const size = calculateFirstAppWindowSize(bounds);
  assert.deepEqual(size, { width: 900, height: 556 });
  assert.ok(size.height <= (1080 - 48) / 1.75);
});

test("fits a 1600x900 panel at 150% scaling above the taskbar", () => {
  const bounds = calculateWindowSizeBounds(workArea(1600, 900, 40, 1.5));

  assert.deepEqual(bounds.maximum, { width: 1066, height: 573 });
  const size = calculateFirstAppWindowSize(bounds);
  assert.deepEqual(size, { width: 900, height: 556 });
  assert.ok(size.height <= (900 - 40) / 1.5);
});

test("preserves the desktop CSS width floor under Windows text scaling", () => {
  const bounds = calculateWindowSizeBounds(workArea(2880, 1800, 0, 2.5));
  const cssSafeLogicalWidth = 768 * 1.25;

  assert.deepEqual(calculateFirstAppWindowSize(bounds, cssSafeLogicalWidth), {
    width: 960,
    height: 600,
  });
  assert.equal(960 / 1.25, 768);

  const compactBounds = calculateWindowSizeBounds({ width: 920, height: 550 });
  assert.deepEqual(
    calculateFirstAppWindowSize(compactBounds, cssSafeLogicalWidth),
    { width: 920, height: 550 },
  );
});

test("relaxes a minimum that does not fit, keeping room to resize", () => {
  for (const [panel, scaleFactor] of [
    [768, 1.25],
    [768, 1.5],
    [1080, 1.75],
    [900, 1.5],
  ] as const) {
    const { minimum, maximum } = calculateWindowSizeBounds(
      workArea(1366, panel, 40, scaleFactor),
    );
    assert.ok(maximum);
    assert.ok(
      minimum.height <= maximum.height,
      `minimum ${minimum.height} exceeds work area ${maximum.height}`,
    );
    // Keep a vertical resize range.
    assert.ok(
      minimum.height < maximum.height,
      `minimum ${minimum.height} leaves no vertical resize range`,
    );
  }
});

test("fits the non-resizable setup window to the available area", () => {
  const bounds = calculateWindowSizeBounds(workArea(1366, 768, 40, 1.5));

  assert.deepEqual(fitWindowSize(PREFERRED_SETUP_WINDOW_SIZE, bounds.maximum), {
    width: 760,
    height: 485,
  });
  // Roomy panels retain the preferred size.
  assert.deepEqual(
    fitWindowSize(
      PREFERRED_SETUP_WINDOW_SIZE,
      calculateWindowSizeBounds({ width: 1920, height: 1040 }).maximum,
    ),
    PREFERRED_SETUP_WINDOW_SIZE,
  );
});

test("caps an oversized window to a compact work area", () => {
  const bounds = calculateWindowSizeBounds({ width: 1080, height: 550 });

  assert.deepEqual(
    constrainWindowSize({ width: 1400, height: 900 }, bounds.minimum, bounds),
    { width: 1080, height: 550 },
  );
});

test("grows past a stale setup size without exceeding the work area", () => {
  const bounds = calculateWindowSizeBounds(workArea(1366, 768, 40, 1.25));
  const requested = calculateFirstAppWindowSize(bounds);

  assert.deepEqual(
    constrainWindowSize({ width: 760, height: 560 }, requested, bounds),
    { width: 900, height: 560 },
  );
});

test("centers against the work area of the monitor it is on", () => {
  const secondary = {
    position: { x: 1920, y: 40 },
    size: { width: 1366, height: 728 },
  };

  assert.deepEqual(
    calculateCenteredPosition(secondary, { width: 1125, height: 728 }),
    { x: 2040, y: 40 },
  );
  // Oversized windows pin to the work-area origin.
  assert.deepEqual(
    calculateCenteredPosition(secondary, { width: 1500, height: 800 }),
    { x: 1920, y: 40 },
  );
});

test("falls back to the nominal size when no monitor can be read", () => {
  assert.deepEqual(
    calculateFirstAppWindowSize(DEFAULT_APP_WINDOW_SIZE_BOUNDS),
    {
      width: 900,
      height: 600,
    },
  );
  assert.deepEqual(
    fitWindowSize(PREFERRED_SETUP_WINDOW_SIZE, undefined),
    PREFERRED_SETUP_WINDOW_SIZE,
  );
});

test("remeasures a restored window after show on its compact secondary", async () => {
  const events: string[] = [];
  let visible = false;
  let savedSize = { width: 900, height: 556 };
  const monitor = (
    name: string,
    width: number,
    height: number,
    scale: number,
  ) => ({
    name,
    scaleFactor: scale,
    workArea: {
      size: {
        toLogical: () => ({ width: width / scale, height: height / scale }),
      },
    },
  });
  const primary = monitor("roomy primary", 1920, 1040, 1);
  const secondary = monitor("compact secondary", 1366, 728, 1.25);
  const currentMonitor = async () => {
    events.push(`current:${visible ? "visible" : "hidden"}`);
    return visible ? secondary : null;
  };
  const primaryMonitor = async () => {
    events.push("primary");
    return primary;
  };
  const measure = () =>
    measureWindowLayout({ currentMonitor, primaryMonitor }, () => true);

  let measured = await measure();
  assert.ok(measured);
  events.push("restore");
  measured = (await measure()) ?? measured;

  let constrainedMinimum: { width: number; height: number } | undefined;
  let enforcementBounds: Parameters<typeof constrainWindowSize>[2] | undefined;
  await finalizeAppWindowLayout({
    restored: true,
    measured,
    show: async () => {
      events.push("show");
      visible = true;
    },
    measure,
    setMinimumConstraints: async (minimum) => {
      events.push("constraints");
      constrainedMinimum = minimum;
    },
    enforceBounds: async (bounds) => {
      events.push("enforce");
      enforcementBounds = bounds;
      savedSize = constrainWindowSize(savedSize, bounds.minimum, bounds);
    },
    isCurrent: () => true,
  });

  assert.deepEqual(events, [
    "current:hidden",
    "primary",
    "restore",
    "current:hidden",
    "primary",
    "show",
    "current:visible",
    "constraints",
    "enforce",
  ]);
  assert.deepEqual(constrainedMinimum, { width: 900, height: 494 });
  assert.deepEqual(enforcementBounds, {
    minimum: { width: 900, height: 494 },
  });
  assert.deepEqual(savedSize, { width: 900, height: 556 });
});
