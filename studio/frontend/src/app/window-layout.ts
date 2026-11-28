// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export type LogicalWindowSize = {
  width: number;
  height: number;
};

export type PhysicalWindowRect = {
  position: { x: number; y: number };
  size: { width: number; height: number };
};

export type WindowSizeBounds = {
  minimum: LogicalWindowSize;
  maximum?: LogicalWindowSize;
};

export const MINIMUM_APP_WINDOW_SIZE: LogicalWindowSize = {
  width: 460,
  height: 480,
};

/** Separate from the resize floor: shrinking the floor must not shrink the opened window. */
export const NOMINAL_APP_WINDOW_SIZE: LogicalWindowSize = {
  width: 900,
  height: 600,
};

export const PREFERRED_SETUP_WINDOW_SIZE: LogicalWindowSize = {
  width: 760,
  height: 560,
};

// Preserve the nominal fallback when no monitor can be read.
export const DEFAULT_APP_WINDOW_SIZE_BOUNDS: WindowSizeBounds = {
  minimum: MINIMUM_APP_WINDOW_SIZE,
};

const RELAXED_MINIMUM_RATIO = 0.85;
const FIRST_WINDOW_WIDTH_RATIO = 0.75;
const FIRST_WINDOW_HEIGHT_RATIO = 0.85;
const FIRST_WINDOW_ASPECT_RATIO = 1.618;

function relaxMinimum(preferred: number, maximum: number): number {
  if (preferred <= maximum) return preferred;
  return Math.max(1, Math.floor(maximum * RELAXED_MINIMUM_RATIO));
}

/** `logicalPerCssPx` keeps the floor in CSS pixels under Windows text scaling. */
export function calculateWindowSizeBounds(
  workAreaSize: LogicalWindowSize,
  logicalPerCssPx = 1,
): WindowSizeBounds {
  const maximum = {
    width: Math.max(1, Math.floor(workAreaSize.width)),
    height: Math.max(1, Math.floor(workAreaSize.height)),
  };
  const floor = {
    width: Math.round(MINIMUM_APP_WINDOW_SIZE.width * logicalPerCssPx),
    height: Math.round(MINIMUM_APP_WINDOW_SIZE.height * logicalPerCssPx),
  };
  return {
    minimum: {
      width: relaxMinimum(floor.width, maximum.width),
      height: relaxMinimum(floor.height, maximum.height),
    },
    maximum,
  };
}

export function fitWindowSize(
  size: LogicalWindowSize,
  maximum?: LogicalWindowSize,
): LogicalWindowSize {
  if (!maximum) return size;
  return {
    width: Math.min(size.width, maximum.width),
    height: Math.min(size.height, maximum.height),
  };
}

export function calculateFirstAppWindowSize(
  { minimum, maximum }: WindowSizeBounds,
  cssSafeLogicalWidth?: number,
): LogicalWindowSize {
  if (!maximum) return NOMINAL_APP_WINDOW_SIZE;

  // A first window floors at the nominal size, relaxed (never below the floor) on small work areas.
  const nominal = {
    width: Math.max(
      minimum.width,
      relaxMinimum(NOMINAL_APP_WINDOW_SIZE.width, maximum.width),
    ),
    height: Math.max(
      minimum.height,
      relaxMinimum(NOMINAL_APP_WINDOW_SIZE.height, maximum.height),
    ),
  };
  const width = Math.max(
    nominal.width,
    Math.round(maximum.width * FIRST_WINDOW_WIDTH_RATIO),
    Math.min(cssSafeLogicalWidth ?? 0, maximum.width),
  );
  const heightCap = Math.max(
    NOMINAL_APP_WINDOW_SIZE.height,
    Math.round(maximum.height * FIRST_WINDOW_HEIGHT_RATIO),
  );
  const height = Math.max(
    nominal.height,
    Math.min(Math.round(width / FIRST_WINDOW_ASPECT_RATIO), heightCap),
  );
  return fitWindowSize({ width, height }, maximum);
}

export function constrainWindowSize(
  currentSize: LogicalWindowSize,
  requestedSize: LogicalWindowSize,
  { minimum, maximum }: WindowSizeBounds,
): LogicalWindowSize {
  return fitWindowSize(
    {
      width: Math.max(currentSize.width, minimum.width, requestedSize.width),
      height: Math.max(
        currentSize.height,
        minimum.height,
        requestedSize.height,
      ),
    },
    maximum,
  );
}

// Windows' invisible resize borders (scaled by DPI) put a snapped window's outer rect off screen.
const OFF_SCREEN_TOLERANCE_PX = 16;

// Overlapping monitors (mirrored outputs) must not count the same pixels twice, so test the union.
function isCoveredBy(
  rect: PhysicalWindowRect,
  screens: PhysicalWindowRect[],
): boolean {
  const right = rect.position.x + rect.size.width;
  const bottom = rect.position.y + rect.size.height;
  const edges = (start: number, end: number, cuts: number[]) =>
    [
      ...new Set([start, end, ...cuts.filter((c) => c > start && c < end)]),
    ].sort((a, b) => a - b);
  const xs = edges(
    rect.position.x,
    right,
    screens.flatMap((s) => [s.position.x, s.position.x + s.size.width]),
  );
  const ys = edges(
    rect.position.y,
    bottom,
    screens.flatMap((s) => [s.position.y, s.position.y + s.size.height]),
  );
  for (let i = 0; i + 1 < xs.length; i++) {
    for (let j = 0; j + 1 < ys.length; j++) {
      const x = (xs[i] + xs[i + 1]) / 2;
      const y = (ys[j] + ys[j + 1]) / 2;
      const covered = screens.some(
        (s) =>
          x > s.position.x &&
          x < s.position.x + s.size.width &&
          y > s.position.y &&
          y < s.position.y + s.size.height,
      );
      if (!covered) return false;
    }
  }
  return true;
}

/** Outer rect shrunk and moved into the work area if part is off every screen (#19590), else null. */
export function calculateOnScreenRect(
  window: PhysicalWindowRect,
  screens: PhysicalWindowRect[],
  workArea: PhysicalWindowRect,
  scaleFactor = 1,
): PhysicalWindowRect | null {
  const inset = Math.min(
    Math.round(OFF_SCREEN_TOLERANCE_PX * Math.max(1, scaleFactor)),
    Math.floor(window.size.width / 4),
    Math.floor(window.size.height / 4),
  );
  const core: PhysicalWindowRect = {
    position: { x: window.position.x + inset, y: window.position.y + inset },
    size: {
      width: Math.max(1, window.size.width - 2 * inset),
      height: Math.max(1, window.size.height - 2 * inset),
    },
  };
  if (isCoveredBy(core, screens)) return null;

  const width = Math.max(1, Math.min(window.size.width, workArea.size.width));
  const height = Math.max(
    1,
    Math.min(window.size.height, workArea.size.height),
  );
  const clamp = (value: number, start: number, room: number) =>
    Math.min(Math.max(value, start), start + Math.max(0, room));
  return {
    position: {
      x: clamp(
        window.position.x,
        workArea.position.x,
        workArea.size.width - width,
      ),
      y: clamp(
        window.position.y,
        workArea.position.y,
        workArea.size.height - height,
      ),
    },
    size: { width, height },
  };
}

export function calculateCenteredPosition(
  workArea: PhysicalWindowRect,
  windowSize: { width: number; height: number },
): { x: number; y: number } {
  return {
    x:
      workArea.position.x +
      Math.max(0, Math.floor((workArea.size.width - windowSize.width) / 2)),
    y:
      workArea.position.y +
      Math.max(0, Math.floor((workArea.size.height - windowSize.height) / 2)),
  };
}
