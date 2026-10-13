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

/**
 * Resize floor: a companion width, as other chat apps allow, and no narrower
 * than the desktop layout it keeps at every size can fit.
 */
export const MINIMUM_APP_WINDOW_SIZE: LogicalWindowSize = {
  width: 460,
  height: 480,
};

/**
 * The size a first launch aims for. Separate from the resize floor: shrinking
 * the floor must not shrink the window we open.
 */
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

// Leave resize room when the nominal minimum does not fit.
const RELAXED_MINIMUM_RATIO = 0.85;
const FIRST_WINDOW_WIDTH_RATIO = 0.75;
const FIRST_WINDOW_HEIGHT_RATIO = 0.85;
const FIRST_WINDOW_ASPECT_RATIO = 1.618;

function relaxMinimum(preferred: number, maximum: number): number {
  if (preferred <= maximum) return preferred;
  return Math.max(1, Math.floor(maximum * RELAXED_MINIMUM_RATIO));
}

/**
 * Bounds a frameless window to the monitor work area.
 *
 * `logicalPerCssPx` keeps the floor a CSS-pixel floor. Windows text scaling
 * zooms the webview above the display scale, so a window sized in logical
 * pixels lays out in fewer CSS pixels: at 150% a 460px floor is a 307px
 * viewport, narrower than the layout can hold.
 */
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

  // A first window floors at the nominal size, not the resize floor: opening
  // at a width the user may shrink to would be a surprise. Never below the
  // floor either: a work area too small for the nominal size relaxes it, and
  // the constraints would then grow the window off the centre it was placed on.
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
  // Preserve requested height when the work area is short.
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

/** Centers a physical window size inside the work area. */
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
