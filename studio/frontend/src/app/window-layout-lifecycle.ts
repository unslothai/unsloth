// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  DEFAULT_APP_WINDOW_SIZE_BOUNDS,
  type LogicalWindowSize,
  type WindowSizeBounds,
  calculateWindowSizeBounds,
} from "./window-layout.ts";

export type WindowLayoutGuard = () => boolean;

type WorkAreaMonitor = {
  scaleFactor: number;
  workArea: {
    size: {
      toLogical: (scaleFactor: number) => LogicalWindowSize;
    };
  };
};

export type MeasuredWindowLayout<Monitor extends WorkAreaMonitor> = {
  bounds: WindowSizeBounds;
  monitor: Monitor | null;
};

type WindowMonitorReader<Monitor extends WorkAreaMonitor> = {
  currentMonitor: () => Promise<Monitor | null>;
  primaryMonitor: () => Promise<Monitor | null>;
};

/** Size bounds the window has to stay within on its current monitor. */
export async function measureWindowLayout<Monitor extends WorkAreaMonitor>(
  { currentMonitor, primaryMonitor }: WindowMonitorReader<Monitor>,
  isCurrent: WindowLayoutGuard,
): Promise<MeasuredWindowLayout<Monitor> | null> {
  // Some platforms cannot resolve the monitor for a hidden window.
  const monitor = (await currentMonitor()) ?? (await primaryMonitor());
  if (!isCurrent()) return null;

  const bounds = monitor
    ? calculateWindowSizeBounds(
        monitor.workArea.size.toLogical(monitor.scaleFactor),
      )
    : DEFAULT_APP_WINDOW_SIZE_BOUNDS;
  return { bounds, monitor };
}

type FinalizeAppWindowLayoutOptions<Monitor extends WorkAreaMonitor> = {
  restored: boolean;
  measured: MeasuredWindowLayout<Monitor>;
  show: () => Promise<void>;
  measure: () => Promise<MeasuredWindowLayout<Monitor> | null>;
  setMinimumConstraints: (minimum: LogicalWindowSize) => Promise<void>;
  enforceBounds: (bounds: WindowSizeBounds) => Promise<void>;
  isCurrent: WindowLayoutGuard;
};

/** Shows the app window, then applies bounds from the visible monitor. */
export async function finalizeAppWindowLayout<Monitor extends WorkAreaMonitor>({
  restored,
  measured,
  show,
  measure,
  setMinimumConstraints,
  enforceBounds,
  isCurrent,
}: FinalizeAppWindowLayoutOptions<Monitor>): Promise<void> {
  if (!isCurrent()) return;
  await show();
  if (!isCurrent()) return;

  // Once visible, a restored window can resolve its saved secondary monitor.
  if (restored) {
    measured = (await measure()) ?? measured;
    if (!isCurrent()) return;
  }

  await setMinimumConstraints(measured.bounds.minimum);
  if (!isCurrent()) return;
  // Do not cap restored sizes against a temporary monitor fallback.
  await enforceBounds(
    restored ? { minimum: measured.bounds.minimum } : measured.bounds,
  );
}
