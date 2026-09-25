// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  DEFAULT_APP_WINDOW_SIZE_BOUNDS,
  type LogicalWindowSize,
  type WindowSizeBounds,
  calculateWindowSizeBounds,
} from "./window-layout.ts";

export type WindowLayoutGuard = () => boolean;

export async function prepareSetupWindow(options: {
  resetLayout: () => Promise<unknown>;
  unmaximize: () => Promise<void>;
  clearConstraints: () => Promise<void>;
  enableResize: () => Promise<void>;
  resizeForSetup: () => Promise<boolean>;
  disableResize: () => Promise<void>;
  isCurrent: WindowLayoutGuard;
}): Promise<boolean> {
  const {
    resetLayout,
    unmaximize,
    clearConstraints,
    enableResize,
    resizeForSetup,
    disableResize,
    isCurrent,
  } = options;
  await resetLayout();
  if (!isCurrent()) return false;
  await unmaximize();
  if (!isCurrent()) return false;
  await clearConstraints();
  if (!isCurrent()) return false;
  // GTK can ignore a size change on a non-resizable restored window.
  await enableResize();
  if (!isCurrent()) return false;
  if (!(await resizeForSetup()) || !isCurrent()) return false;
  await disableResize();
  return isCurrent();
}

type WorkAreaMonitor = {
  scaleFactor: number;
  workArea: {
    size: {
      toLogical: (scaleFactor: number) => LogicalWindowSize;
    };
  };
};

type PhysicalWindowSize = {
  width: number;
  height: number;
};

export type MeasuredWindowLayout<Monitor extends WorkAreaMonitor> = {
  bounds: WindowSizeBounds;
  monitor: Monitor | null;
  /** Physical pixels outside the webview's inner rectangle. */
  frameSize: PhysicalWindowSize;
};

type WindowMonitorReader<Monitor extends WorkAreaMonitor> = {
  currentMonitor: () => Promise<Monitor | null>;
  primaryMonitor: () => Promise<Monitor | null>;
  innerSize?: () => Promise<PhysicalWindowSize>;
  outerSize?: () => Promise<PhysicalWindowSize>;
};

/**
 * Size bounds the window has to stay within on its current monitor.
 *
 * `logicalPerCssPx` reports the webview's zoom above a monitor's display scale,
 * keeping the resize floor a CSS-pixel floor under Windows text scaling. It
 * defaults to a no-op, which is every platform without it.
 */
export async function measureWindowLayout<Monitor extends WorkAreaMonitor>(
  reader: WindowMonitorReader<Monitor>,
  isCurrent: WindowLayoutGuard,
  logicalPerCssPx: (monitorScale: number) => number = () => 1,
): Promise<MeasuredWindowLayout<Monitor> | null> {
  // Some platforms cannot resolve the monitor for a hidden window.
  const monitor =
    (await reader.currentMonitor()) ?? (await reader.primaryMonitor());
  if (!isCurrent()) return null;

  const frameSize = { width: 0, height: 0 };
  let availableInnerSize: LogicalWindowSize | undefined;
  if (monitor) {
    availableInnerSize = monitor.workArea.size.toLogical(monitor.scaleFactor);
    if (reader.innerSize && reader.outerSize) {
      const [innerSize, outerSize] = await Promise.all([
        reader.innerSize(),
        reader.outerSize(),
      ]);
      if (!isCurrent()) return null;
      frameSize.width = Math.max(0, outerSize.width - innerSize.width);
      frameSize.height = Math.max(0, outerSize.height - innerSize.height);
      // Tauri sizes the inner rectangle but positions the outer rectangle.
      availableInnerSize = {
        width: Math.max(
          1,
          availableInnerSize.width - frameSize.width / monitor.scaleFactor,
        ),
        height: Math.max(
          1,
          availableInnerSize.height - frameSize.height / monitor.scaleFactor,
        ),
      };
    }
  }

  const bounds = availableInnerSize
    ? calculateWindowSizeBounds(
        availableInnerSize,
        monitor ? logicalPerCssPx(monitor.scaleFactor) : 1,
      )
    : DEFAULT_APP_WINDOW_SIZE_BOUNDS;
  return { bounds, monitor, frameSize };
}

export function shouldFinishWindowLayoutWait(
  sawNativeChange: boolean,
): boolean {
  return sawNativeChange;
}

type ResolutionQuery = {
  addEventListener: (type: "change", listener: () => void) => void;
  removeEventListener: (type: "change", listener: () => void) => void;
};

export type PixelRatioSource = {
  devicePixelRatio: () => number;
  matchResolution: (dppx: number) => ResolutionQuery | null;
};

/**
 * Reports a change in the webview's device pixel ratio, which moves the
 * CSS-pixel resize floor and is otherwise only read at launch. There is no
 * event for the ratio itself, so a query for the ratio in force stands in: it
 * stops matching, and a fresh query for the new one takes over.
 */
export function observeDevicePixelRatio(
  source: PixelRatioSource,
  onChange: () => void,
): () => void {
  let query: ResolutionQuery | null = null;
  let disposed = false;
  const listen = () => {
    query?.removeEventListener("change", handle);
    query = disposed ? null : source.matchResolution(source.devicePixelRatio());
    query?.addEventListener("change", handle);
  };
  function handle() {
    listen();
    if (!disposed) onChange();
  }
  listen();
  return () => {
    disposed = true;
    query?.removeEventListener("change", handle);
    query = null;
  };
}
type FinalizeAppWindowLayoutOptions<Monitor extends WorkAreaMonitor> = {
  restored: boolean;
  measured: MeasuredWindowLayout<Monitor>;
  show: () => Promise<boolean>;
  waitForSettled?: () => Promise<void>;
  measure: () => Promise<MeasuredWindowLayout<Monitor> | null>;
  setMinimumConstraints: (minimum: LogicalWindowSize) => Promise<void>;
  enforceBounds: (bounds: WindowSizeBounds) => Promise<void>;
  isCurrent: WindowLayoutGuard;
};

/** Reveals the settled app window, then applies bounds from the visible monitor. */
export async function finalizeAppWindowLayout<Monitor extends WorkAreaMonitor>({
  restored,
  measured,
  show,
  waitForSettled,
  measure,
  setMinimumConstraints,
  enforceBounds,
  isCurrent,
}: FinalizeAppWindowLayoutOptions<Monitor>): Promise<void> {
  if (!isCurrent()) return;
  // restoreState returns before native resize lands; showing now flashes the setup size.
  if (restored) {
    await waitForSettled?.();
    if (!isCurrent()) return;
  }
  const shown = await show();
  if (!isCurrent()) return;
  // A restored hidden autostart cannot reliably resolve its saved monitor yet.
  // Keep the plugin-restored geometry untouched until native tray reveal.
  if (restored && !shown) return;
  // Showing can change the resolved monitor (e.g. a compact secondary).
  if (restored) {
    if (!isCurrent()) return;
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
