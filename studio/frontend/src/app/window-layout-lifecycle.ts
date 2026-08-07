


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

/** Size bounds the window has to stay within on its current monitor. */
export async function measureWindowLayout<Monitor extends WorkAreaMonitor>(
  reader: WindowMonitorReader<Monitor>,
  isCurrent: WindowLayoutGuard,
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
    ? calculateWindowSizeBounds(availableInnerSize)
    : DEFAULT_APP_WINDOW_SIZE_BOUNDS;
  return { bounds, monitor, frameSize };
}

export function shouldFinishWindowLayoutWait(
  sawPostShowChange: boolean,
): boolean {
  return sawPostShowChange;
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

/** Shows the app window, then applies bounds from the visible monitor. */
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
  const shown = await show();
  if (!isCurrent()) return;
  // A restored hidden autostart cannot reliably resolve its saved monitor yet.
  // Keep the plugin-restored geometry untouched until native tray reveal.
  if (restored && !shown) return;

  // Native restore calls complete before GTK/Cocoa move and resize events have
  // necessarily updated Tauri's cached geometry.
  if (restored) {
    await waitForSettled?.();
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
