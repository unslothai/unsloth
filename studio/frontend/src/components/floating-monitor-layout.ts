// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Where the Live resource monitor stands relative to the Run settings panel.
 *
 * The monitor and the panel share the bottom-right corner, and the monitor is
 * the one that can move: docking it left of the panel leaves both usable. It
 * yields instead where there is nowhere to dock to.
 */

/** The resting inset on every edge, `inset-4` at a 16px root font size. */
export const FLOATING_MONITOR_EDGE_INSET = 16;

/** The monitor's own width, `w-64`. */
export const FLOATING_MONITOR_WIDTH = 256;

export interface FloatingMonitorDockState {
  isOpen: boolean;
  isMobile: boolean;
  isChatRoute: boolean;
  settingsPanelOpen: boolean;
  /** The panel's painted width, which leads the stored one during a drag. */
  settingsWidth: number;
  /** A pinned sidebar's width, 0 when it holds no column. */
  sidebarWidth: number;
  /** Viewport width; 0 or less means unknown, which skips the capacity check. */
  viewportWidth: number;
  /** The monitor's rendered width; natively resizable, so `w-64` is a floor. */
  monitorWidth?: number;
}

export interface FloatingMonitorLayout {
  visible: boolean;
  /**
   * Hidden because the settings panel covers the corner, not because the user
   * closed the monitor. The caller keeps the panel mounted so its geometry
   * survives a temporary overlay.
   */
  suppressed: boolean;
  dockedBesideRunSettings: boolean;
}

export function getFloatingMonitorLayout({
  isOpen,
  isMobile,
  isChatRoute,
  settingsPanelOpen,
  settingsWidth,
  sidebarWidth,
  viewportWidth,
  monitorWidth,
}: FloatingMonitorDockState): FloatingMonitorLayout {
  // The panel is chat-only and the store survives navigation, so a stale open
  // flag on another route must not move the monitor.
  const runSettingsVisible = isChatRoute && settingsPanelOpen;
  // Docking needs room for the sidebar, the monitor and the panel at once.
  // At the 768 px breakpoint the default 280 px sidebar and 272 px panel leave
  // less than the monitor's own width, and a docked monitor would cover the
  // sidebar instead of the panel.
  const dockable =
    !isMobile &&
    dockedMonitorFits({
      viewportWidth,
      sidebarWidth,
      settingsWidth,
      monitorWidth,
    });
  const suppressed = isOpen && runSettingsVisible && !dockable;

  return {
    visible: isOpen && !suppressed,
    suppressed,
    dockedBesideRunSettings: isOpen && runSettingsVisible && dockable,
  };
}

export function dockedMonitorFits({
  viewportWidth,
  sidebarWidth,
  settingsWidth,
  monitorWidth,
}: {
  viewportWidth: number;
  sidebarWidth: number;
  settingsWidth: number;
  monitorWidth?: number;
}): boolean {
  // An unmeasured window keeps the dock rather than hiding the monitor on a guess.
  if (!(viewportWidth > 0)) {
    return true;
  }
  const usable =
    viewportWidth - Math.max(0, sidebarWidth) - Math.max(0, settingsWidth);
  // A hand-resized monitor is wider than `w-64`, and the constant also scales
  // with `--ui-space-scale`. Reserving only the constant docks it over the
  // sidebar, so reserve what the panel actually renders.
  const width =
    monitorWidth === undefined || !Number.isFinite(monitorWidth)
      ? FLOATING_MONITOR_WIDTH
      : Math.max(monitorWidth, FLOATING_MONITOR_WIDTH);
  return usable >= width + 2 * FLOATING_MONITOR_EDGE_INSET;
}

/**
 * The inset from the viewport's right edge, in px. Docked, the container ends
 * where the panel begins, so the monitor clears it at every width the panel can
 * be dragged to; a fixed offset is only correct at one width.
 */
export function floatingMonitorRightInset({
  dockedBesideRunSettings,
  settingsWidth,
}: {
  dockedBesideRunSettings: boolean;
  settingsWidth: number;
}): number {
  return dockedBesideRunSettings ? settingsWidth : FLOATING_MONITOR_EDGE_INSET;
}

/**
 * The constraint container's inline style. Returned as an object so the
 * geometry a reviewer checks is the geometry the browser applies.
 */
export function floatingMonitorConstraintStyle({
  zIndex,
  dockedBesideRunSettings,
  settingsWidth,
}: {
  zIndex: number;
  dockedBesideRunSettings: boolean;
  settingsWidth: number;
}): { zIndex: number; right: number } {
  return {
    zIndex,
    right: floatingMonitorRightInset({
      dockedBesideRunSettings,
      settingsWidth,
    }),
  };
}
