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
/**
 * The settings aside's resize handle is `-left-1 w-2`, so half of its target
 * sits outside the aside. Both are Tailwind spacing utilities, which resolve
 * through `--spacing: calc(0.25rem * var(--ui-space-scale, 1))` (index.css),
 * so the outward half is 4px at the 15px default UI font and 5.333px at the
 * 20px maximum. The monitor's layer paints above the aside and would
 * otherwise swallow that outward half, so the clearance has to be derived
 * from the live scale rather than pinned at the default's 4px.
 */
export const FLOATING_MONITOR_HANDLE_HALF_WIDTH = 4;

/**
 * The outward half of the resize handle at a given `--ui-space-scale`. Both
 * callers can hand over a non-finite or non-positive reading (the CSS variable
 * before the appearance store paints, or a cleared store), so anything that is
 * not a usable scale falls back to the shipped 1:1 default.
 */
export function floatingMonitorHandleClearance(uiSpaceScale?: number): number {
  const scale =
    uiSpaceScale !== undefined && Number.isFinite(uiSpaceScale) && uiSpaceScale > 0
      ? uiSpaceScale
      : 1;
  return FLOATING_MONITOR_HANDLE_HALF_WIDTH * scale;
}

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
  /** Live --ui-space-scale; defaults to 1 before the appearance setting loads. */
  uiSpaceScale?: number;
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
  uiSpaceScale,
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
      uiSpaceScale,
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
  uiSpaceScale,
}: {
  viewportWidth: number;
  sidebarWidth: number;
  settingsWidth: number;
  monitorWidth?: number;
  uiSpaceScale?: number;
}): boolean {
  // An unmeasured window keeps the dock rather than hiding the monitor on a guess.
  if (!(viewportWidth > 0)) {
    return true;
  }
  const usable =
    viewportWidth - Math.max(0, sidebarWidth) - Math.max(0, settingsWidth);
  const scale =
    uiSpaceScale !== undefined && Number.isFinite(uiSpaceScale) && uiSpaceScale > 0
      ? uiSpaceScale
      : 1;
  // The monitor starts at the scaled left edge inset and ends at the panel's
  // scaled resize-handle clearance; there is no second full edge inset on right.
  const width =
    monitorWidth === undefined || !Number.isFinite(monitorWidth)
      ? FLOATING_MONITOR_WIDTH * scale
      : monitorWidth;
  return (
    usable >=
    width +
      FLOATING_MONITOR_EDGE_INSET * scale +
      floatingMonitorHandleClearance(scale)
  );
}

/**
 * The inset from the viewport's right edge, in px. Docked, the container ends
 * where the panel begins, so the monitor clears it at every width the panel can
 * be dragged to; a fixed offset is only correct at one width.
 */
export function floatingMonitorRightInset({
  dockedBesideRunSettings,
  settingsWidth,
  uiSpaceScale = 1,
}: {
  dockedBesideRunSettings: boolean;
  settingsWidth: number;
  uiSpaceScale?: number;
}): number {
  return dockedBesideRunSettings
    ? settingsWidth + floatingMonitorHandleClearance(uiSpaceScale)
    : FLOATING_MONITOR_EDGE_INSET *
        (Number.isFinite(uiSpaceScale) && uiSpaceScale > 0 ? uiSpaceScale : 1);
}

/**
 * The constraint container's inline style. Returned as an object so the
 * geometry a reviewer checks is the geometry the browser applies.
 */
export function floatingMonitorConstraintStyle({
  zIndex,
  dockedBesideRunSettings,
  settingsWidth,
  uiSpaceScale,
}: {
  zIndex: number;
  dockedBesideRunSettings: boolean;
  settingsWidth: number;
  uiSpaceScale?: number;
}): { zIndex: number; right: number } {
  return {
    zIndex,
    right: floatingMonitorRightInset({
      dockedBesideRunSettings,
      settingsWidth,
      uiSpaceScale,
    }),
  };
}
