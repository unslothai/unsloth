// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** The monitor docks left of the Run settings panel, or yields where there is no room. */

export const FLOATING_MONITOR_EDGE_INSET = 16;
/** Half the aside's resize handle sits outside it; it scales with --ui-space-scale (index.css). */
export const FLOATING_MONITOR_HANDLE_HALF_WIDTH = 4;

/** Non-finite or non-positive scales fall back to 1. */
export function floatingMonitorHandleClearance(uiSpaceScale?: number): number {
  const scale =
    uiSpaceScale !== undefined && Number.isFinite(uiSpaceScale) && uiSpaceScale > 0
      ? uiSpaceScale
      : 1;
  return FLOATING_MONITOR_HANDLE_HALF_WIDTH * scale;
}

export const FLOATING_MONITOR_WIDTH = 256;

export interface FloatingMonitorDockState {
  isOpen: boolean;
  isMobile: boolean;
  isChatRoute: boolean;
  settingsPanelOpen: boolean;
  /** The panel's painted width, which leads the stored one during a drag. */
  settingsWidth: number;
  sidebarWidth: number;
  /** Viewport width; 0 or less means unknown, which skips the capacity check. */
  viewportWidth: number;
  /** The monitor's rendered width; natively resizable, so `w-64` is a floor. */
  monitorWidth?: number;
  uiSpaceScale?: number;
}

export interface FloatingMonitorLayout {
  visible: boolean;
  /** Hidden because the settings panel covers the corner; the caller keeps it mounted. */
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
  // The store survives navigation, so a stale open flag on another route must not move the monitor.
  const runSettingsVisible = isChatRoute && settingsPanelOpen;
  // Docking needs room for sidebar, monitor and panel; at 768px there is not enough.
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
