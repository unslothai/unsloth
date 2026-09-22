// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Where the Live resource monitor stands relative to the Run settings panel.
 *
 * The monitor lives in the bottom-right corner, which is exactly where the
 * settings panel opens. A 256 px monitor and a 248-560 px panel cannot both
 * hold that corner, and the monitor is the one that can move: docking it to
 * the panel's left edge leaves both usable. A window too narrow to put the two
 * side by side has nowhere to dock to, so the monitor yields to the sheet
 * instead of hiding behind it.
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
}

export function getFloatingMonitorLayout({
  isOpen,
  isMobile,
  isChatRoute,
  settingsPanelOpen,
}: FloatingMonitorDockState): {
  visible: boolean;
  dockedBesideRunSettings: boolean;
} {
  // The panel is chat-only, and the store survives navigation, so a stale open
  // flag on another route must not move the monitor.
  const runSettingsVisible = isChatRoute && settingsPanelOpen;
  const hiddenBehindMobileSettings = isOpen && isMobile && runSettingsVisible;

  return {
    visible: isOpen && !hiddenBehindMobileSettings,
    dockedBesideRunSettings: isOpen && !isMobile && runSettingsVisible,
  };
}

/**
 * The inset the monitor takes from the viewport's right edge, in px.
 *
 * Docked, the container ends exactly where the panel begins, so the monitor
 * cannot overlap it at any width the panel can be dragged to. That width is the
 * panel's own live width (`useChatSettingsWidth`), not a constant: a fixed
 * offset is only correct at one panel width.
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
 * The constraint container's inline style.
 *
 * Returned as an object rather than written inline so the geometry a reviewer
 * checks is the geometry the browser applies, byte for byte.
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
    right: floatingMonitorRightInset({ dockedBesideRunSettings, settingsWidth }),
  };
}
