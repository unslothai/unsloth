// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Coming back from the floating orb. The widget's own Back button was the only way this
 * side heard about it: a Dock click, the tray or the app menu bring the main window back
 * natively (show_main_window), so the page still thought it was floating and kept the orb
 * collapsed, and the widget could be left up beside it. Any return now goes through here:
 * the in-app orb is shown again and the native side hides the widget.
 */
export interface WidgetReturn {
  floating: boolean;
}

export function returnFromWidget(
  w: WidgetReturn,
  setCollapsed: (collapsed: boolean) => void,
  showMain: () => void,
): boolean {
  if (!w.floating) return false;
  w.floating = false;
  setCollapsed(false);
  showMain();
  return true;
}

/** The main window took focus while the widget was up: that is a return too. */
export function onMainFocus(
  focused: boolean,
  w: WidgetReturn,
  setCollapsed: (collapsed: boolean) => void,
  showMain: () => void,
): boolean {
  return focused && returnFromWidget(w, setCollapsed, showMain);
}
