// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

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

export function onMainFocus(
  focused: boolean,
  w: WidgetReturn,
  setCollapsed: (collapsed: boolean) => void,
  showMain: () => void,
): boolean {
  return focused && returnFromWidget(w, setCollapsed, showMain);
}
