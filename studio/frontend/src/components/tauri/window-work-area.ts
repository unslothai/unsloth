// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { Window as TauriWindow } from "@tauri-apps/api/window";

export async function isFillingWorkArea(appWindow: TauriWindow): Promise<boolean> {
  return appWindow.isMaximized();
}

/** Maximize, or restore, the way dragging to the top edge does.
 *
 * why: the window is configured `resizable: false` because the app draws its own
 * resize handles, and maximizing a non-resizable window is undefined on Windows,
 * where it hides the window instead. Turning resizing on for the duration lets the
 * OS run its normal maximize, which lands flush with the work area. Sizing to the
 * work area by hand does not: Windows reports an outer rect that includes the
 * invisible resize border, so the visible window ends up inset a few pixels.
 *
 * Resizing goes back off on restore, so the custom handles stay the only way to
 * resize a floating window. */
export async function toggleWorkAreaFill(appWindow: TauriWindow): Promise<void> {
  if (await appWindow.isMaximized()) {
    await appWindow.toggleMaximize();
    await appWindow.setResizable(false);
    return;
  }
  await appWindow.setResizable(true);
  await appWindow.toggleMaximize();
}
