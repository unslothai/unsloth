// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Toggle the native desktop window with F11; web pages keep their browser shortcut.
 * `keyTaken` lets a user's own F11 binding from Settings win.
 */
export function watchDesktopFullscreen(
  target: Window,
  keyTaken: () => boolean = () => false,
): void {
  let changing = false;
  target.addEventListener("keydown", (event) => {
    if (
      event.defaultPrevented ||
      event.code !== "F11" ||
      event.repeat ||
      event.altKey ||
      event.ctrlKey ||
      event.metaKey ||
      event.shiftKey ||
      keyTaken()
    )
      return;

    event.preventDefault();
    // A second press while the native window is transitioning must not race the first.
    if (changing) return;
    changing = true;
    void import("@tauri-apps/api/window")
      .then(async ({ getCurrentWindow }) => {
        const win = getCurrentWindow();
        await win.setFullscreen(!(await win.isFullscreen()));
      })
      .catch((error: unknown) =>
        console.error("Could not toggle fullscreen", error),
      )
      .finally(() => {
        changing = false;
      });
  });
}
