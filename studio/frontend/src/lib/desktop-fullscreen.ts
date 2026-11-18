// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Whether the native desktop window is fullscreen; the custom titlebar hides while it is.
let fullscreen = false;
const listeners = new Set<() => void>();

function publish(next: boolean): void {
  if (next === fullscreen) return;
  fullscreen = next;
  for (const listener of listeners) listener();
}

export function getDesktopFullscreen(): boolean {
  return fullscreen;
}

export function subscribeDesktopFullscreen(listener: () => void): () => void {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}

/** Re-read the native state; the window manager can change it without us (e.g. its own shortcut). */
export async function refreshDesktopFullscreen(): Promise<void> {
  const { getCurrentWindow } = await import("@tauri-apps/api/window");
  publish(await getCurrentWindow().isFullscreen());
}

let changing = false;
const RESIZE_SETTLE_MS = 250;

/** Enter or leave fullscreen; `next` omitted toggles. A call during a transition is dropped. */
export async function setDesktopFullscreen(next?: boolean): Promise<void> {
  // A second request while the native window is transitioning must not race the first.
  if (changing) return;
  changing = true;
  try {
    const { getCurrentWindow } = await import("@tauri-apps/api/window");
    const win = getCurrentWindow();
    const target = next ?? !(await win.isFullscreen());
    await win.setFullscreen(target);
    // Some window managers apply it after the call resolves; the resize refresh corrects us then.
    publish(target);
  } finally {
    changing = false;
  }
}

/**
 * Toggle the native desktop window with F11; web pages keep their browser shortcut.
 * `keyTaken` lets a user's own F11 binding from Settings win.
 */
export function watchDesktopFullscreen(
  target: Window,
  keyTaken: () => boolean = () => false,
): void {
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
    setDesktopFullscreen().catch((error: unknown) =>
      console.error("Could not toggle fullscreen", error),
    );
  });
  // The resize can land before the window manager reports the new state, so read it once things settle.
  let settle: ReturnType<typeof setTimeout> | undefined;
  target.addEventListener("resize", () => {
    clearTimeout(settle);
    settle = setTimeout(() => {
      refreshDesktopFullscreen().catch(() => undefined);
    }, RESIZE_SETTLE_MS);
  });
  refreshDesktopFullscreen().catch(() => undefined);
}
