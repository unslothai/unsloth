// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Marks <html data-input-modality> as "pointer" or "keyboard" from the last input.
 * Radix refocuses a menu's trigger on close and Chrome counts that as
 * :focus-visible, so CSS uses this to keep keyboard rings off after a click.
 */
export function watchInputModality(win: Window): void {
  const root = win.document.documentElement;
  const set = (modality: "pointer" | "keyboard") => {
    if (root.dataset.inputModality !== modality) root.dataset.inputModality = modality;
  };
  win.addEventListener("pointerdown", () => set("pointer"), { capture: true, passive: true });
  win.addEventListener(
    "keydown",
    (event) => {
      // A bare modifier is not navigation.
      if (!["Shift", "Control", "Alt", "Meta"].includes(event.key)) set("keyboard");
    },
    { capture: true },
  );
}
