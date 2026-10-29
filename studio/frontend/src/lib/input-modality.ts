// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Sets <html data-input-modality> so CSS can hide keyboard rings after a click: Radix refocuses
 * a menu trigger on close and Chrome counts that as :focus-visible. */
export function watchInputModality(win: Window): void {
  const root = win.document.documentElement;
  const set = (modality: "pointer" | "keyboard") => {
    if (root.dataset.inputModality !== modality) root.dataset.inputModality = modality;
  };
  win.addEventListener("pointerdown", () => set("pointer"), { capture: true, passive: true });
  win.addEventListener(
    "keydown",
    (event) => {
      if (!["Shift", "Control", "Alt", "Meta"].includes(event.key)) set("keyboard");
    },
    { capture: true },
  );
}
