// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** 1 zooms in, -1 out, 0 back to 100%. */
export type ZoomDirection = 1 | -1 | 0;

/** Chord per direction. A user shortcut bound to one of these wins. */
export const ZOOM_CHORDS: Record<ZoomDirection, string> = {
  1: "Mod+Equal",
  [-1]: "Mod+Minus",
  0: "Mod+Digit0",
};

type ZoomKeyEvent = Pick<
  KeyboardEvent,
  "key" | "code" | "metaKey" | "ctrlKey" | "altKey"
>;

/** Browser-style zoom keys: Cmd on macOS, Ctrl elsewhere, Shift optional, keypad included. */
export function zoomDirectionForKey(
  event: ZoomKeyEvent,
  mac: boolean,
): ZoomDirection | null {
  const mod = mac
    ? event.metaKey && !event.ctrlKey
    : event.ctrlKey && !event.metaKey;
  if (!mod || event.altKey) return null;
  if (event.key === "+" || event.key === "=" || event.code === "NumpadAdd")
    return 1;
  if (event.key === "-" || event.key === "_" || event.code === "NumpadSubtract")
    return -1;
  if (event.key === "0" || event.code === "Digit0" || event.code === "Numpad0")
    return 0;
  if (event.code === "Equal") return 1;
  if (event.code === "Minus") return -1;
  return null;
}
