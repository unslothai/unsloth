// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The macOS app's child webview (Tauri `unstable`) re-sends a key the page leaves unhandled to the
// window, whose text input then types its control character into the field: U+001C left at the
// start, U+001D right at the end, U+0001 Home, U+0004 End (#14764, tauri-apps/tauri#10194). No key
// types C0 controls or DEL on purpose; tab, line feed and carriage return are kept.
// biome-ignore lint/suspicious/noControlCharactersInRegex: control characters are what this finds
const STRAY_KEY_TEXT = /^[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f]+$/;

export function isStrayKeyText(
  inputType: string,
  data: string | null,
): boolean {
  return (
    inputType === "insertText" &&
    typeof data === "string" &&
    STRAY_KEY_TEXT.test(data)
  );
}

export function watchMacStrayKeyText(win: Window): void {
  win.addEventListener(
    "beforeinput",
    (event) => {
      if (isStrayKeyText(event.inputType, event.data)) {
        event.preventDefault();
      }
    },
    true,
  );
}
