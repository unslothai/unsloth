// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// In the macOS desktop app the webview is a child view (Tauri `unstable`). A key the page leaves
// unhandled (left at the start of a field, right at its end, Home, End) is re-sent to the
// window, whose text input types the key's control character (U+001C, U+001D, U+0001, U+0004)
// into the field (#14764, tauri-apps/tauri#10194). No key types a control character on purpose, so
// cancelling those insertions leaves every key's own behaviour as it was.

// C0 controls and DEL, except tab, line feed and carriage return.
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
