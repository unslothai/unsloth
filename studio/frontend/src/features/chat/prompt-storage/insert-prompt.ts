// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { insertAtCaret } from "../utils/rich-paste";

export function insertSavedPrompt(
  input: HTMLTextAreaElement | null,
  text: string,
  setText: (next: string) => void,
): void {
  if (!input?.value) {
    setText(text);
    return;
  }
  // After the menu or dialog lets go of focus, so the insert lands in the composer.
  requestAnimationFrame(() => {
    const { selectionStart, selectionEnd, value } = input;
    const before = value.slice(0, selectionStart);
    const after = value.slice(selectionEnd);
    const insert = `${before && !before.endsWith("\n") ? "\n\n" : ""}${text}${after && !after.startsWith("\n") ? "\n\n" : ""}`;
    insertAtCaret(input, insert, (next, caret) => {
      setText(next);
      requestAnimationFrame(() => input.setSelectionRange(caret, caret));
    });
  });
}
