// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export function saveDirectoryField(
  custom: string | null,
  defaultSaveDirectory: string,
): { inputValue: string; saveDirectory: string } {
  // Echoing the trimmed path back into the input would drop a space as it is
  // typed and refill a cleared field with the default.
  return {
    inputValue: custom ?? defaultSaveDirectory,
    saveDirectory: custom?.trim() || defaultSaveDirectory,
  };
}
