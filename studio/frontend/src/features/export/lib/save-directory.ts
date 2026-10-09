// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export function saveDirectoryField(
  custom: string | null,
  defaultSaveDirectory: string,
): { inputValue: string; saveDirectory: string } {
  const saveDirectory = custom?.trim() || defaultSaveDirectory;
  return { inputValue: saveDirectory, saveDirectory };
}
