// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

const GGUF_SPLIT_SUFFIX = /-\d{3,}-of-\d{3,}(?=\.gguf$)/i;

function normalizeGgufFilename(filename: string): string {
  return filename
    .trim()
    .replace(/\\/g, "/")
    .replace(GGUF_SPLIT_SUFFIX, "")
    .toLowerCase();
}

export function ggufFilenamesMatch(
  left: string | null | undefined,
  right: string | null | undefined,
): boolean {
  if (!(left && right)) return false;
  return normalizeGgufFilename(left) === normalizeGgufFilename(right);
}

export function ggufSelectionOverrideMatchesIntent(
  preferredFile: string | null | undefined,
  preferredFileIntent: number,
  selectedPreferredFile: string | null | undefined,
  selectedPreferredFileIntent: number | undefined,
): boolean {
  return (
    !preferredFile ||
    (selectedPreferredFile === preferredFile &&
      selectedPreferredFileIntent === preferredFileIntent)
  );
}
