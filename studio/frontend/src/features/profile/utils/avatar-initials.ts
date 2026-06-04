// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export function initialsFromName(name: string): string {
  const trimmed = name.trim();
  if (!trimmed) return "?";
  return trimmed[0]!.toUpperCase();
}

/**
 * Default Unsloth-brand background for the avatar fallback (readable white text).
 *
 * Uses the shared `--primary` design token (#17b88b) instead of a one-off
 * hardcoded shade, so the avatar always matches the app's general brand green
 * (send button, primary buttons, etc.). It previously hardcoded a slightly
 * different `#14b789`, which looked inconsistent next to primary-colored UI
 * such as the artifact preview/code panel.
 */
export function avatarBgStyle(): { backgroundColor: string } {
  return { backgroundColor: "var(--primary)" };
}
