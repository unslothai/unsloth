// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export function initialsFromName(name: string): string {
  const trimmed = name.trim();
  if (!trimmed) return "?";
  return trimmed[0]!.toUpperCase();
}

/**
 * Accent background for the avatar fallback with a readable foreground.
 *
 * Uses `--control-accent`, the token behind toggles and badges, so the
 * avatar follows the palette accent (green in standard, blue in classic)
 * and any custom accent the user picks in Appearance. The literals only
 * apply outside the theme root, keeping the avatar branded there.
 */
export function avatarBgStyle(): { backgroundColor: string; color: string } {
  return {
    backgroundColor: "var(--control-accent, #17b88b)",
    color: "var(--control-accent-foreground, #ffffff)",
  };
}
