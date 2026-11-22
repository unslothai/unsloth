// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export function initialsFromName(name: string): string {
  const trimmed = name.trim();
  if (!trimmed) return "?";
  return trimmed[0]!.toUpperCase();
}

/** Uses --control-accent so the avatar follows the palette and any custom accent. */
export function avatarBgStyle(): { backgroundColor: string; color: string } {
  return {
    backgroundColor: "var(--control-accent, #17b88b)",
    color: "var(--control-accent-foreground, #ffffff)",
  };
}
