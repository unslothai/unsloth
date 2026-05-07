// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export function initialsFromName(name: string): string {
  const trimmed = name.trim();
  if (!trimmed) return "?";
  return trimmed[0]!.toUpperCase();
}

/** Default blue background for avatar fallback (readable white text). */
export function avatarBgStyle(): { backgroundColor: string } {
  return { backgroundColor: "hsl(217 58% 48%)" };
}
