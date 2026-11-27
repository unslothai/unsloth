// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { isTitle } from "../prompt/speakers";

const HUES = [12, 28, 160, 190, 215, 250, 280, 330];

/** A stable hue per name, so an avatar-less character keeps its color everywhere. */
export function nameHue(name: string): number {
  let hash = 0;
  for (const char of name) hash = (hash * 31 + char.codePointAt(0)!) >>> 0;
  return HUES[hash % HUES.length];
}

export function initials(name: string): string {
  // "Inspector Theo Lindqvist" is TL, not IL.
  const all = name.trim().split(/\s+/).filter(Boolean);
  const start = all.findIndex((word) => !isTitle(word));
  const words = start > 0 ? all.slice(start) : all;
  if (words.length === 0) return "?";
  const first = [...words[0]][0] ?? "";
  const second = words.length > 1 ? ([...words[words.length - 1]][0] ?? "") : "";
  return (first + second).toUpperCase();
}
