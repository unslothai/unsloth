// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * The folder holding `path`, where the folder picker starts. A drive or share root keeps its
 * separator: `D:` alone is the current folder on drive D, not its root, and `\\server` is no
 * folder at all. Undefined when there is nothing above `path` to start from.
 */
export function parentFolder(path: string): string | undefined {
  // `/` and `D:\` are their own roots; anything else loses a trailing separator.
  const trimmed = /^([A-Za-z]:)?[\\/]$/.test(path) ? path : path.replace(/[\\/]+$/, "");
  const cut = Math.max(trimmed.lastIndexOf("/"), trimmed.lastIndexOf("\\"));
  if (cut < 0 || cut === trimmed.length - 1) return undefined;
  const separator = trimmed[cut]!;
  const parent = trimmed.slice(0, cut);
  if (parent === "") return separator === "/" ? "/" : undefined;
  if (/^[A-Za-z]:$/.test(parent)) return parent + separator;
  // `\\server\share\x`: the share root, with its separator. `\\server\share` has none above it.
  if (/^[\\/]{2}[^\\/]+$/.test(parent)) return undefined;
  if (/^[\\/]{2}[^\\/]+[\\/][^\\/]+$/.test(parent)) return parent + separator;
  return parent;
}
