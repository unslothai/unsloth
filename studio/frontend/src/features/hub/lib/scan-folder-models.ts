// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

function comparable(path: string): string {
  const slashed = path.replace(/\\/g, "/").replace(/\/+$/, "");
  // Drive-letter and UNC paths are Windows, which ignores case.
  return /^([A-Za-z]:|\/\/)/.test(slashed) ? slashed.toLowerCase() : slashed;
}

/** Listed models per registered folder, counted from rows the inventory already holds. A model
 *  under two nested folders counts for the innermost one only. */
export function countModelsPerFolder(
  folderPaths: readonly string[],
  modelPaths: readonly string[],
): Map<string, number> {
  const folders = folderPaths
    .map((path) => ({ path, key: comparable(path) }))
    .sort((a, b) => b.key.length - a.key.length);
  const counts = new Map<string, number>(folderPaths.map((path) => [path, 0]));
  for (const modelPath of modelPaths) {
    const key = comparable(modelPath);
    const owner = folders.find(
      (folder) => key === folder.key || key.startsWith(`${folder.key}/`),
    );
    if (owner) counts.set(owner.path, (counts.get(owner.path) ?? 0) + 1);
  }
  return counts;
}

export function folderDisplayName(path: string): string {
  const parts = path.split(/[\\/]/).filter(Boolean);
  return parts.at(-1) ?? path;
}
