// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export function agentCacheLoadIds(
  copies: readonly {
    repo_id: string;
    load_id?: string | null;
    active_cache?: boolean | null;
    partial?: boolean;
  }[],
): Record<string, string> {
  const targets: Record<string, string> = {};
  const seen = new Set<string>();
  const ordered = [...copies].sort(
    (a, b) =>
      Number(a.partial === true) - Number(b.partial === true) ||
      Number(b.active_cache === true) - Number(a.active_cache === true) ||
      (a.load_id || a.repo_id).localeCompare(b.load_id || b.repo_id),
  );
  for (const copy of ordered) {
    const key = copy.repo_id.toLowerCase();
    if (seen.has(key)) continue;
    seen.add(key);
    if (copy.load_id && copy.load_id !== copy.repo_id) {
      targets[key] = copy.load_id;
    }
  }
  return targets;
}
