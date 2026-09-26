// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

const ROLE_ORDER: Record<string, number> = { system: 0, user: 1, assistant: 2 };

/**
 * The displayed branch, rebuilt as the history adapter does: sort by (createdAt, role, id),
 * parent legacy records to the previous one, then walk the last record's ancestor chain. Greedy
 * newest-child descent would pick another branch and drop pre-parentId history.
 */
export function orderBySelectedBranch<
  T extends {
    id: string;
    createdAt: number | Date;
    role: string;
    parentId?: string | null;
  },
>(messages: readonly T[]): T[] {
  const toTime = (v: number | Date): number =>
    typeof v === "number" ? v : v.getTime();

  const sorted = messages.slice().sort((a, b) => {
    const timeA = toTime(a.createdAt);
    const timeB = toTime(b.createdAt);
    if (timeA !== timeB) return timeA - timeB;
    const aOrder = ROLE_ORDER[a.role] ?? 99;
    const bOrder = ROLE_ORDER[b.role] ?? 99;
    if (aOrder !== bOrder) return aOrder - bOrder;
    return a.id < b.id ? -1 : a.id > b.id ? 1 : 0;
  });

  const byId = new Map<string, T>();
  const parentOf = new Map<string, string | null>();
  let previousId: string | null = null;
  for (const m of sorted) {
    byId.set(m.id, m);
    parentOf.set(m.id, m.parentId ?? previousId);
    previousId = m.id;
  }

  const chain: T[] = [];
  const seen = new Set<string>();
  let cur: string | null = sorted.at(-1)?.id ?? null;
  while (cur != null && !seen.has(cur)) {
    seen.add(cur);
    const record = byId.get(cur);
    if (!record) break;
    chain.push(record);
    cur = parentOf.get(cur) ?? null;
  }
  return chain.reverse();
}