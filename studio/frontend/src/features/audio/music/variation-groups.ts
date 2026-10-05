// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export interface GroupableClip {
  id: string;
  group_id?: string | null;
  settings?: Record<string, unknown> | null;
}

export type HistoryEntry<T extends GroupableClip> =
  | { kind: "clip"; clip: T }
  | { kind: "group"; groupId: string; clips: T[] };

function variationIndex(clip: GroupableClip): number | null {
  const value = clip.settings?.variation;
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function orderGroup<T extends GroupableClip>(clips: readonly T[]): T[] {
  return clips
    .map((clip, at) => ({ clip, at, index: variationIndex(clip) }))
    .sort((a, b) =>
      a.index !== null && b.index !== null && a.index !== b.index
        ? a.index - b.index
        : a.at - b.at,
    )
    .map((item) => item.clip);
}

function groupsOf<T extends GroupableClip>(
  clips: readonly T[],
): Map<string, T[]> {
  const groups = new Map<string, T[]>();
  for (const clip of clips) {
    if (!clip.group_id) continue;
    const members = groups.get(clip.group_id);
    if (members) members.push(clip);
    else groups.set(clip.group_id, [clip]);
  }
  for (const [id, members] of groups) {
    if (members.length < 2) groups.delete(id);
    else groups.set(id, orderGroup(members));
  }
  return groups;
}

export function groupHistory<T extends GroupableClip>(
  clips: readonly T[],
): HistoryEntry<T>[] {
  const groups = groupsOf(clips);
  const placed = new Set<string>();
  const entries: HistoryEntry<T>[] = [];
  for (const clip of clips) {
    const members = clip.group_id ? groups.get(clip.group_id) : undefined;
    if (!(members && clip.group_id)) {
      entries.push({ kind: "clip", clip });
      continue;
    }
    if (placed.has(clip.group_id)) continue;
    placed.add(clip.group_id);
    entries.push({ kind: "group", groupId: clip.group_id, clips: members });
  }
  return entries;
}

export function variationLabel(count: number): string {
  return count === 1 ? "1 variation" : `${count} variations`;
}

export function variationSiblings<T extends GroupableClip>(
  clips: readonly T[],
  selectedId: string | null,
): T[] {
  if (!selectedId) return [];
  const selected = clips.find((clip) => clip.id === selectedId);
  if (!selected?.group_id) return [];
  const members = clips.filter((clip) => clip.group_id === selected.group_id);
  return members.length > 1 ? orderGroup(members) : [];
}

export type HistoryRow<T extends GroupableClip> =
  | { clip: T; header?: undefined; nested: boolean }
  | {
      clip: T;
      header: { groupId: string; clips: T[]; open: boolean };
      nested: false;
    };

export function historyRows<T extends GroupableClip>(
  clips: readonly T[],
  openGroups: ReadonlySet<string>,
): HistoryRow<T>[] {
  const rows: HistoryRow<T>[] = [];
  for (const entry of groupHistory(clips)) {
    if (entry.kind === "clip") {
      rows.push({ clip: entry.clip, nested: false });
      continue;
    }
    const open = openGroups.has(entry.groupId);
    rows.push({
      clip: entry.clips[0],
      header: { groupId: entry.groupId, clips: entry.clips, open },
      nested: false,
    });
    if (open) for (const clip of entry.clips) rows.push({ clip, nested: true });
  }
  return rows;
}
