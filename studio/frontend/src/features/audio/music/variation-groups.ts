// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Clips one run made together (variations share a `group_id`) shown as one history row. Free of
// app imports so the node test runner can load it directly.

export interface GroupableClip {
  id: string;
  group_id?: string | null;
  settings?: Record<string, unknown> | null;
}

export type HistoryEntry<T extends GroupableClip> =
  | { kind: "clip"; clip: T }
  | { kind: "group"; groupId: string; clips: T[] };

/** The variation number a run recorded in the clip's settings, if any. */
function variationIndex(clip: GroupableClip): number | null {
  const value = clip.settings?.variation;
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

/** A group's clips in the order the run made them: by recorded variation number, else as listed. */
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

/** Every listed clip's group, keyed by id, for groups with more than one clip. */
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

/** History rows: a group of variations is one row where its first clip was listed; every other
 *  clip stays a row of its own, in the order given. */
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

/** "3 variations". */
export function variationLabel(count: number): string {
  return count === 1 ? "1 variation" : `${count} variations`;
}

/** The selected clip's group in run order, or [] when it is not one of several. */
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

/** One history row as drawn: a clip (nested under its group when the group is open), or a
 *  group's header carrying its clips. */
export type HistoryRow<T extends GroupableClip> =
  | { clip: T; header?: undefined; nested: boolean }
  | {
      clip: T;
      header: { groupId: string; clips: T[]; open: boolean };
      nested: false;
    };

/** The rows to draw, with open groups' clips listed under their header. */
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
