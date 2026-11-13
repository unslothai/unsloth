// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Free of app imports so the node test runner can load it directly.

import type { AudioGalleryClip } from "./api";

/** Display order. Mirrors _STEM_ORDER in studio/backend/routes/inference.py. */
export const STEM_ORDER: readonly string[] = [
  "vocals",
  "drums",
  "bass",
  "guitar",
  "piano",
  "other",
  "instrumental",
];

const STEM_LABELS: Record<string, string> = {
  vocals: "Vocals",
  drums: "Drums",
  bass: "Bass",
  guitar: "Guitar",
  piano: "Piano",
  other: "Other",
  instrumental: "Instrumental",
};

export function stemLabel(id: string): string {
  const known = STEM_LABELS[id];
  if (known) return known;
  const words = id.replace(/[_-]+/g, " ").trim();
  return words ? words[0].toUpperCase() + words.slice(1) : "Stem";
}

export const SEPARATION_STEMS_BY_FAMILY: Readonly<
  Record<string, { stems: readonly string[] }>
> = {
  htdemucs: { stems: ["vocals", "drums", "bass", "other"] },
  htdemucs_6stems: {
    stems: ["vocals", "drums", "bass", "guitar", "piano", "other"],
  },
  bs_roformer: { stems: ["vocals", "instrumental"] },
  mel_band_roformer: { stems: ["vocals", "instrumental"] },
};

export function orderStems<T extends string>(ids: readonly T[]): T[] {
  const rank = (id: string) => {
    const index = STEM_ORDER.indexOf(id);
    return index === -1 ? STEM_ORDER.length : index;
  };
  return ids
    .map((id, index) => ({ id, index }))
    .sort((a, b) => rank(a.id) - rank(b.id) || a.index - b.index)
    .map(({ id }) => id);
}

export interface SeparationGroup {
  groupId: string;
  title: string;
  model: string;
  durationS: number;
  createdAt: string;
  stems: AudioGalleryClip[];
  complete: boolean;
  expectedStems: number;
  pinned: boolean;
}

function settingsStems(clip: AudioGalleryClip): string[] | null {
  const stems = clip.settings?.stems;
  return Array.isArray(stems) && stems.every((s) => typeof s === "string")
    ? (stems as string[])
    : null;
}

export function groupSeparationClips(
  clips: readonly AudioGalleryClip[],
  hasMore = false,
): SeparationGroup[] {
  const byId = new Map<string, AudioGalleryClip[]>();
  const order: string[] = [];
  for (const clip of clips) {
    const key = clip.group_id || `clip:${clip.id}`;
    let list = byId.get(key);
    if (!list) {
      list = [];
      byId.set(key, list);
      order.push(key);
    }
    list.push(clip);
  }
  const groups = order.map((key) => {
    const list = byId.get(key) ?? [];
    const roleOf = (clip: AudioGalleryClip) => clip.role || clip.id;
    const ordered = orderStems(list.map(roleOf)).map(
      (role) => list.find((clip) => roleOf(clip) === role) as AudioGalleryClip,
    );
    const first = ordered[0];
    const expected = Math.max(
      ordered.length,
      ...ordered.map((clip) => settingsStems(clip)?.length ?? 0),
    );
    return {
      groupId: key,
      title: first?.prompt || "Separated track",
      model: first?.model ?? "",
      durationS: Math.max(0, ...ordered.map((clip) => clip.duration_s || 0)),
      createdAt: first?.created_at ?? "",
      stems: ordered,
      complete: ordered.length >= expected,
      expectedStems: expected,
      pinned: ordered.some((clip) => clip.pinned),
    };
  });
  // The oldest group may continue on the next page: hide it until loaded.
  const tail = groups[groups.length - 1];
  return hasMore && tail && !tail.complete ? groups.slice(0, -1) : groups;
}
