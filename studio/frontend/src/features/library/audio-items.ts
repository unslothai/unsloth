// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Free of app imports so the node test runner can load it directly.

import { formatSeconds } from "../audio/components/waveform-peaks";
import { orderStems, stemLabel } from "../audio/separation-stems";
import { AUDIO_WORKFLOWS } from "../audio/workflows";
import type { LibraryItem } from "./api";

export type LibraryAudioWorkflow = (typeof AUDIO_WORKFLOWS)[number];

/** The Audio page workflow that made a clip, or null for any other audio file. */
export function audioWorkflow(item: LibraryItem): LibraryAudioWorkflow | null {
  const id = item.audio?.workflow;
  return AUDIO_WORKFLOWS.find((workflow) => workflow.id === id) ?? null;
}

const MUSIC_MODES: Record<string, string> = { sfx: "Sound effect", edit: "Edit" };

/** What tells a clip apart from the rest of its run: its stem, or the kind of music and its take. */
export function audioDetail(item: LibraryItem): string | null {
  const audio = item.audio;
  if (!audio) return null;
  if (audio.workflow === "separate") return audio.role ? stemLabel(audio.role) : null;
  if (audio.workflow !== "music") return null;
  const kind = MUSIC_MODES[audio.role === "edit" ? "edit" : (audio.mode ?? "")] ?? "Song";
  return audio.variation ? `${kind} ${audio.variation}` : kind;
}

/** A clip's length and what it is ("0:42", "Vocals"), for card and list captions. */
export function audioSummary(item: LibraryItem): string[] {
  const audio = item.audio;
  if (!audio) return [];
  const what = audioDetail(item) ?? audioWorkflow(item)?.label;
  return [audio.durationS ? formatSeconds(audio.durationS) : null, what ?? null].filter(
    (part): part is string => part !== null,
  );
}

/** The clips made with this one, in the order the Audio page shows them; empty outside a run. */
export function runSiblings(items: readonly LibraryItem[], item: LibraryItem): LibraryItem[] {
  const groupId = item.audio?.groupId;
  if (!groupId) return [];
  const run = items.filter((other) => other.audio?.groupId === groupId);
  const stems = orderStems(run.map((other) => other.audio?.role ?? ""));
  const rank = (other: LibraryItem) =>
    other.audio?.workflow === "separate"
      ? stems.indexOf(other.audio.role ?? "")
      : (other.audio?.variation ?? 0);
  return run.sort((a, b) => rank(a) - rank(b));
}

/** The workflows among these items, in Audio page order, for the Library's Workflow filter. */
export function audioWorkflowOptions(items: readonly LibraryItem[]): LibraryAudioWorkflow[] {
  const present = new Set(items.map((item) => item.audio?.workflow));
  return AUDIO_WORKFLOWS.filter((workflow) => present.has(workflow.id));
}
