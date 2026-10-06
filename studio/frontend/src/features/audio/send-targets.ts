// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Free of app imports so the node test runner can load it directly.

import {
  EDIT_SOURCE_MAX_SECONDS,
  EDIT_SOURCE_SLACK_SECONDS,
} from "./audio-run-request";
import type { SendTarget } from "./components/stem-mixer-types";
import type { TranscriptSource } from "./transcript-model";
import { type AudioWorkflowId, clipWorkflow } from "./workflows";

const SPEECH_WORKFLOWS: ReadonlySet<string> = new Set([
  "speak",
  "clone",
  "edit",
  "convert",
]);

// Edit refuses a longer recording rather than cutting it.
function fitsEdit(durationS: number | null | undefined): boolean {
  return (
    durationS == null ||
    durationS <= EDIT_SOURCE_MAX_SECONDS + EDIT_SOURCE_SLACK_SECONDS
  );
}

/** Edit takes speech only, and its own results again; the other pages take any clip. */
export function clipSendTargets(
  clip: {
    workflow?: string | null;
    audio_type?: string | null;
    duration_s: number | null;
  },
  current: AudioWorkflowId,
): AudioWorkflowId[] {
  const targets: AudioWorkflowId[] = [
    "clone",
    "convert",
    "separate",
    "transcribe",
  ];
  if (SPEECH_WORKFLOWS.has(clipWorkflow(clip)) && fitsEdit(clip.duration_s)) {
    targets.push("edit");
  }
  return targets.filter((id) => id !== current || id === "edit");
}

export const STEM_SEND_TARGETS: readonly SendTarget[] = [
  { id: "clone", workflow: "clone", label: "Clone (as reference)" },
  { id: "convert", workflow: "convert", label: "Convert" },
  { id: "music", workflow: "music", label: "Music edit" },
  { id: "transcribe", workflow: "transcribe", label: "Transcribe" },
  { id: "voice", workflow: "voice", label: "Save as voice…" },
];

/** The text goes to Speak; Edit and Clone also need the audio it came from. */
export function transcriptSendTargets({
  text,
  source,
  duration,
}: {
  text: string;
  source: TranscriptSource | null | undefined;
  duration: number | null | undefined;
}): AudioWorkflowId[] {
  if (!text.trim()) return [];
  const targets: AudioWorkflowId[] = ["speak"];
  if (!source) return targets;
  // Edit cannot take a saved voice as its recording.
  if (source.kind !== "voice" && fitsEdit(duration)) targets.push("edit");
  targets.push("clone");
  return targets;
}
