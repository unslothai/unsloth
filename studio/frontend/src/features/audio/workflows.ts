// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Free of app imports so the node test runner and the sidebar can load it directly.

import {
  ArrowReloadHorizontalIcon,
  ClosedCaptionIcon,
  Copy02Icon,
  Edit03Icon,
  MusicThreeIcon,
  SplitIcon,
} from "@hugeicons/core-free-icons";
import type { IconSvgElement } from "@hugeicons/react";
import { AUDIO_CPP_MUSIC_AUDIO_TYPE } from "./audio-cpp-catalog.ts";

const stroke = {
  stroke: "currentColor",
  strokeLinecap: "round",
  strokeLinejoin: "round",
  strokeWidth: "1.5",
} as const;

// Hugeicons "ai-speech" (MIT). Only in core-free-icons 4.3+, newer than the pinned 4.1.1.
const AiSpeechIcon: IconSvgElement = [
  ["path", { d: "M12 7V17", ...stroke, key: "0" }],
  ["path", { d: "M16 11L16 19", ...stroke, key: "1" }],
  ["path", { d: "M20 11L20 14", ...stroke, key: "2" }],
  ["path", { d: "M8 3V21", ...stroke, key: "3" }],
  ["path", { d: "M4 9V15", ...stroke, key: "4" }],
  [
    "path",
    {
      d: "M18.5 3.9375V5.5M18.5 5.5V7.0625M18.5 5.5H17.25M18.5 5.5H19.75M21 5.5L19.9156 5.13852C19.4179 4.97263 19.0274 4.58211 18.8615 4.08443L18.5 3L18.1385 4.08443C17.9726 4.58211 17.5821 4.97263 17.0844 5.13852L16 5.5L17.0844 5.86148C17.5821 6.02737 17.9726 6.41789 18.1385 6.91557L18.5 8L18.8615 6.91557C19.0274 6.41789 19.4179 6.02737 19.9156 5.86148L21 5.5Z",
      ...stroke,
      key: "5",
    },
  ],
];

export type AudioWorkflowId =
  | "speak"
  | "clone"
  | "edit"
  | "convert"
  | "music"
  | "separate"
  | "transcribe";

export type AudioWorkflowSlot = "speak" | "transcribe";

export const AUDIO_WORKFLOWS: ReadonlyArray<{
  id: AudioWorkflowId;
  label: string;
  heading: string;
  icon: IconSvgElement;
  hint: string;
  slot: AudioWorkflowSlot;
  createTrain: boolean;
}> = [
  {
    id: "speak",
    label: "Text to Speech",
    heading: "Text to speech",
    icon: AiSpeechIcon,
    hint: "Turn text into speech with a built-in or designed voice",
    slot: "speak",
    createTrain: true,
  },
  {
    id: "clone",
    label: "Clone",
    heading: "Clone a voice",
    icon: Copy02Icon,
    hint: "Speak in the voice from a short recording",
    slot: "speak",
    createTrain: true,
  },
  {
    id: "edit",
    label: "Edit",
    heading: "Edit speech",
    icon: Edit03Icon,
    hint: "Change words in a recording, same voice",
    slot: "speak",
    createTrain: false,
  },
  {
    id: "convert",
    label: "Convert",
    heading: "Convert voice",
    icon: ArrowReloadHorizontalIcon,
    hint: "Make a recording sound like another voice",
    slot: "speak",
    createTrain: false,
  },
  {
    id: "music",
    label: "Music",
    heading: "Create music",
    icon: MusicThreeIcon,
    hint: "Songs, instrumentals and sound effects",
    slot: "speak",
    createTrain: false,
  },
  {
    id: "separate",
    label: "Separate",
    heading: "Separate audio",
    icon: SplitIcon,
    hint: "Split a track into vocals and instruments",
    slot: "speak",
    createTrain: false,
  },
  {
    id: "transcribe",
    label: "Transcribe",
    heading: "Transcribe",
    icon: ClosedCaptionIcon,
    hint: "Turn speech into text",
    slot: "transcribe",
    createTrain: true,
  },
];

/** Mirrors MUSIC_AUDIO_TYPES in studio/backend/core/inference/audio_workflows.py. */
export const MUSIC_AUDIO_TYPES: ReadonlySet<string> = new Set([
  "minimax_music3",
  AUDIO_CPP_MUSIC_AUDIO_TYPE,
]);

export function isAudioWorkflowId(value: unknown): value is AudioWorkflowId {
  return AUDIO_WORKFLOWS.some((tab) => tab.id === value);
}

export function audioWorkflowTab(id: AudioWorkflowId) {
  return AUDIO_WORKFLOWS.find((tab) => tab.id === id) ?? AUDIO_WORKFLOWS[0];
}

export function slotForWorkflow(id: AudioWorkflowId): AudioWorkflowSlot {
  return audioWorkflowTab(id).slot;
}

export function audioWorkflowForTask(
  task: string | null | undefined,
): AudioWorkflowId | null {
  switch (task) {
    case "text-to-speech":
      return "speak";
    case "automatic-speech-recognition":
      return "transcribe";
    case "text-to-audio":
      return "music";
    default:
      return null;
  }
}

export function audioWorkflowForAudioType(
  audioType: string | null | undefined,
): "speak" | "music" {
  return audioType && MUSIC_AUDIO_TYPES.has(audioType) ? "music" : "speak";
}

export function clipWorkflow(clip: {
  workflow?: string | null;
  audio_type?: string | null;
}): "speak" | "clone" | "edit" | "convert" | "music" | "separate" {
  if (
    clip.workflow === "speak" ||
    clip.workflow === "clone" ||
    clip.workflow === "edit" ||
    clip.workflow === "convert" ||
    clip.workflow === "music" ||
    clip.workflow === "separate"
  ) {
    return clip.workflow;
  }
  return audioWorkflowForAudioType(clip.audio_type);
}

export function workflowForLoadedModel({
  current,
  audioWorkflows,
  music,
}: {
  current: AudioWorkflowId;
  audioWorkflows: readonly string[] | null | undefined;
  music: boolean;
}): AudioWorkflowId {
  const runnable = (audioWorkflows ?? []).filter(
    (id): id is AudioWorkflowId =>
      isAudioWorkflowId(id) && slotForWorkflow(id) === "speak",
  );
  if (runnable.includes(current)) return current;
  if (runnable.length > 0) return runnable[0];
  return music ? "music" : "speak";
}

/** Older backends send no list: the audio type picks Music or Speak, and nothing clones. */
export function loadedModelRunsWorkflow({
  workflow,
  audioWorkflows,
  music,
}: {
  workflow: AudioWorkflowId;
  audioWorkflows: readonly string[] | null | undefined;
  music: boolean;
}): boolean {
  if (audioWorkflows && audioWorkflows.length > 0) {
    return audioWorkflows.includes(workflow);
  }
  return workflow === (music ? "music" : "speak");
}
