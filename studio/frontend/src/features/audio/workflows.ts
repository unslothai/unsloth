// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Free of app imports so the node test runner and the sidebar can load it directly.

import {
  ArrowReloadHorizontalIcon,
  ClosedCaptionIcon,
  Edit03Icon,
  MusicThreeIcon,
  SpeechIcon,
  SplitIcon,
  UserMultipleIcon,
} from "@hugeicons/core-free-icons";
import type { IconSvgElement } from "@hugeicons/react";
import { AUDIO_CPP_MUSIC_AUDIO_TYPE } from "./audio-cpp-catalog.ts";

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
    label: "Speak",
    heading: "Text to speech",
    icon: SpeechIcon,
    hint: "Turn text into speech with a built-in or designed voice",
    slot: "speak",
    createTrain: true,
  },
  {
    id: "clone",
    label: "Clone",
    heading: "Clone a voice",
    icon: UserMultipleIcon,
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
