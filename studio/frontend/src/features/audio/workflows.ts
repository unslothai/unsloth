// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Free of app imports so the node test runner and the sidebar can load it directly.

import {
  AiVoiceIcon,
  MusicNote03Icon,
  SpeechToTextIcon,
  UserSwitchIcon,
  VoiceIdIcon,
} from "@hugeicons/core-free-icons";
import type { IconSvgElement } from "@hugeicons/react";
import { AUDIO_CPP_MUSIC_AUDIO_TYPE } from "./audio-cpp-catalog";

export type AudioWorkflowId =
  | "speak"
  | "clone"
  | "convert"
  | "music"
  | "transcribe";

/** Which model slot a workflow runs in: Speak, Clone, Convert and Music share the main inference slot, Transcribe uses the STT sidecar. */
export type AudioWorkflowSlot = "speak" | "transcribe";

/** The Audio workflows, shared by the page, the sidebar list and the More-flyout submenu. */
export const AUDIO_WORKFLOWS: ReadonlyArray<{
  id: AudioWorkflowId;
  label: string;
  heading: string;
  icon: IconSvgElement;
  hint: string;
  slot: AudioWorkflowSlot;
  /** Whether the header offers the Create|Train pill. */
  createTrain: boolean;
}> = [
  {
    id: "speak",
    label: "Speak",
    heading: "Text to speech",
    icon: AiVoiceIcon,
    hint: "Turn text into speech with a built-in or designed voice",
    slot: "speak",
    createTrain: true,
  },
  {
    id: "clone",
    label: "Clone",
    heading: "Clone a voice",
    icon: VoiceIdIcon,
    hint: "Speak in the voice from a short recording",
    slot: "speak",
    createTrain: true,
  },
  {
    id: "convert",
    label: "Convert",
    heading: "Convert voice",
    icon: UserSwitchIcon,
    hint: "Make a recording sound like another voice",
    slot: "speak",
    createTrain: false,
  },
  {
    id: "music",
    label: "Music",
    heading: "Create music",
    icon: MusicNote03Icon,
    hint: "Songs, instrumentals and sound effects",
    slot: "speak",
    createTrain: false,
  },
  {
    id: "transcribe",
    label: "Transcribe",
    heading: "Transcribe",
    icon: SpeechToTextIcon,
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

/** The `?task=` pipeline tags other pages hand to Audio. */
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

/** Which generation workflow made a clip of this audio type. */
export function audioWorkflowForAudioType(
  audioType: string | null | undefined,
): "speak" | "music" {
  return audioType && MUSIC_AUDIO_TYPES.has(audioType) ? "music" : "speak";
}

/** A gallery clip's workflow. Clips saved before the field existed fall back to their audio type. */
export function clipWorkflow(clip: {
  workflow?: string | null;
  audio_type?: string | null;
}): "speak" | "clone" | "convert" | "music" {
  if (
    clip.workflow === "speak" ||
    clip.workflow === "clone" ||
    clip.workflow === "convert" ||
    clip.workflow === "music"
  ) {
    return clip.workflow;
  }
  return audioWorkflowForAudioType(clip.audio_type);
}

/** The page a just-loaded main-slot model opens: the current page when the model can run it, else
 *  the first page the backend lists for it, else Music or Speak by its audio type. */
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

/** Whether the loaded main-slot model can run this page. Older backends send no list, so the
 *  audio type decides between Music and Speak, and nothing older clones. */
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
