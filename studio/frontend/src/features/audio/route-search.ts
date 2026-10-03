// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Free of app imports so the node test runner can load it directly.

import { AUDIO_CPP_AUDIO_TYPES, audioCppModelFor } from "./audio-cpp-catalog";
import {
  type AudioWorkflowId,
  MUSIC_AUDIO_TYPES,
  audioWorkflowForTask,
  isAudioWorkflowId,
} from "./workflows";

const NATIVE_AUDIO_TYPES = new Set([
  "higgs_tts2",
  "moss_tts_local",
  "moss_tts_nano",
  "higgs_tts3",
  "minimax_music3",
  ...AUDIO_CPP_AUDIO_TYPES,
]);

/** The pipeline tags other pages hand to Audio as `?task=`. */
const AUDIO_ROUTE_TASKS: ReadonlySet<string> = new Set([
  "text-to-speech",
  "automatic-speech-recognition",
  "text-to-audio",
]);

export interface AudioRouteSearch {
  model?: string;
  quant?: string;
  ggufQuant?: string;
  task?: string;
  audioType?: string;
  loadId?: string;
  item?: string;
  workflow?: AudioWorkflowId;
}

/** `/audio` search params. An audio pick made from the chat picker arrives as ?model= (+ ?quant=,
 *  ?ggufQuant=, task and workflow); Settings and the sidebar send a bare task or workflow. */
export function validateAudioSearch(
  search: Record<string, unknown>,
): AudioRouteSearch {
  return {
    ...(typeof search.model === "string" ? { model: search.model } : {}),
    ...(typeof search.quant === "string" ? { quant: search.quant } : {}),
    ...(typeof search.ggufQuant === "string"
      ? { ggufQuant: search.ggufQuant }
      : {}),
    ...(typeof search.task === "string" && AUDIO_ROUTE_TASKS.has(search.task)
      ? { task: search.task }
      : {}),
    ...(typeof search.audioType === "string" &&
    NATIVE_AUDIO_TYPES.has(search.audioType)
      ? { audioType: search.audioType }
      : {}),
    ...(typeof search.loadId === "string" && search.loadId.trim()
      ? { loadId: search.loadId }
      : {}),
    ...(typeof search.item === "string" ? { item: search.item } : {}),
    ...(isAudioWorkflowId(search.workflow)
      ? { workflow: search.workflow }
      : {}),
  };
}

/** The workflow a deep link asks for: an explicit ?workflow= wins over the one ?task= implies. */
export function audioRouteIntent(search: {
  task?: string | null;
  workflow?: string | null;
}): AudioWorkflowId | null {
  if (isAudioWorkflowId(search.workflow)) {
    return search.workflow;
  }
  return audioWorkflowForTask(search.task);
}

/** The workflow a chat-picker pick opens on. A speech-tagged music model (MiniMax Music 3, an
 *  audio.cpp music package) goes to Music, told apart by its audio type or catalog entry. */
export function audioWorkflowForPick(pick: {
  id: string;
  task?: string | null;
  audioType?: string | null;
}): AudioWorkflowId | null {
  const workflow = audioWorkflowForTask(pick.task);
  if (workflow !== "speak") {
    return workflow;
  }
  const music =
    MUSIC_AUDIO_TYPES.has(pick.audioType ?? "") ||
    audioCppModelFor(pick.id)?.task === "music";
  return music ? "music" : "speak";
}
