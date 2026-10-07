// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Free of app imports so the node test runner can load it directly.

import {
  AUDIO_CPP_AUDIO_TYPES,
  audioCppModelFor,
  isCloneOnlyFamilyId,
} from "./audio-cpp-catalog";
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
  /** The pick is a GGUF: an id without a GGUF suffix otherwise reads as a Transformers checkpoint. */
  gguf?: boolean;
  item?: string;
  workflow?: AudioWorkflowId;
}

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
    ...(search.gguf === true || search.gguf === "true" ? { gguf: true } : {}),
    ...(typeof search.item === "string" ? { item: search.item } : {}),
    ...(isAudioWorkflowId(search.workflow)
      ? { workflow: search.workflow }
      : {}),
  };
}

export function audioRouteIntent(search: {
  task?: string | null;
  workflow?: string | null;
}): AudioWorkflowId | null {
  if (isAudioWorkflowId(search.workflow)) {
    return search.workflow;
  }
  return audioWorkflowForTask(search.task);
}

export function audioWorkflowForPick(pick: {
  id: string;
  task?: string | null;
  audioType?: string | null;
}): AudioWorkflowId | null {
  // Only a separation row's audio-to-audio pick is routed here.
  if (pick.task === "audio-to-audio") return "separate";
  const workflow = audioWorkflowForTask(pick.task);
  if (workflow !== null && workflow !== "speak") {
    return workflow;
  }
  // No task (or a speech one): the audio type or catalog entry can still name the page.
  const catalog = audioCppModelFor(pick.id);
  if (MUSIC_AUDIO_TYPES.has(pick.audioType ?? "") || catalog?.task === "music") {
    return "music";
  }
  if (catalog?.workflows) {
    return catalog.workflows.includes("speak")
      ? workflow
      : (catalog.workflows[0] ?? workflow);
  }
  return workflow === "speak" && isCloneOnlyFamilyId(pick.id) ? "clone" : workflow;
}

/** The /audio search for a model picked elsewhere (the chat picker, the Hub), opening the page
 *  that runs it. */
export function audioPickSearch(
  id: string,
  pick: {
    ggufFilename?: string | null;
    ggufVariant?: string | null;
    task?: string | null;
    audioType?: string | null;
    loadId?: string | null;
    isGguf?: boolean | null;
  },
): AudioRouteSearch {
  return {
    model: id,
    // `quant` is used verbatim as the gguf filename, so a label like "Q4_K_M" rides ggufQuant; both
    // go along, since the dictation sidecar picks its quant by label alone.
    quant: pick.ggufFilename ?? undefined,
    ggufQuant: pick.ggufVariant ?? undefined,
    task: pick.task ?? undefined,
    audioType: pick.audioType ?? undefined,
    loadId: pick.loadId ?? undefined,
    gguf: pick.isGguf ? true : undefined,
    workflow:
      audioWorkflowForPick({
        id,
        task: pick.task,
        audioType: pick.audioType,
      }) ?? undefined,
  };
}
