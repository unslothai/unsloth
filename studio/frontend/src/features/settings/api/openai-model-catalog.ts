// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export type OpenAIModel = {
  id: string;
  // Resident in memory now; the rest are downloaded and servable.
  loaded?: boolean;
  // On-disk GGUF quant. Ids stay bare for OpenAI compat, so append `:quant` to pin it.
  quant?: string;
};

type ApiOpenAIModelList = {
  data?: {
    id?: unknown;
    loaded?: unknown;
    quant?: unknown;
    task?: unknown;
    audio_workflows?: unknown;
  }[];
};

const CHAT_TASKS = new Set(["text-generation"]);

export type AudioApiTask =
  | "text-to-speech"
  | "automatic-speech-recognition"
  | "audio-to-audio";

export type AudioApiModel = {
  id: string;
  loaded: boolean;
  // text-to-speech also covers music models, which /v1/audio/speech serves too.
  task: AudioApiTask;
  // The Audio page workflows the server says this row runs, when it says.
  workflows?: string[];
};

const AUDIO_TASKS = new Set<string>([
  "text-to-speech",
  "automatic-speech-recognition",
  "audio-to-audio",
]);

export function chatModelsFromCatalog(body: unknown): OpenAIModel[] {
  const data = (body as ApiOpenAIModelList | null)?.data;
  if (!Array.isArray(data)) {
    return [];
  }
  return data.flatMap((entry) =>
    typeof entry?.id === "string" &&
    entry.id &&
    (typeof entry.task !== "string" || CHAT_TASKS.has(entry.task))
      ? [
          {
            id: entry.id,
            loaded: entry.loaded === true,
            quant: typeof entry.quant === "string" ? entry.quant : undefined,
          },
        ]
      : [],
  );
}

export function audioModelsFromCatalog(body: unknown): AudioApiModel[] {
  const data = (body as ApiOpenAIModelList | null)?.data;
  if (!Array.isArray(data)) {
    return [];
  }
  return data.flatMap((entry) =>
    typeof entry?.id === "string" &&
    entry.id &&
    typeof entry.task === "string" &&
    AUDIO_TASKS.has(entry.task)
      ? [
          {
            id: entry.id,
            loaded: entry.loaded === true,
            task: entry.task as AudioApiTask,
            ...(Array.isArray(entry.audio_workflows)
              ? {
                  workflows: entry.audio_workflows.filter(
                    (item): item is string => typeof item === "string",
                  ),
                }
              : {}),
          },
        ]
      : [],
  );
}
