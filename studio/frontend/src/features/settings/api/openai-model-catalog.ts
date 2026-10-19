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

/** The loaded model that can speak text, for read aloud: a TTS model in memory whose workflows
 *  include "speak" (music, transcription and convert-only models are skipped). With a saved voice,
 *  prefer one that can also clone. Null when none is loaded. */
export function loadedSpeechModelId(
  models: AudioApiModel[],
  wantsClone = false,
): string | null {
  const speakers = models.filter(
    (m) =>
      m.loaded && m.task === "text-to-speech" && m.workflows?.includes("speak"),
  );
  const cloner = wantsClone
    ? speakers.find((m) => m.workflows?.includes("clone"))
    : undefined;
  return (cloner ?? speakers[0])?.id ?? null;
}

const AUDIO_CATALOG_RETRY_MS = 15000;
const AUDIO_CATALOG_IDLE_MS = 60000;

// Polls because downloads, loads and idle unloads move no store state; slower once one is resident.
export function watchAudioModels(
  list: () => Promise<AudioApiModel[]>,
  onModels: (models: AudioApiModel[]) => void,
): () => void {
  let stopped = false;
  let timer: ReturnType<typeof setTimeout> | null = null;
  const update = () => {
    list()
      .then(
        (models) => {
          if (!stopped) onModels(models);
          return models.some((m) => m.loaded);
        },
        () => false,
      )
      .then((resident) => {
        if (stopped) return;
        timer = setTimeout(
          update,
          resident ? AUDIO_CATALOG_IDLE_MS : AUDIO_CATALOG_RETRY_MS,
        );
      });
  };
  update();
  return () => {
    stopped = true;
    if (timer !== null) clearTimeout(timer);
  };
}
