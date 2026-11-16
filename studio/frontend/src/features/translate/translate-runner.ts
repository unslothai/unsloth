// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  getInferenceStatus,
  listGgufVariants,
  loadModel,
  streamChatCompletions,
} from "@/features/chat/api/chat-api";
import { prepareHfTokenForUse } from "@/features/hf-auth";
import { DOWNLOAD_KIND } from "@/features/hub/download-manager/constants";
import {
  downloadManager,
  jobKeyOf,
  selectActiveJob,
  subscribeJobListeners,
  useDownloadManagerStore,
} from "@/features/hub/download-manager/download-manager-controller";
import { modelIdsMatch } from "@/features/hub/lib/model-identity";
import { useHfTokenStore } from "@/features/hub/stores/hf-token-store";
import {
  CHAT_TRANSLATE_MODEL,
  type DedicatedTranslateModel,
  GENERAL_SAMPLING,
  TRANSLATE_CONTEXT_TOKENS,
  TRANSLATE_MAX_TOKENS,
  type TranslateModelId,
  dedicatedTranslateModel,
  translateMessages,
} from "./models";
import {
  type TranslateTone,
  restoreChunkWhitespace,
  splitForTranslation,
} from "./translate-text";

export type TranslateRunErrorCode = "noChatModel" | "downloadFailed";

/** A failure the page words itself; anything else shows the server's message. */
export class TranslateRunError extends Error {
  readonly code: TranslateRunErrorCode;
  constructor(code: TranslateRunErrorCode) {
    super(code);
    this.code = code;
  }
}

export type TranslatePhase =
  | { kind: "preparing" }
  | { kind: "downloading"; downloadedBytes: number; totalBytes: number }
  | { kind: "loading" }
  | { kind: "translating"; part: number; parts: number };

export type TranslateRunOptions = {
  text: string;
  targetName: string;
  tone: TranslateTone;
  modelId: TranslateModelId;
  signal: AbortSignal;
  onPhase: (phase: TranslatePhase) => void;
  /** The whole translation so far. */
  onOutput: (output: string) => void;
};

function deltaText(content: unknown): string {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  return content
    .map((part) =>
      typeof part === "string"
        ? part
        : part &&
            typeof part === "object" &&
            (part as { type?: unknown }).type === "text"
          ? String((part as { text?: unknown }).text ?? "")
          : "",
    )
    .join("");
}

function isServing(
  status: Awaited<ReturnType<typeof getInferenceStatus>>,
  repo: string,
): boolean {
  if (status.loading.some((id) => modelIdsMatch(id, repo))) return false;
  return [
    status.active_model,
    ...(status.serving ?? []),
    ...(status.serving_checkpoints ?? []),
  ].some((id) => modelIdsMatch(id, repo));
}

/** Whether the model's file is already on disk; null when the listing could not be read. */
export async function translateModelDownloaded(
  model: DedicatedTranslateModel,
  signal?: AbortSignal,
): Promise<boolean | null> {
  try {
    const listing = await listGgufVariants(model.repo, undefined, { signal });
    const variant = listing.variants.find(
      (entry) => entry.quant?.toLowerCase() === model.variant.toLowerCase(),
    );
    return variant
      ? variant.downloaded === true && variant.partial !== true
      : null;
  } catch {
    return null;
  }
}

/** Runs the transfer as a download-manager job, so it also shows in the Hub's download panel. */
async function downloadModel(
  model: DedicatedTranslateModel,
  hfToken: string | null,
  signal: AbortSignal,
  onPhase: (phase: TranslatePhase) => void,
): Promise<void> {
  const prepared = await prepareHfTokenForUse(hfToken);
  signal.throwIfAborted();
  if (!prepared.proceed) throw new DOMException("Aborted", "AbortError");

  const kind = DOWNLOAD_KIND.MODEL;
  const jobKey = jobKeyOf(kind, model.repo, model.variant);
  const ours = (variant: string | null) =>
    (variant ?? "").toLowerCase() === model.variant.toLowerCase();
  onPhase({
    kind: "downloading",
    downloadedBytes: 0,
    totalBytes: model.sizeBytes,
  });

  await new Promise<void>((resolve, reject) => {
    const cleanups: Array<() => void> = [];
    let settled = false;
    const finish = (error?: unknown) => {
      if (settled) return;
      settled = true;
      for (const cleanup of cleanups) cleanup();
      if (error === undefined) resolve();
      else reject(error);
    };
    const cancelJob = () => {
      const active = selectActiveJob(
        useDownloadManagerStore.getState(),
        kind,
        model.repo,
        model.variant,
      );
      if (active && active.state !== "cancelling")
        void downloadManager.cancel(active.key);
    };
    // Before the start request, so a fast completion cannot be missed.
    cleanups.push(
      subscribeJobListeners(kind, model.repo, {
        onComplete: (variant) => ours(variant) && finish(),
        onCancelled: (variant) =>
          ours(variant) && finish(new DOMException("Aborted", "AbortError")),
        onError: (variant) =>
          ours(variant) && finish(new TranslateRunError("downloadFailed")),
      }),
      useDownloadManagerStore.subscribe((state) => {
        const job = state.jobs[jobKey];
        if (!job) return;
        // Stopped before the job existed: cancel it as soon as it does.
        if (signal.aborted) {
          cancelJob();
          return;
        }
        onPhase({
          kind: "downloading",
          downloadedBytes: job.downloadedBytes,
          totalBytes: job.expectedBytes || model.sizeBytes,
        });
      }),
    );
    // Stop means stop: the user asked for a translation, not for a model in the background.
    const onAbort = () => {
      cancelJob();
      finish(signal.reason ?? new DOMException("Aborted", "AbortError"));
    };
    signal.addEventListener("abort", onAbort, { once: true });
    cleanups.push(() => signal.removeEventListener("abort", onAbort));

    void downloadManager
      .requestStart({
        kind,
        repoId: model.repo,
        variant: model.variant,
        expectedBytes: model.sizeBytes,
      })
      .then(
        (outcome) => {
          // "conflict" and "busy" are settled in the Hub's download card; "error" already toasted.
          if (outcome !== "started")
            finish(new TranslateRunError("downloadFailed"));
        },
        () => finish(new TranslateRunError("downloadFailed")),
      );
  });
}

/** Downloads the model if it is missing and loads it if it is not serving. Returns its request id. */
async function ensureDedicatedModel(
  model: DedicatedTranslateModel,
  signal: AbortSignal,
  onPhase: (phase: TranslatePhase) => void,
): Promise<string> {
  const status = await getInferenceStatus(signal);
  if (isServing(status, model.repo)) return model.repo;

  const hfToken = useHfTokenStore.getState().token || null;
  // An unreadable listing is left to the load, which fetches a missing file itself.
  if ((await translateModelDownloaded(model, signal)) === false) {
    signal.throwIfAborted();
    await downloadModel(model, hfToken, signal, onPhase);
  }
  signal.throwIfAborted();
  onPhase({ kind: "loading" });
  await loadModel(
    {
      model_path: model.repo,
      gguf_variant: model.variant,
      hf_token: hfToken,
      max_seq_length: TRANSLATE_CONTEXT_TOKENS,
      load_in_4bit: true,
      is_lora: false,
      // Beside the chat model under Keep multiple models loaded; the server ignores this when off.
      alongside: true,
    },
    { signal, runtime: "translate" },
  );
  return model.repo;
}

async function loadedChatModel(signal: AbortSignal): Promise<string> {
  const status = await getInferenceStatus(signal);
  if (!status.active_model || status.is_audio || status.is_diffusion) {
    throw new TranslateRunError("noChatModel");
  }
  return status.active_model;
}

export async function runTranslation(
  options: TranslateRunOptions,
): Promise<string> {
  const { text, targetName, tone, modelId, signal, onPhase, onOutput } =
    options;
  onPhase({ kind: "preparing" });
  const dedicated = dedicatedTranslateModel(modelId);
  const requestModel = dedicated
    ? await ensureDedicatedModel(dedicated, signal, onPhase)
    : await loadedChatModel(signal);
  const sampling = dedicated?.sampling ?? GENERAL_SAMPLING;

  const chunks = splitForTranslation(text);
  const parts = chunks.filter((chunk) => chunk.text).length;
  let output = "";
  let part = 0;
  for (const chunk of chunks) {
    signal.throwIfAborted();
    if (!chunk.text) {
      output += chunk.after;
      onOutput(output);
      continue;
    }
    part += 1;
    onPhase({ kind: "translating", part, parts });
    let translated = "";
    const stream = streamChatCompletions(
      {
        model: requestModel,
        messages: translateMessages(
          dedicated ? modelId : CHAT_TRANSLATE_MODEL,
          chunk.text,
          targetName,
          tone,
        ),
        stream: true,
        max_tokens: TRANSLATE_MAX_TOKENS,
        ...sampling,
        enable_tools: false,
        enable_thinking: false,
      },
      signal,
    );
    for await (const event of stream) {
      const delta = deltaText(event.choices?.[0]?.delta?.content);
      if (!delta) continue;
      translated += delta;
      onOutput(output + translated.trimStart());
    }
    output += restoreChunkWhitespace(chunk, translated);
    onOutput(output);
  }
  return output;
}
