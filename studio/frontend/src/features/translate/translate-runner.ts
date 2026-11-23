// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  getInferenceStatus,
  listGgufVariants,
  loadModel,
  streamChatCompletions,
  unloadModel,
} from "@/features/chat/api/chat-api";
import {
  isExternalModelId,
  parseExternalModelId,
} from "@/features/chat/external-providers";
import { useChatRuntimeStore } from "@/features/chat/stores/chat-runtime-store";
import { buildTitleRequest } from "@/features/chat/utils/chat-title";
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
import { createNavigationNonce } from "@/lib/navigation-nonce";
import {
  isTranslationHeldSlot,
  isTranslationOnlyRepo,
} from "@/lib/translation-models";
import {
  CUSTOM_TRANSLATE_MODEL,
  type CustomTranslateModel,
  HY_MT2_SAMPLING,
  TRANSLATE_CONTEXT_TOKENS,
  TRANSLATE_MAX_TOKENS,
  type TranslateModelId,
  type TranslateSampling,
  dedicatedTranslateModel,
  translateMessages,
  translatePromptModel,
} from "./models";
import { quantOnDisk } from "./quant-on-disk";
import {
  parseSubtitles,
  translateSubtitles,
  translateTexts,
} from "./translate-subtitles";
import {
  type TranslatePromptOptions,
  restoreChunkWhitespace,
  splitForTranslation,
  stripLeadingReasoning,
} from "./translate-text";

export type TranslateRunErrorCode =
  | "noChatModel"
  | "downloadFailed"
  | "unavailable"
  | "cutOff"
  | "filtered";

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
  prompt: TranslatePromptOptions;
  modelId: TranslateModelId;
  custom: CustomTranslateModel | null;
  signal: AbortSignal;
  onPhase: (phase: TranslatePhase) => void;
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

type LocalModel = {
  repo: string;
  /** Null for a model with no quants to pick from; the load fetches whatever it needs. */
  variant: string | null;
  sizeBytes: number;
};

/** Disk first: a quant already on disk needs no Hub listing. `remote: false` never asks the Hub. */
export async function translateModelDownloaded(
  model: LocalModel,
  signal?: AbortSignal,
  options?: { remote?: boolean },
): Promise<boolean | null> {
  const quant = model.variant?.toLowerCase();
  if (!quant) return null;
  return quantOnDisk(
    quant,
    async (localOnly) =>
      (await listGgufVariants(model.repo, undefined, { signal, localOnly }))
        .variants,
    { remote: options?.remote !== false, signal },
  );
}

async function downloadModel(
  model: LocalModel & { variant: string },
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

  // A job still cancelling from an earlier Stop would be joined, and its cancellation end this run.
  await new Promise<void>((resolve, reject) => {
    const cancelling = () =>
      useDownloadManagerStore.getState().jobs[jobKey]?.state === "cancelling";
    if (!cancelling()) {
      resolve();
      return;
    }
    const unsubscribe = useDownloadManagerStore.subscribe(() => {
      if (cancelling()) return;
      unsubscribe();
      signal.removeEventListener("abort", onAbort);
      resolve();
    });
    const onAbort = () => {
      unsubscribe();
      reject(signal.reason ?? new DOMException("Aborted", "AbortError"));
    };
    signal.addEventListener("abort", onAbort, { once: true });
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
          if (outcome === "started" && signal.aborted) cancelJob();
          // "conflict" and "busy" are settled in the Hub's download card; "error" already toasted.
          if (outcome !== "started")
            finish(new TranslateRunError("downloadFailed"));
        },
        () => finish(new TranslateRunError("downloadFailed")),
      );
  });
}

async function ensureLocalModel(
  model: LocalModel,
  signal: AbortSignal,
  onPhase: (phase: TranslatePhase) => void,
): Promise<string> {
  // A bare repo id matches any loaded quant of it, so the picked quant is part of the id.
  const requestModel = model.variant
    ? `${model.repo}:${model.variant}`
    : model.repo;
  const status = await getInferenceStatus(signal, requestModel);
  if (
    isServing(status, model.repo) &&
    (!model.variant ||
      (modelIdsMatch(status.active_model, model.repo) &&
        status.gguf_variant?.toLowerCase() === model.variant.toLowerCase()))
  )
    return requestModel;

  const hfToken = useHfTokenStore.getState().token || null;
  // An unreadable listing is left to the load, which fetches a missing file itself.
  const { variant } = model;
  if (variant && (await translateModelDownloaded(model, signal)) === false) {
    signal.throwIfAborted();
    await downloadModel({ ...model, variant }, hfToken, signal, onPhase);
  }
  signal.throwIfAborted();
  onPhase({ kind: "loading" });
  // A dropped fetch does not stop the server's load; this cancel id does, and never unloads a resident model.
  const loadRequestId = createNavigationNonce();
  let sent = false;
  const cancelLoad = () => {
    if (!sent) return;
    void unloadModel({
      model_path: model.repo,
      cancel_load_request_id: loadRequestId,
    }).catch(() => {});
  };
  signal.addEventListener("abort", cancelLoad, { once: true });
  try {
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
        load_request_id: loadRequestId,
      },
      {
        signal,
        runtime: "translate",
        onRequestStart: () => {
          sent = true;
        },
      },
    );
  } finally {
    signal.removeEventListener("abort", cancelLoad);
  }
  return requestModel;
}

async function chatModel(signal: AbortSignal): Promise<string> {
  const picked = useChatRuntimeStore.getState().params.checkpoint;
  if (isExternalModelId(picked)) return picked;
  // The slot serving Chat's pick, which is not the primary under Keep multiple models loaded.
  const status = await getInferenceStatus(signal, picked || undefined);
  if (
    !status.active_model ||
    status.is_audio ||
    status.is_diffusion ||
    isTranslationHeldSlot(status.active_model, picked)
  ) {
    throw new TranslateRunError("noChatModel");
  }
  return status.active_model;
}

/** A provider's model by its own id, never the internal `external::...` one. */
export function translateModelName(requestModel: string): string {
  return parseExternalModelId(requestModel)?.modelId ?? requestModel;
}

type Request = Parameters<typeof streamChatCompletions>[0];

async function baseRequest(
  sampling: TranslateSampling | null,
  requestModel: string,
): Promise<Request> {
  if (sampling) {
    return {
      model: requestModel,
      messages: [],
      stream: true,
      max_tokens: TRANSLATE_MAX_TOKENS,
      ...sampling,
      enable_tools: false,
      enable_thinking: false,
    };
  }
  const base = await buildTitleRequest(requestModel, "", "");
  if (!base) throw new TranslateRunError("unavailable");
  // A provider's own minimum can sit above this cap.
  return {
    ...base,
    max_tokens: Math.max(base.max_tokens ?? 0, TRANSLATE_MAX_TOKENS),
  };
}

type TranslateSetup = Omit<TranslateRunOptions, "text" | "onOutput">;

async function prepareTranslation(options: TranslateSetup): Promise<{
  complete: (
    source: string,
    onText?: (text: string) => void,
  ) => Promise<string>;
  model: string;
}> {
  const { prompt, modelId, signal, onPhase } = options;
  onPhase({ kind: "preparing" });
  const dedicated = dedicatedTranslateModel(modelId);
  const custom = modelId === CUSTOM_TRANSLATE_MODEL ? options.custom : null;
  if (modelId === CUSTOM_TRANSLATE_MODEL && !custom)
    throw new TranslateRunError("unavailable");
  const local: LocalModel | null =
    dedicated ??
    (custom && !isExternalModelId(custom.id)
      ? {
          repo: custom.id,
          variant: custom.variant,
          sizeBytes: custom.sizeBytes ?? 0,
        }
      : null);
  const requestModel = local
    ? await ensureLocalModel(local, signal, onPhase)
    : (custom?.id ?? (await chatModel(signal)));
  const promptModel = translatePromptModel(modelId, custom);
  const sampling =
    dedicated?.sampling ??
    (custom && isTranslationOnlyRepo(custom.id) ? HY_MT2_SAMPLING : null);
  const base = await baseRequest(sampling, requestModel);

  const complete = async (
    source: string,
    onText?: (text: string) => void,
  ): Promise<string> => {
    let raw = "";
    let finishReason: string | null = null;
    const stream = streamChatCompletions(
      {
        ...base,
        messages: translateMessages(promptModel, source, prompt),
      },
      signal,
    );
    for await (const event of stream) {
      const choice = event.choices?.[0];
      // A later usage chunk has no finish reason; it must not erase this one.
      finishReason = choice?.finish_reason ?? finishReason;
      const delta = deltaText(choice?.delta?.content);
      if (!delta) continue;
      raw += delta;
      onText?.(stripLeadingReasoning(raw).trimStart());
    }
    // Either stop leaves a partial passage: never one to save as a finished translation.
    if (finishReason === "length") throw new TranslateRunError("cutOff");
    if (finishReason === "content_filter")
      throw new TranslateRunError("filtered");
    return stripLeadingReasoning(raw);
  };
  return {
    complete,
    model: dedicated?.label ?? custom?.name ?? translateModelName(requestModel),
  };
}

export async function runTranslation(
  options: TranslateRunOptions,
): Promise<{ output: string; model: string }> {
  const { text, signal, onPhase, onOutput } = options;
  const { complete, model } = await prepareTranslation(options);
  const subtitles = parseSubtitles(text);
  let output = "";
  if (subtitles) {
    output = await translateSubtitles(subtitles, complete, {
      signal,
      onPhase,
      onOutput,
    });
  } else {
    const chunks = splitForTranslation(text);
    const parts = chunks.filter((chunk) => chunk.text).length;
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
      const translated = await complete(chunk.text, (partial) =>
        onOutput(output + partial),
      );
      output += restoreChunkWhitespace(chunk, translated);
      onOutput(output);
    }
  }
  return { output, model };
}

/** Each text translated on its own, in numbered batches like subtitle cues. */
export async function runTextListTranslation(
  options: TranslateSetup & { texts: readonly string[] },
): Promise<string[]> {
  const { complete } = await prepareTranslation(options);
  return translateTexts(options.texts, complete, {
    signal: options.signal,
    onPhase: options.onPhase,
  });
}
