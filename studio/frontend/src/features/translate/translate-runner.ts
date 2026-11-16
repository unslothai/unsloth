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
  prompt: TranslatePromptOptions;
  modelId: TranslateModelId;
  /** The model behind the "custom" id. */
  custom: CustomTranslateModel | null;
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

/** A model this page fetches and loads itself. */
type LocalModel = {
  repo: string;
  /** Null for a model with no quants to pick from; the load fetches whatever it needs. */
  variant: string | null;
  sizeBytes: number;
};

/** Whether the model's file is already on disk; null when the listing could not be read. */
export async function translateModelDownloaded(
  model: LocalModel,
  signal?: AbortSignal,
): Promise<boolean | null> {
  const quant = model.variant?.toLowerCase();
  if (!quant) return null;
  try {
    const listing = await listGgufVariants(model.repo, undefined, { signal });
    const variant = listing.variants.find(
      (entry) => entry.quant?.toLowerCase() === quant,
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
          // Stopped during the preflight, before the job existed.
          if (outcome === "started" && signal.aborted) cancelJob();
          // "conflict" and "busy" are settled in the Hub's download card; "error" already toasted.
          if (outcome !== "started")
            finish(new TranslateRunError("downloadFailed"));
        },
        () => finish(new TranslateRunError("downloadFailed")),
      );
  });
}

/** Downloads the model if it is missing and loads it if it is not serving. Returns its request id. */
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

/** The model picked in Chat when it is a connected provider's, else the one loaded on the server. */
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

/** The request, its messages filled in per passage. A chat model reuses the title request's provider routing and sampling gates. */
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
    max_tokens: Math.max(base.max_tokens, TRANSLATE_MAX_TOKENS),
  };
}

export async function runTranslation(
  options: TranslateRunOptions,
): Promise<{ output: string; model: string }> {
  const { text, prompt, modelId, signal, onPhase, onOutput } = options;
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
  // Any Hy-MT2 quant takes the card's sampling; other models the chat title request's.
  const sampling =
    dedicated?.sampling ??
    (custom && isTranslationOnlyRepo(custom.id) ? HY_MT2_SAMPLING : null);
  const base = await baseRequest(sampling, requestModel);

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
    let raw = "";
    let finishReason: string | null = null;
    const stream = streamChatCompletions(
      {
        ...base,
        messages: translateMessages(promptModel, chunk.text, prompt),
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
      onOutput(output + stripLeadingReasoning(raw).trimStart());
    }
    // Either stop leaves a partial passage: never one to save as a finished translation.
    if (finishReason === "length") throw new TranslateRunError("cutOff");
    if (finishReason === "content_filter")
      throw new TranslateRunError("filtered");
    output += restoreChunkWhitespace(chunk, stripLeadingReasoning(raw));
    onOutput(output);
  }
  return {
    output,
    model: dedicated?.label ?? custom?.name ?? translateModelName(requestModel),
  };
}
