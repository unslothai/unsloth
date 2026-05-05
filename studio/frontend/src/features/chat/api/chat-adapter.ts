// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { ChatModelAdapter } from "@assistant-ui/react";
import type { MessageTiming, ToolCallMessagePart } from "@assistant-ui/core";
import { toast } from "sonner";
import { getAuthToken } from "@/features/auth/session";
import { apiUrl } from "@/lib/api-base";
import {
  generateAudio,
  listCachedGguf,
  listCachedModels,
  listGgufVariants,
  loadModel,
  streamChatCompletions,
  validateModel,
} from "./chat-api";
import { db } from "../db";
import { useChatRuntimeStore } from "../stores/chat-runtime-store";
import { isMultimodalResponse } from "../types/api";
import type { ChatModelSummary } from "../types/runtime";
import {
  hasClosedThinkTag,
  parseAssistantContent,
} from "../utils/parse-assistant-content";

/** Server-side usage data from llama-server (via stream_options.include_usage). */
interface ServerUsage {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
}

/** Server-side timing data from llama-server's timings object. */
interface ServerTimings {
  prompt_n: number;
  cache_n: number;
  prompt_ms: number;
  prompt_per_token_ms: number;
  prompt_per_second: number;
  predicted_n: number;
  predicted_ms: number;
  predicted_per_token_ms: number;
  predicted_per_second: number;
}

type RunMessages = Parameters<ChatModelAdapter["run"]>[0]["messages"];
type RunMessage = RunMessages[number];

/** Tracks which user messages were sent with an audio file (messageId → filename). */
export const sentAudioNames = new Map<string, string>();

/**
 * Match error messages that indicate the request filled or would fill
 * the KV cache, so the UI can show a dedicated toast pointing at the
 * ``Context Length`` setting.
 *
 * Two wordings reach the client and both must hit:
 *
 *   1. The raw llama-server text when ``--no-context-shift`` trips --
 *      "the request exceeds the available context size (N tokens)".
 *   2. The rewritten friendly text emitted by
 *      ``backend/routes/inference.py::_friendly_error`` -- "Message too
 *      long: X tokens exceeds the Y-token context window. Try
 *      increasing the Context Length ..." This is the one most users
 *      see on the streaming GGUF path.
 *
 * We match on substrings rather than full regexes because both layers
 * have drifted across versions (llama.cpp master has tweaked the
 * phrasing; ``_friendly_error`` has gone through several copy edits).
 */
export function isContextLimitError(message: string): boolean {
  if (!message) return false;
  const m = message.toLowerCase();
  return (
    // Raw llama-server wording.
    m.includes("context size") ||
    m.includes("context shift") ||
    m.includes("exceeds the available context") ||
    // Backend _friendly_error rewrite.
    m.includes("message too long") ||
    m.includes("context window") ||
    // n_ctx mentions that carry an "exceed"/"full" signal.
    (m.includes("n_ctx") && (m.includes("exceed") || m.includes("full")))
  );
}

/** Parse "Title: ...\nURL: ...\nSnippet: ..." blocks into source content parts. */
function parseSourcesFromResult(raw: string): { type: "source"; sourceType: "url"; id: string; url: string; title: string; metadata?: { description: string } }[] {
  if (!raw) return [];
  const blocks = raw.split(/\n---\n/).filter(Boolean);
  const sources: { type: "source"; sourceType: "url"; id: string; url: string; title: string; metadata?: { description: string } }[] = [];
  for (const block of blocks) {
    const titleMatch = block.match(/Title:\s*(.+)/);
    const urlMatch = block.match(/URL:\s*(.+)/);
    const snippetMatch = block.match(/Snippet:\s*(.+)/);
    if (titleMatch && urlMatch) {
      const url = urlMatch[1].trim();
      const snippet = snippetMatch?.[1]?.trim();
      sources.push({
        type: "source" as const,
        sourceType: "url" as const,
        id: url,
        url,
        title: titleMatch[1].trim(),
        ...(snippet ? { metadata: { description: snippet } } : {}),
      });
    }
  }
  return sources;
}

function estimateTokenCount(text: string): number | undefined {
  const trimmed = text.trim();
  if (!trimmed) {
    return undefined;
  }
  return Math.max(1, Math.round(trimmed.length / 4));
}

function buildTiming(
  streamStartTime: number,
  totalChunks: number,
  firstTokenTime?: number,
  totalStreamTime?: number,
  tokenCount?: number,
  toolCallCount = 0,
  tokensPerSecondOverride?: number,
): MessageTiming {
  return {
    streamStartTime,
    firstTokenTime,
    totalStreamTime,
    tokenCount,
    tokensPerSecond:
      tokensPerSecondOverride ??
      (typeof totalStreamTime === "number" &&
      totalStreamTime > 0 &&
      typeof tokenCount === "number"
        ? tokenCount / (totalStreamTime / 1000)
        : undefined),
    totalChunks,
    toolCallCount,
  };
}

function collectTextParts(message: RunMessage): string[] {
  const textParts = message.content
    .filter((part) => part.type === "text")
    .map((part) => part.text);

  if ("attachments" in message && (message.attachments?.length ?? 0) > 0) {
    for (const attachment of message.attachments ?? []) {
      for (const part of attachment.content ?? []) {
        if (part.type === "text") {
          textParts.push(part.text);
        }
      }
    }
  }

  return textParts;
}

function toOpenAIMessage(message: RunMessage): {
  role: "system" | "user" | "assistant";
  content: string;
} | null {
  if (
    message.role !== "system" &&
    message.role !== "user" &&
    message.role !== "assistant"
  ) {
    return null;
  }

  let content = collectTextParts(message).join("\n");
  // Strip inline audio base64 from prior assistant messages to avoid
  // inflating token counts (e.g. audio-player responses with embedded WAV).
  if (message.role === "assistant") {
    content = content.replace(
      /data:audio\/[a-z0-9.+-]+;base64,[A-Za-z0-9+/=]+/g,
      "[audio]",
    );
  }

  return { role: message.role, content };
}

function extractImageBase64(input: string): string | undefined {
  if (!input) {
    return undefined;
  }
  if (input.startsWith("data:")) {
    const commaIndex = input.indexOf(",");
    return commaIndex >= 0 ? input.slice(commaIndex + 1) : undefined;
  }
  return input;
}

function findLatestUserImageBase64(messages: RunMessages): string | undefined {
  for (let i = messages.length - 1; i >= 0; i -= 1) {
    const message = messages[i];
    if (!message || message.role !== "user") {
      continue;
    }

    // Image in message.content (e.g. compare view appends content with image parts)
    for (const part of message.content ?? []) {
      if (part.type === "image" && "image" in part) {
        const encoded = extractImageBase64(part.image);
        if (encoded) return encoded;
      }
    }

    // Image in message.attachments (e.g. chat composer)
    if ("attachments" in message && (message.attachments?.length ?? 0) > 0) {
      for (const attachment of message.attachments ?? []) {
        for (const part of attachment.content ?? []) {
          if (part.type !== "image") {
            continue;
          }
          const encoded = extractImageBase64(part.image);
          if (encoded) {
            return encoded;
          }
        }
      }
    }
  }

  return undefined;
}

function findLatestUserAudioBase64(messages: RunMessages): string | undefined {
  // Check message content parts (from compare view's CompareMessagePart with type: "audio")
  for (let i = messages.length - 1; i >= 0; i -= 1) {
    const message = messages[i];
    if (!message || message.role !== "user") continue;

    for (const part of message.content ?? []) {
      if (part.type === "audio" && "audio" in part) {
        const audioPart = (part as unknown as { type: "audio"; audio: string | { data: string; format: string } }).audio;
        const raw = typeof audioPart === "string" ? audioPart : audioPart?.data;
        if (raw) return raw.startsWith("data:") ? raw.split(",")[1] : raw;
      }
    }
  }

  // Check the runtime store (from main composer's audio upload)
  const pendingAudio = useChatRuntimeStore.getState().pendingAudioBase64;
  return pendingAudio ?? undefined;
}

async function resolveUseAdapter(
  threadId: string | undefined,
): Promise<boolean | undefined> {
  if (!threadId) {
    return undefined;
  }
  try {
    const thread = await db.threads.get(threadId);
    if (!thread?.pairId) {
      return undefined;
    }
    // model1/model2 threads don't use the adapter toggle — each side
    // loads its own model via /api/inference/load before generation.
    if (thread.modelType === "model1" || thread.modelType === "model2") {
      return undefined;
    }
    return thread.modelType === "lora";
  } catch {
    return undefined;
  }
}

/** Wait for an in-progress model load to finish (polls store every 500ms). */
function waitForModelReady(abortSignal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    const check = () => {
      if (abortSignal?.aborted) { reject(new Error("Aborted")); return; }
      if (!useChatRuntimeStore.getState().modelLoading) { resolve(); return; }
      setTimeout(check, 500);
    };
    check();
  });
}

/**
 * Auto-load the smallest downloaded model when the user tries to chat
 * without selecting one. Prefers GGUF (picks smallest cached variant),
 * falls back to smallest cached safetensors model.
 */
async function autoLoadSmallestModel(): Promise<{
  loaded: boolean;
  blockedByTrustRemoteCode: boolean;
}> {
  const store = useChatRuntimeStore.getState();
  const hfToken = store.hfToken || null;
  const trustRemoteCode = store.params.trustRemoteCode ?? false;
  const toastId = toast("Loading a model…", {
    description: "Auto-selecting the smallest downloaded model.",
    duration: 5000,
    closeButton: true,
  });
  let blockedByTrustRemoteCode = false;
  let hadNonTrustFailure = false;

  async function canAutoLoad(payload: {
    model_path: string;
    max_seq_length: number;
    is_lora: boolean;
    gguf_variant?: string | null;
  }): Promise<boolean> {
    const validation = await validateModel({
      ...payload,
      hf_token: hfToken,
      load_in_4bit: true,
      trust_remote_code: trustRemoteCode,
    });
    if (validation.requires_trust_remote_code && !trustRemoteCode) {
      blockedByTrustRemoteCode = true;
      return false;
    }
    return true;
  }
  try {
    const [ggufRepos, modelRepos] = await Promise.all([
      listCachedGguf().catch(() => []),
      listCachedModels().catch(() => []),
    ]);

    // Try GGUF first: pick the repo with the smallest total size,
    // then pick its smallest downloaded variant.
    if (ggufRepos.length > 0) {
      const sorted = [...ggufRepos].sort((a, b) => a.size_bytes - b.size_bytes);
      for (const repo of sorted) {
        try {
          const variants = await listGgufVariants(repo.repo_id);
          const downloaded = variants.variants
            .filter((v) => v.downloaded)
            .sort((a, b) => a.size_bytes - b.size_bytes);
          if (downloaded.length > 0) {
            const variant = downloaded[0];
            if (
              !(await canAutoLoad({
                model_path: repo.repo_id,
                max_seq_length: 0,
                is_lora: false,
                gguf_variant: variant.quant,
              }))
            ) {
              continue;
            }
            const loadResp = await loadModel({
              model_path: repo.repo_id,
              hf_token: hfToken,
              max_seq_length: 0,
              load_in_4bit: true,
              is_lora: false,
              gguf_variant: variant.quant,
              trust_remote_code: trustRemoteCode,
            });
            useChatRuntimeStore.getState().setCheckpoint(repo.repo_id, variant.quant);
            const store = useChatRuntimeStore.getState();
            store.setModelRequiresTrustRemoteCode(
              loadResp.requires_trust_remote_code ?? false,
            );
            store.setParams({ ...store.params, maxTokens: loadResp.context_length ?? 131072 });
            // Add model to store so the selector shows the name
            const autoModel: ChatModelSummary = {
              id: repo.repo_id,
              name: loadResp.display_name ?? repo.repo_id,
              isVision: loadResp.is_vision ?? false,
              isLora: loadResp.is_lora ?? false,
              isGguf: loadResp.is_gguf ?? false,
              isAudio: loadResp.is_audio ?? false,
              audioType: loadResp.audio_type ?? null,
              hasAudioInput: loadResp.has_audio_input ?? false,
            };
            const existingModels = store.models;
            if (!existingModels.some((m) => m.id === repo.repo_id)) {
              store.setModels([...existingModels, autoModel]);
            }
            useChatRuntimeStore.setState({
              ggufContextLength: loadResp.context_length ?? 131072,
              ggufMaxContextLength: loadResp.max_context_length ?? loadResp.context_length ?? 131072,
              supportsReasoning: loadResp.supports_reasoning ?? false,
              reasoningAlwaysOn: loadResp.reasoning_always_on ?? false,
              reasoningEnabled: loadResp.supports_reasoning ?? false,
              reasoningStyle: loadResp.reasoning_style ?? "enable_thinking",
              supportsPreserveThinking: loadResp.supports_preserve_thinking ?? false,
              supportsTools: loadResp.supports_tools ?? false,
              toolsEnabled: loadResp.supports_tools ?? false,
              codeToolsEnabled: loadResp.supports_tools ?? false,
              kvCacheDtype: loadResp.cache_type_kv ?? null,
              loadedKvCacheDtype: loadResp.cache_type_kv ?? null,
              defaultChatTemplate: loadResp.chat_template ?? null,
              chatTemplateOverride: null,
              loadedChatTemplateOverride: null,
              loadedIsMultimodal: isMultimodalResponse(loadResp),
            });
            toast.success(`Loaded ${repo.repo_id} (${variant.quant})`, { id: toastId });
            return { loaded: true, blockedByTrustRemoteCode: false };
          }
        } catch {
          hadNonTrustFailure = true;
          continue;
        }
      }
    }

    // Fall back to safetensors models
    if (modelRepos.length > 0) {
      const sorted = [...modelRepos].sort((a, b) => a.size_bytes - b.size_bytes);
      for (const repo of sorted) {
        try {
          if (
            !(await canAutoLoad({
              model_path: repo.repo_id,
              max_seq_length: 4096,
              is_lora: false,
              gguf_variant: null,
            }))
          ) {
            continue;
          }
          const sfLoadResp = await loadModel({
            model_path: repo.repo_id,
            hf_token: hfToken,
            max_seq_length: 4096,
            load_in_4bit: true,
            is_lora: false,
            gguf_variant: null,
            trust_remote_code: trustRemoteCode,
          });
          useChatRuntimeStore.getState().setCheckpoint(repo.repo_id);
          const store = useChatRuntimeStore.getState();
          store.setModelRequiresTrustRemoteCode(
            sfLoadResp.requires_trust_remote_code ?? false,
          );
          store.setParams({ ...store.params, maxTokens: 4096 });
          useChatRuntimeStore.setState({
            supportsReasoning: sfLoadResp.supports_reasoning ?? false,
            reasoningAlwaysOn: sfLoadResp.reasoning_always_on ?? false,
            reasoningEnabled: sfLoadResp.supports_reasoning ?? false,
            reasoningStyle: sfLoadResp.reasoning_style ?? "enable_thinking",
            supportsPreserveThinking: sfLoadResp.supports_preserve_thinking ?? false,
            supportsTools: sfLoadResp.supports_tools ?? false,
          });
          const sfModel: ChatModelSummary = {
            id: repo.repo_id,
            name: sfLoadResp.display_name ?? repo.repo_id,
            isVision: sfLoadResp.is_vision ?? false,
            isLora: sfLoadResp.is_lora ?? false,
            isGguf: sfLoadResp.is_gguf ?? false,
          };
          if (!store.models.some((m) => m.id === repo.repo_id)) {
            store.setModels([...store.models, sfModel]);
          }
          useChatRuntimeStore.setState({
            loadedIsMultimodal: isMultimodalResponse(sfLoadResp),
          });
          toast.success(`Loaded ${repo.repo_id}`, { id: toastId });
          return { loaded: true, blockedByTrustRemoteCode: false };
        } catch {
          hadNonTrustFailure = true;
          continue;
        }
      }
    }

    // No cached models found — try downloading a small default GGUF
    toast("Downloading a small model…", {
      id: toastId,
      description: "No downloaded models found. Fetching Gemma-4-E2B-it (UD-Q4_K_XL).",
      duration: 30000,
    });
    try {
      if (
        !(await canAutoLoad({
          model_path: "unsloth/gemma-4-E2B-it-GGUF",
          max_seq_length: 0,
          is_lora: false,
          gguf_variant: "UD-Q4_K_XL",
        }))
      ) {
        toast.dismiss(toastId);
        return { loaded: false, blockedByTrustRemoteCode };
      }
      const loadResp = await loadModel({
        model_path: "unsloth/gemma-4-E2B-it-GGUF",
        hf_token: hfToken,
        max_seq_length: 0,
        load_in_4bit: true,
        is_lora: false,
        gguf_variant: "UD-Q4_K_XL",
        trust_remote_code: trustRemoteCode,
      });
      useChatRuntimeStore.getState().setCheckpoint("unsloth/gemma-4-E2B-it-GGUF", "UD-Q4_K_XL");
      const store = useChatRuntimeStore.getState();
      store.setModelRequiresTrustRemoteCode(
        loadResp.requires_trust_remote_code ?? false,
      );
      store.setParams({ ...store.params, maxTokens: loadResp.context_length ?? 131072 });
      const defaultModel: ChatModelSummary = {
        id: "unsloth/gemma-4-E2B-it-GGUF",
        name: loadResp.display_name ?? "gemma-4-E2B-it-GGUF",
        isVision: loadResp.is_vision ?? false,
        isLora: false,
        isGguf: true,
      };
      if (!store.models.some((m) => m.id === "unsloth/gemma-4-E2B-it-GGUF")) {
        store.setModels([...store.models, defaultModel]);
      }
      useChatRuntimeStore.setState({
        ggufContextLength: loadResp.context_length ?? 131072,
        ggufMaxContextLength: loadResp.max_context_length ?? loadResp.context_length ?? 131072,
        supportsReasoning: loadResp.supports_reasoning ?? false,
        reasoningAlwaysOn: loadResp.reasoning_always_on ?? false,
        reasoningEnabled: loadResp.supports_reasoning ?? false,
        reasoningStyle: loadResp.reasoning_style ?? "enable_thinking",
        supportsPreserveThinking: loadResp.supports_preserve_thinking ?? false,
        supportsTools: loadResp.supports_tools ?? false,
        toolsEnabled: loadResp.supports_tools ?? false,
        codeToolsEnabled: loadResp.supports_tools ?? false,
        kvCacheDtype: loadResp.cache_type_kv ?? null,
        loadedKvCacheDtype: loadResp.cache_type_kv ?? null,
        defaultChatTemplate: loadResp.chat_template ?? null,
        chatTemplateOverride: null,
        loadedIsMultimodal: isMultimodalResponse(loadResp),
      });
      toast.success("Loaded Gemma-4-E2B-it (UD-Q4_K_XL)", { id: toastId });
      return { loaded: true, blockedByTrustRemoteCode: false };
    } catch {
      toast.dismiss(toastId);
      hadNonTrustFailure = true;
      return {
        loaded: false,
        blockedByTrustRemoteCode:
          blockedByTrustRemoteCode && !hadNonTrustFailure,
      };
    }
  } catch {
    toast.dismiss(toastId);
    hadNonTrustFailure = true;
    return {
      loaded: false,
      blockedByTrustRemoteCode:
        blockedByTrustRemoteCode && !hadNonTrustFailure,
    };
  }
}

export function createOpenAIStreamAdapter(): ChatModelAdapter {
  return {
    async *run({ messages, abortSignal, unstable_threadId }) {
      let runtime = useChatRuntimeStore.getState();
      // Capture the thread ID once at the start so it stays stable even if
      // the user switches chats while waiting for model load / auto-load.
      const resolvedThreadId =
        (unstable_threadId ?? runtime.activeThreadId) || undefined;

      // Wait for in-progress model load to finish before inferring
      if (runtime.modelLoading) {
        toast.info("Waiting for model to finish loading…");
        await waitForModelReady(abortSignal);
      }

      if (!useChatRuntimeStore.getState().params.checkpoint) {
        // Auto-load the smallest downloaded model
        const { loaded, blockedByTrustRemoteCode } =
          await autoLoadSmallestModel();
        if (!loaded) {
          toast.error(
            blockedByTrustRemoteCode
              ? "Enable custom code to auto-load this model"
              : "No model loaded",
            {
              description: blockedByTrustRemoteCode
                ? 'Turn on "Enable custom code" in Chat Settings, or pick another model in the top bar.'
                : "Pick a model in the top bar, then retry.",
            },
          );
          throw new Error("Load a model first.");
        }
      }

      // Re-read store after potential auto-load / model ready wait
      runtime = useChatRuntimeStore.getState();
      const { params } = runtime;
      const {
        supportsTools,
        toolsEnabled,
        codeToolsEnabled,
      } = runtime;

      const outboundMessages = messages
        .map(toOpenAIMessage)
        .filter((message): message is NonNullable<typeof message> =>
          Boolean(message),
        );

      const safeSystemPrompt =
        typeof params.systemPrompt === "string" ? params.systemPrompt : "";
      if (safeSystemPrompt.trim()) {
        outboundMessages.unshift({
          role: "system",
          content: safeSystemPrompt.trim(),
        });
      }
      const imageBase64 = findLatestUserImageBase64(messages);
      const audioBase64 = findLatestUserAudioBase64(messages);
      // Clear pending audio from store after extracting (consumed on send)
      if (audioBase64) {
        const audioName = runtime.pendingAudioName;
        if (audioName) {
          const lastUserMsg = [...messages].reverse().find((m) => m.role === "user");
          if (lastUserMsg) sentAudioNames.set(lastUserMsg.id, audioName);
        }
        runtime.clearPendingAudio();
      }
      const useAdapter = await resolveUseAdapter(resolvedThreadId);

      // ── Audio model path (non-streaming) ─────────────────────
      const activeModel = runtime.models.find(
        (m) => m.id === params.checkpoint,
      );
      if (activeModel?.isAudio && !activeModel?.hasAudioInput) {
        const threadKey = resolvedThreadId || "__default";
        runtime.setThreadRunning(threadKey, true);
        try {
          yield {
            content: [{ type: "text" as const, text: "Generating audio..." }],
          };

          const result = await generateAudio(
            {
              model: params.checkpoint,
              messages: outboundMessages,
              stream: false,
              temperature: params.temperature,
              top_p: params.topP,
              max_tokens: params.maxTokens,
              top_k: params.topK,
              min_p: params.minP,
              repetition_penalty: params.repetitionPenalty,
              presence_penalty: params.presencePenalty,
              ...(useAdapter === undefined ? {} : { use_adapter: useAdapter }),
            },
            abortSignal,
          );

          const audioUrl = `data:audio/wav;base64,${result.audio.data}`;
          yield {
            content: [
              {
                type: "text" as const,
                text: `<audio-player src="${audioUrl}" />`,
              },
            ],
          };
        } catch (err) {
          if (!abortSignal.aborted) {
            toast.error("Audio generation failed", {
              description:
                err instanceof Error ? err.message : "Unknown error",
            });
          }
          throw err;
        } finally {
          runtime.setThreadRunning(threadKey, false);
        }
        return;
      }

      const threadKey = resolvedThreadId || "__default";
      let waitingFirstChunk = true;
      let firstTokenSettled = false;
      const streamStartTime = Date.now();
      let firstTokenTime: number | undefined;
      let totalChunks = 0;
      let resolveFirstToken: (() => void) | null = null;
      let rejectFirstToken: ((err: unknown) => void) | null = null;
      const firstTokenPromise = new Promise<void>((resolve, reject) => {
        resolveFirstToken = resolve;
        rejectFirstToken = reject;
      });
      // Avoid unhandled rejections if toast.promise never attached.
      void firstTokenPromise.catch(() => {});

      function settleFirstTokenOk(): void {
        if (firstTokenSettled) return;
        firstTokenSettled = true;
        resolveFirstToken?.();
      }

      function settleFirstTokenErr(err: unknown): void {
        if (firstTokenSettled) return;
        firstTokenSettled = true;
        rejectFirstToken?.(err);
      }

      const warmupDelayMs = 450;
      const warmupTimer = setTimeout(() => {
        if (!waitingFirstChunk) return;
        if (abortSignal.aborted) return;
        runtime.setGeneratingStatus("waiting");
      }, warmupDelayMs);
      runtime.setThreadRunning(threadKey, true);
      let cumulativeText = "";
      let reasoningStartAt: number | null = null;
      let reasoningDuration = 0;
      // Tool call content parts — accumulated and yielded cumulatively.
      // result is set directly on the tool-call part when tool_end arrives.
      const toolCallParts: ToolCallMessagePart[] = [];
      let serverMetadata: { usage?: ServerUsage; timings?: ServerTimings } | null = null;

      // Per-run cancellation token so a delayed stop POST cannot match
      // the next run on the same thread.
      const cancelId =
        typeof crypto !== "undefined" && "randomUUID" in crypto
          ? crypto.randomUUID()
          : `${Date.now()}-${Math.random().toString(36).slice(2)}`;

      // Colab-style proxies can swallow fetch aborts, so also POST
      // /inference/cancel explicitly on abort.
      const onAbortCancel = () => {
        const body: Record<string, string> = { cancel_id: cancelId };
        if (resolvedThreadId) body.session_id = resolvedThreadId;
        // Plain fetch, not authFetch: authFetch redirects to login on
        // 401, which would kick the user out mid-stop.
        const token = getAuthToken();
        // Use apiUrl so the cancel POST reaches the right origin in
        // Tauri production builds (where the webview origin is not the
        // backend at 127.0.0.1:<port>). Browser/dev builds get the empty
        // base, so the path is unchanged there.
        void fetch(apiUrl("/api/inference/cancel"), {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            ...(token ? { Authorization: `Bearer ${token}` } : {}),
          },
          body: JSON.stringify(body),
          keepalive: true,
        }).catch(() => {});
      };
      try {
        if (abortSignal.aborted) {
          onAbortCancel();
        } else {
          abortSignal.addEventListener("abort", onAbortCancel, { once: true });
        }

        const {
          supportsReasoning,
          reasoningEnabled,
          reasoningStyle,
          reasoningEffort,
          supportsPreserveThinking,
          preserveThinking,
        } = runtime;
        const stream = streamChatCompletions(
          {
            model: params.checkpoint,
            messages: outboundMessages,
            stream: true,
            temperature: params.temperature,
            top_p: params.topP,
            max_tokens: params.maxTokens,
            top_k: params.topK,
            min_p: params.minP,
            repetition_penalty: params.repetitionPenalty,
            presence_penalty: params.presencePenalty,
            image_base64: imageBase64,
            audio_base64: audioBase64,
            cancel_id: cancelId,
            ...(resolvedThreadId ? { session_id: resolvedThreadId } : {}),
            ...(useAdapter === undefined ? {} : { use_adapter: useAdapter }),
            ...(supportsReasoning
              ? reasoningStyle === "reasoning_effort"
                ? { reasoning_effort: reasoningEffort }
                : { enable_thinking: reasoningEnabled }
              : {}),
            ...(supportsPreserveThinking ? { preserve_thinking: preserveThinking } : {}),
            ...(supportsTools && (toolsEnabled || codeToolsEnabled)
              ? {
                  enable_tools: true,
                  enabled_tools: [
                    ...(toolsEnabled ? ["web_search"] : []),
                    ...(codeToolsEnabled ? ["python", "terminal"] : []),
                  ],
                  auto_heal_tool_calls: useChatRuntimeStore.getState().autoHealToolCalls,
                  max_tool_calls_per_message: useChatRuntimeStore.getState().maxToolCallsPerMessage,
                  tool_call_timeout: (() => {
                    const mins = useChatRuntimeStore.getState().toolCallTimeout;
                    return mins >= 9999 ? 9999 : mins * 60;
                  })(),
                }
              : {}),
          },
          abortSignal,
        );

        for await (const chunk of stream) {
          // Handle tool status events
          const toolStatusText = (chunk as unknown as { _toolStatus?: string })._toolStatus;
          if (toolStatusText !== undefined) {
            runtime.setToolStatus(toolStatusText || null);
            continue;
          }

          // Emit tool-call content parts for assistant-ui.
          // On tool_start: add a new tool-call part (renders in "running" state).
          // On tool_end: set result on the existing part (transitions to "complete").
          const toolEvent = (chunk as unknown as { _toolEvent?: Record<string, unknown> })._toolEvent;
          if (toolEvent !== undefined) {
            if (toolEvent.type === "tool_start") {
              const id = (toolEvent.tool_call_id as string) || `${toolEvent.tool_name}_${Date.now()}`;
              const toolArgs = (toolEvent.arguments ?? {}) as ToolCallMessagePart["args"];
              toolCallParts.push({
                type: "tool-call" as const,
                toolCallId: id,
                toolName: toolEvent.tool_name as string,
                argsText: JSON.stringify(toolArgs),
                args: toolArgs,
              });
            } else if (toolEvent.type === "tool_end") {
              const id = (toolEvent.tool_call_id as string) ||
                toolCallParts[toolCallParts.length - 1]?.toolCallId || "";
              const idx = toolCallParts.findIndex((p) => p.toolCallId === id);
              if (idx !== -1) {
                const rawResult = (toolEvent.result as string) ?? "";
                const imgMarker = "\n__IMAGES__:";
                const imgIdx = rawResult.lastIndexOf(imgMarker);
                let parsedResult: string | { text: string; images: string[]; sessionId: string };
                if (imgIdx !== -1) {
                  const text = rawResult.slice(0, imgIdx);
                  // Fall back to "_default" to match the backend sandbox directory
                  // used when no session_id is provided (see tools.py _get_workdir).
                  const sessionId = resolvedThreadId || "_default";
                  try {
                    const images = JSON.parse(rawResult.slice(imgIdx + imgMarker.length)) as string[];
                    parsedResult = { text, images, sessionId };
                  } catch {
                    parsedResult = rawResult;
                  }
                } else {
                  parsedResult = rawResult;
                }
                toolCallParts[idx] = { ...toolCallParts[idx], result: parsedResult };
              }
            }
            // Yield cumulative state so tool UI updates (tools first, text after)
            const textParts = parseAssistantContent(cumulativeText);
            yield {
              content: [...toolCallParts, ...textParts],
              metadata: {
                timing: buildTiming(streamStartTime, totalChunks, firstTokenTime),
                custom: { reasoningDuration },
              },
            };
            continue;
          }

          // OpenAI-standard usage chunk: choices=[], usage populated
          if (chunk.choices?.length === 0 && chunk.usage) {
            serverMetadata = {
              usage: chunk.usage,
              timings: (chunk as Record<string, unknown>).timings as ServerTimings | undefined,
            };
            continue;
          }

          totalChunks += 1;
          const delta = chunk.choices?.[0]?.delta?.content;
          if (!delta) {
            continue;
          }
          if (waitingFirstChunk) {
            waitingFirstChunk = false;
            firstTokenTime = Date.now() - streamStartTime;
            settleFirstTokenOk();
            runtime.setGeneratingStatus(null);
          }

          cumulativeText += delta;
          const parts = parseAssistantContent(cumulativeText);

          if (parts.some((part) => part.type === "reasoning") && !reasoningStartAt) {
            reasoningStartAt = Date.now();
          }
          if (hasClosedThinkTag(cumulativeText) && reasoningStartAt && !reasoningDuration) {
            reasoningDuration = Math.round((Date.now() - reasoningStartAt) / 1000);
          }

          if (parts.length > 0 || toolCallParts.length > 0) {
            yield {
              content: [...toolCallParts, ...parts],
              metadata: {
                timing: buildTiming(
                  streamStartTime,
                  totalChunks,
                  firstTokenTime,
                ),
                custom: { reasoningDuration },
              },
            };
          }
        }
        settleFirstTokenOk();

        // Extract source parts from completed web_search tool calls
        const sourceParts = toolCallParts.flatMap((tc) => {
          if (tc.toolName !== "web_search" || !tc.result) return [];
          return parseSourcesFromResult(typeof tc.result === "string" ? tc.result : "");
        });

        const meta = serverMetadata;
        const finalTokenCount = meta?.usage?.completion_tokens
          ?? estimateTokenCount(cumulativeText);
        const finalTokPerSec = meta?.timings?.predicted_per_second;
        const serverPromptEvalTime = meta?.timings?.prompt_ms;

        // Update context usage in store if we got valid server data
        if (
          meta?.usage &&
          typeof meta.usage.prompt_tokens === "number" &&
          typeof meta.usage.completion_tokens === "number" &&
          typeof meta.usage.total_tokens === "number"
        ) {
          useChatRuntimeStore.getState().setContextUsage({
            promptTokens: meta.usage.prompt_tokens,
            completionTokens: meta.usage.completion_tokens,
            totalTokens: meta.usage.total_tokens,
            cachedTokens: meta.timings?.cache_n ?? 0,
          });
        }

        const finalTiming = buildTiming(
          streamStartTime,
          totalChunks,
          serverPromptEvalTime ?? firstTokenTime,
          Date.now() - streamStartTime,
          finalTokenCount,
          toolCallParts.length,
          finalTokPerSec,
        );

        yield {
          content: [
            ...toolCallParts,
            ...parseAssistantContent(cumulativeText),
            ...sourceParts,
          ],
          metadata: {
            timing: finalTiming,
            custom: {
              reasoningDuration,
              serverTimings: meta?.timings ?? undefined,
              contextUsage: meta?.usage ? {
                promptTokens: meta.usage.prompt_tokens,
                completionTokens: meta.usage.completion_tokens,
                totalTokens: meta.usage.total_tokens,
                cachedTokens: meta.timings?.cache_n ?? 0,
                modelId: params.checkpoint,
              } : undefined,
              timing: finalTiming,
            },
          },
        };
      } catch (err) {
        settleFirstTokenErr(err instanceof Error ? err : new Error("Generation failed"));
        if (!abortSignal.aborted) {
          const msg = err instanceof Error ? err.message : String(err);
          if (isContextLimitError(msg)) {
            // llama-server was launched with --no-context-shift, so it
            // returns a hard error instead of silently dropping old
            // turns from the KV cache. Point the user at the exact
            // control that raises the ceiling.
            toast.error("Context limit reached", {
              description:
                "The conversation has filled the model's context window. " +
                "Increase \"Context Length\" in the chat Settings panel (⚙ in the top-right), " +
                "or start a new chat.",
              duration: 8000,
            });
          } else {
            toast.error("Generation failed", {
              description: msg || "Unknown error",
            });
          }
        }
        throw err;
      } finally {
        abortSignal.removeEventListener("abort", onAbortCancel);
        runtime.setGeneratingStatus(null);
        runtime.setToolStatus(null);
        clearTimeout(warmupTimer);
        if (waitingFirstChunk) {
          if (!firstTokenSettled) {
            if (abortSignal.aborted) {
              settleFirstTokenErr(new Error("Cancelled"));
            } else {
              settleFirstTokenErr(new Error("No tokens received"));
            }
          } else {
            settleFirstTokenOk();
          }
        }
        runtime.setThreadRunning(threadKey, false);
      }
    },
  };
}
