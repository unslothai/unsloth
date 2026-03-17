// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { ChatModelAdapter } from "@assistant-ui/react";
import type { MessageTiming } from "@assistant-ui/core";
import { toast } from "sonner";
import {
  generateAudio,
  listCachedGguf,
  listCachedModels,
  listGgufVariants,
  loadModel,
  streamChatCompletions,
} from "./chat-api";
import { db } from "../db";
import { useChatRuntimeStore } from "../stores/chat-runtime-store";
import type { ChatModelSummary } from "../types/runtime";
import {
  hasClosedThinkTag,
  parseAssistantContent,
} from "../utils/parse-assistant-content";

type RunMessages = Parameters<ChatModelAdapter["run"]>[0]["messages"];
type RunMessage = RunMessages[number];

/** Tracks which user messages were sent with an audio file (messageId → filename). */
export const sentAudioNames = new Map<string, string>();

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
): MessageTiming {
  return {
    streamStartTime,
    firstTokenTime,
    totalStreamTime,
    tokenCount,
    tokensPerSecond:
      typeof totalStreamTime === "number" &&
      totalStreamTime > 0 &&
      typeof tokenCount === "number"
        ? tokenCount / (totalStreamTime / 1000)
        : undefined,
    totalChunks,
    toolCallCount: 0,
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
async function autoLoadSmallestModel(): Promise<boolean> {
  const toastId = toast("Loading a model…", {
    description: "Auto-selecting the smallest downloaded model.",
    duration: 5000,
    closeButton: true,
  });
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
            const loadResp = await loadModel({
              model_path: repo.repo_id,
              hf_token: null,
              max_seq_length: 4096,
              load_in_4bit: true,
              is_lora: false,
              gguf_variant: variant.quant,
              trust_remote_code: false,
            });
            const store = useChatRuntimeStore.getState();
            store.setCheckpoint(repo.repo_id, variant.quant);
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
              supportsReasoning: loadResp.supports_reasoning ?? false,
              reasoningEnabled: loadResp.supports_reasoning ?? false,
              supportsTools: loadResp.supports_tools ?? false,
              toolsEnabled: false,
              defaultChatTemplate: loadResp.chat_template ?? null,
              chatTemplateOverride: null,
            });
            toast.success(`Loaded ${repo.repo_id} (${variant.quant})`, { id: toastId });
            return true;
          }
        } catch {
          continue;
        }
      }
    }

    // Fall back to safetensors models
    if (modelRepos.length > 0) {
      const sorted = [...modelRepos].sort((a, b) => a.size_bytes - b.size_bytes);
      for (const repo of sorted) {
        try {
          const sfLoadResp = await loadModel({
            model_path: repo.repo_id,
            hf_token: null,
            max_seq_length: 4096,
            load_in_4bit: true,
            is_lora: false,
            gguf_variant: null,
            trust_remote_code: false,
          });
          const store = useChatRuntimeStore.getState();
          store.setCheckpoint(repo.repo_id);
          store.setParams({ ...store.params, maxTokens: 4096 });
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
          toast.success(`Loaded ${repo.repo_id}`, { id: toastId });
          return true;
        } catch {
          continue;
        }
      }
    }

    // No cached models found — try downloading a small default GGUF
    toast("Downloading a small model…", {
      id: toastId,
      description: "No downloaded models found. Fetching Qwen3.5-4B (UD-Q4_K_XL).",
      duration: 30000,
    });
    try {
      const loadResp = await loadModel({
        model_path: "unsloth/Qwen3.5-4B-GGUF",
        hf_token: null,
        max_seq_length: 4096,
        load_in_4bit: true,
        is_lora: false,
        gguf_variant: "UD-Q4_K_XL",
        trust_remote_code: false,
      });
      const store = useChatRuntimeStore.getState();
      store.setCheckpoint("unsloth/Qwen3.5-4B-GGUF", "UD-Q4_K_XL");
      store.setParams({ ...store.params, maxTokens: loadResp.context_length ?? 131072 });
      const defaultModel: ChatModelSummary = {
        id: "unsloth/Qwen3.5-4B-GGUF",
        name: loadResp.display_name ?? "Qwen3.5-4B-GGUF",
        isVision: loadResp.is_vision ?? false,
        isLora: false,
        isGguf: true,
      };
      if (!store.models.some((m) => m.id === "unsloth/Qwen3.5-4B-GGUF")) {
        store.setModels([...store.models, defaultModel]);
      }
      useChatRuntimeStore.setState({
        ggufContextLength: loadResp.context_length ?? 131072,
        supportsReasoning: loadResp.supports_reasoning ?? false,
        reasoningEnabled: loadResp.supports_reasoning ?? false,
        supportsTools: loadResp.supports_tools ?? false,
        toolsEnabled: false,
        defaultChatTemplate: loadResp.chat_template ?? null,
        chatTemplateOverride: null,
      });
      toast.success("Loaded Qwen3.5-4B (UD-Q4_K_XL)", { id: toastId });
      return true;
    } catch {
      toast.dismiss(toastId);
      return false;
    }
  } catch {
    toast.dismiss(toastId);
    return false;
  }
}

export function createOpenAIStreamAdapter(): ChatModelAdapter {
  return {
    async *run({ messages, abortSignal, unstable_threadId }) {
      const runtime = useChatRuntimeStore.getState();
      const { params } = runtime;

      // Wait for in-progress model load to finish before inferring
      if (runtime.modelLoading) {
        toast.info("Waiting for model to finish loading…");
        await waitForModelReady(abortSignal);
      }

      if (!useChatRuntimeStore.getState().params.checkpoint) {
        // Auto-load the smallest downloaded model
        const loaded = await autoLoadSmallestModel();
        if (!loaded) {
          toast.error("No model loaded", {
            description: "Pick a model in the top bar, then retry.",
          });
          throw new Error("Load a model first.");
        }
      }

      const {
        supportsTools,
        toolsEnabled,
      } = runtime;

      const outboundMessages = messages
        .map(toOpenAIMessage)
        .filter((message): message is NonNullable<typeof message> =>
          Boolean(message),
        );

      if (params.systemPrompt.trim()) {
        outboundMessages.unshift({
          role: "system",
          content: params.systemPrompt.trim(),
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
      const useAdapter = await resolveUseAdapter(unstable_threadId);

      // ── Audio model path (non-streaming) ─────────────────────
      const activeModel = runtime.models.find(
        (m) => m.id === params.checkpoint,
      );
      if (activeModel?.isAudio && !activeModel?.hasAudioInput) {
        const threadKey = unstable_threadId || "__default";
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

      const threadKey = unstable_threadId || "__default";
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

      let warmupToastShown = false;
      const warmupDelayMs = 450;
      const warmupTimer = setTimeout(() => {
        if (!waitingFirstChunk) return;
        if (abortSignal.aborted) return;
        warmupToastShown = true;
        toast.promise(firstTokenPromise, {
          loading: "Generating",
          success: "Generating",
          error: (err) =>
            err instanceof Error && err.message ? err.message : "Generation failed",
          description: "Waiting for first token.",
          duration: 900,
        });
      }, warmupDelayMs);
      runtime.setThreadRunning(threadKey, true);
      let cumulativeText = "";
      let reasoningStartAt: number | null = null;
      let reasoningDuration = 0;

      try {
        const { supportsReasoning, reasoningEnabled } = runtime;
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
            image_base64: imageBase64,
            audio_base64: audioBase64,
            ...(useAdapter === undefined ? {} : { use_adapter: useAdapter }),
            ...(supportsReasoning ? { enable_thinking: reasoningEnabled } : {}),
            ...(supportsTools && toolsEnabled ? { enable_tools: true } : {}),
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

          totalChunks += 1;
          const delta = chunk.choices?.[0]?.delta?.content;
          if (!delta) {
            continue;
          }
          if (waitingFirstChunk) {
            waitingFirstChunk = false;
            firstTokenTime = Date.now() - streamStartTime;
            settleFirstTokenOk();
          }

          cumulativeText += delta;
          const parts = parseAssistantContent(cumulativeText);

          if (parts.some((part) => part.type === "reasoning") && !reasoningStartAt) {
            reasoningStartAt = Date.now();
          }
          if (hasClosedThinkTag(cumulativeText) && reasoningStartAt && !reasoningDuration) {
            reasoningDuration = Math.round((Date.now() - reasoningStartAt) / 1000);
          }

          if (parts.length > 0) {
            yield {
              content: parts,
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
        yield {
          metadata: {
            timing: buildTiming(
              streamStartTime,
              totalChunks,
              firstTokenTime,
              Date.now() - streamStartTime,
              estimateTokenCount(cumulativeText),
            ),
            custom: { reasoningDuration },
          },
        };
      } catch (err) {
        settleFirstTokenErr(err instanceof Error ? err : new Error("Generation failed"));
        const isEarly = waitingFirstChunk;
        if (!abortSignal.aborted && !(warmupToastShown && isEarly)) {
          toast.error("Generation failed", {
            description: err instanceof Error ? err.message : "Unknown error",
          });
        }
        throw err;
      } finally {
        runtime.setToolStatus(null);
        clearTimeout(warmupTimer);
        if (waitingFirstChunk) {
          if (warmupToastShown && !firstTokenSettled) {
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
