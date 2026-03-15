// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { ChatModelAdapter } from "@assistant-ui/react";
import type { MessageTiming } from "@assistant-ui/core";
import { toast } from "sonner";
import { generateAudio, streamChatCompletions } from "./chat-api";
import { db } from "../db";
import { useChatRuntimeStore } from "../stores/chat-runtime-store";
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

  return {
    role: message.role,
    content: collectTextParts(message).join("\n"),
  };
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

export function createOpenAIStreamAdapter(): ChatModelAdapter {
  return {
    async *run({ messages, abortSignal, unstable_threadId }) {
      const runtime = useChatRuntimeStore.getState();
      const { params } = runtime;

      if (!params.checkpoint) {
        toast.error("No model loaded", {
          description: "Pick model in top bar, then retry.",
        });
        throw new Error("Load a model first.");
      }

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
          },
          abortSignal,
        );

        for await (const chunk of stream) {
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
