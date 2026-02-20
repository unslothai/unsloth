import type { ChatModelAdapter } from "@assistant-ui/react";
import { toast } from "sonner";
import { streamChatCompletions } from "./chat-api";
import { db } from "../db";
import { useChatRuntimeStore } from "../stores/chat-runtime-store";
import {
  hasClosedThinkTag,
  parseAssistantContent,
} from "../utils/parse-assistant-content";

type RunMessages = Parameters<ChatModelAdapter["run"]>[0]["messages"];
type RunMessage = RunMessages[number];

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
      const useAdapter = await resolveUseAdapter(unstable_threadId);

      const threadKey = unstable_threadId || "__default";
      let waitingFirstChunk = true;
      let firstTokenSettled = false;
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
            ...(useAdapter === undefined ? {} : { use_adapter: useAdapter }),
          },
          abortSignal,
        );

        for await (const chunk of stream) {
          const delta = chunk.choices?.[0]?.delta?.content;
          if (!delta) {
            continue;
          }
          if (waitingFirstChunk) {
            waitingFirstChunk = false;
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
              metadata: { custom: { reasoningDuration } },
            };
          }
        }
        settleFirstTokenOk();
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
