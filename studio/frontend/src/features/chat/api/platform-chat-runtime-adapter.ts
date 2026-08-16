import {
  type PlatformChatReference,
  type PlatformChatUsage,
  platformChatCitations,
  streamPlatformChatCompletion,
} from "@/integrations/platform-backend";
import type { ChatModelAdapter, ThreadMessage } from "@assistant-ui/react";
import { toast } from "sonner";
import { resolvePlatformChatContextForSession } from "./platform-chat-adapter";
import { stripPlatformCitationMarkers } from "./platform-citation-markers";

function textFromMessage(message: ThreadMessage): string {
  return (message.content ?? [])
    .flatMap((part) => {
      if (part.type === "text") return [part.text];
      return [];
    })
    .join("\n")
    .trim();
}

function contentParts(text: string, reasoning: string) {
  return [
    ...(reasoning ? [{ type: "reasoning" as const, text: reasoning }] : []),
    {
      type: "text" as const,
      text: stripPlatformCitationMarkers(text),
    },
  ];
}

/** Native Rag Platform adapter. It never parses OpenAI choices[]. */
export function createPlatformChatRuntimeAdapter(): ChatModelAdapter {
  return {
    async *run({ messages, abortSignal, unstable_threadId }) {
      const sessionId = unstable_threadId;
      if (!sessionId) {
        throw new Error("Rag Platform Session kimliği hazır değil.");
      }
      const chat = await resolvePlatformChatContextForSession(
        sessionId,
        abortSignal,
      );
      if (!chat) {
        throw new Error("Rag Platform Chat kimliği bulunamadı.");
      }
      const chatId = chat.id;
      const modelId = chat.platformLlmId ?? undefined;
      const question = [...messages]
        .reverse()
        .find((message) => message.role === "user");
      const questionText = question ? textFromMessage(question) : "";
      if (!questionText) {
        throw new Error("Gönderilecek kullanıcı mesajı boş.");
      }

      let text = "";
      let reasoning = "";
      let reference: PlatformChatReference | null = null;
      let usage: PlatformChatUsage | null = null;
      let platformMessageId: string | null = null;
      let platformChatId = chatId;
      let platformSessionId = sessionId;
      let completed = false;
      const startedAt = Date.now();
      let finishedAt: number | undefined;

      try {
        for await (const event of streamPlatformChatCompletion(
          {
            chatId,
            sessionId,
            question: questionText,
            legacy: false,
          },
          abortSignal,
        )) {
          if (event.type === "text-delta") text = event.text;
          else if (event.type === "reasoning-delta") reasoning = event.text;
          else if (event.type === "reasoning-end") reasoning = event.text;
          else if (event.type === "reference-update")
            reference = event.reference;
          else if (event.type === "usage") usage = event.usage;
          else if (event.type === "final") {
            text = event.text;
            reasoning = event.reasoning;
            reference = event.reference;
            usage = event.usage;
            platformMessageId = event.messageId;
            platformChatId = event.chatId ?? platformChatId;
            platformSessionId = event.sessionId ?? platformSessionId;
            completed ||= event.terminal;
            if (completed) finishedAt = Date.now();
          } else if (event.type === "error") {
            throw new Error(event.message);
          } else if (event.type === "reasoning-start") {
            reasoning = "";
          }

          if (event.type !== "reasoning-start") {
            yield {
              content: contentParts(text, reasoning),
              metadata: {
                custom: {
                  responseDetails: {
                    ...(modelId
                      ? {
                          modelId,
                          modelLabel: modelId,
                          responseModelId: modelId,
                        }
                      : {}),
                    providerName: "Rag Platform",
                    providerType: "platform",
                    startedAt,
                    ...(finishedAt === undefined
                      ? {}
                      : {
                          finishedAt,
                          durationMs: finishedAt - startedAt,
                        }),
                    sessionId: platformSessionId,
                    toolCalls: [],
                  },
                  contextUsage: usage,
                  platformChatId,
                  platformSessionId,
                  platformMessageId,
                  platformReference: reference,
                  platformCitations: platformChatCitations(reference),
                  platformUsage: usage,
                  platformStreamCompleted: completed,
                },
              },
            };
          }
        }
      } catch (error) {
        if (abortSignal.aborted) {
          toast.info("Yanıt akışı durduruldu.", {
            description:
              "Rag Platform ayrı bir server-side iptal endpoint'i sunmuyor; yalnızca bu tarayıcı bağlantısı kapatıldı.",
          });
          return;
        }
        throw error;
      }
    },
  };
}
