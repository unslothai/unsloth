import { describe, expect, it, vi } from "vitest";

const mocks = vi.hoisted(() => ({
  resolveChat: vi.fn(),
  stream: vi.fn(),
}));

vi.mock("./platform-chat-adapter", () => ({
  resolvePlatformChatContextForSession: mocks.resolveChat,
}));
vi.mock("@/integrations/platform-backend", async (importOriginal) => {
  const actual =
    await importOriginal<typeof import("@/integrations/platform-backend")>();
  return { ...actual, streamPlatformChatCompletion: mocks.stream };
});

import { createPlatformChatRuntimeAdapter } from "./platform-chat-runtime-adapter";

describe("Rag Platform Phase 8 runtime adapter", () => {
  it("uses native normalized events and publishes citation metadata", async () => {
    mocks.resolveChat.mockResolvedValue({
      id: "chat-1",
      platformLlmId: "provider/model-1",
    });
    mocks.stream.mockImplementation(async function* () {
      yield {
        type: "text-delta",
        delta: "Answer [ID:0]",
        text: "Answer [ID:0]",
      };
      yield {
        type: "reference-update",
        reference: {
          chunks: [
            {
              id: "chunk-1",
              chunkId: "chunk-1",
              documentId: "doc-1",
              datasetId: "dataset-1",
              filename: "Guide.pdf",
              text: "Evidence",
              page: 3,
              score: 0.9,
            },
          ],
          documentAggregations: [],
        },
      };
      yield {
        type: "final",
        terminal: true,
        messageId: "turn-1",
        chatId: "chat-1",
        sessionId: "session-1",
        text: "Answer [ID:0]",
        reasoning: "",
        reference: {
          chunks: [
            {
              id: "chunk-1",
              chunkId: "chunk-1",
              documentId: "doc-1",
              datasetId: "dataset-1",
              filename: "Guide.pdf",
              text: "Evidence",
              page: 3,
              score: 0.9,
            },
          ],
          documentAggregations: [],
        },
        usage: { totalTokens: 12 },
      };
    });

    const outputs: unknown[] = [];
    const adapter = createPlatformChatRuntimeAdapter();
    const run = adapter.run({
      messages: [
        {
          id: "user-1",
          role: "user",
          content: [{ type: "text", text: "Question" }],
          createdAt: new Date(),
        },
      ],
      abortSignal: new AbortController().signal,
      unstable_threadId: "session-1",
      context: {},
    } as never) as AsyncGenerator<unknown>;
    for await (const output of run) outputs.push(output);

    expect(mocks.stream).toHaveBeenCalledWith(
      {
        chatId: "chat-1",
        sessionId: "session-1",
        question: "Question",
        legacy: false,
      },
      expect.any(AbortSignal),
    );
    expect(outputs.at(-1)).toMatchObject({
      content: [{ type: "text", text: "Answer" }],
      metadata: {
        custom: {
          responseDetails: {
            modelId: "provider/model-1",
            modelLabel: "provider/model-1",
            responseModelId: "provider/model-1",
            providerName: "Rag Platform",
            providerType: "platform",
            startedAt: expect.any(Number),
            finishedAt: expect.any(Number),
            durationMs: expect.any(Number),
            sessionId: "session-1",
          },
          contextUsage: { totalTokens: 12 },
          platformChatId: "chat-1",
          platformSessionId: "session-1",
          platformMessageId: "turn-1",
          platformStreamCompleted: true,
          platformCitations: [{ documentId: "doc-1", filename: "Guide.pdf" }],
        },
      },
    });
  });

  it("requires a resolvable backend Chat id", async () => {
    mocks.resolveChat.mockResolvedValue(null);
    const adapter = createPlatformChatRuntimeAdapter();
    const run = adapter.run({
      messages: [],
      abortSignal: new AbortController().signal,
      unstable_threadId: "session-1",
      context: {},
    } as never) as AsyncGenerator<unknown>;
    await expect(run.next()).rejects.toThrow("Chat kimliği bulunamadı");
  });

  it("surfaces native stream errors instead of ending with an empty answer", async () => {
    mocks.resolveChat.mockResolvedValue({ id: "chat-1" });
    mocks.stream.mockImplementation(async function* () {
      yield {
        type: "error",
        code: 500,
        message: "The model could not produce an answer.",
      };
    });
    const adapter = createPlatformChatRuntimeAdapter();
    const run = adapter.run({
      messages: [
        {
          id: "user-1",
          role: "user",
          content: [{ type: "text", text: "Question" }],
          createdAt: new Date(),
        },
      ],
      abortSignal: new AbortController().signal,
      unstable_threadId: "session-1",
      context: {},
    } as never) as AsyncGenerator<unknown>;

    await expect(run.next()).rejects.toThrow(
      "The model could not produce an answer.",
    );
  });
});
