import { render, screen, within } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

const mocks = vi.hoisted(() => ({
  message: {
    id: "ui-message-1",
    createdAt: new Date("2026-08-15T20:00:00.000Z"),
    content: [],
    metadata: {
      custom: {
        responseDetails: {
          providerName: "Rag Platform",
          providerType: "platform",
          startedAt: Date.parse("2026-08-15T20:00:01.000Z"),
          finishedAt: Date.parse("2026-08-15T20:00:02.250Z"),
          durationMs: 1_250,
          sessionId: "session-1",
          toolCalls: [],
        },
        contextUsage: {
          promptTokens: 11,
          completionTokens: 7,
          totalTokens: 18,
        },
        platformChatId: "chat-1",
        platformSessionId: "session-1",
        platformMessageId: "backend-message-1",
        platformStreamCompleted: true,
        platformReference: {
          chunks: [{ id: "chunk-1" }, { id: "chunk-2" }],
          documentAggregations: [{ documentId: "document-1" }],
        },
        authorization: "Bearer must-not-render",
        authToken: "must-also-not-render",
      },
    },
  },
  timing: {
    firstTokenTime: 120,
    totalStreamTime: 1_250,
    tokenCount: 7,
    tokensPerSecond: 5.6,
    totalChunks: 4,
    toolCallCount: 0,
  },
}));

vi.mock("@assistant-ui/react", () => ({
  useMessage: () => mocks.message,
  useMessageTiming: () => mocks.timing,
}));

vi.mock("@/features/chat", () => ({
  customProviderDisplayName: (value: string | undefined) => value ?? null,
  parseExternalModelId: () => null,
  useChatPreferencesStore: (selector: (state: object) => unknown) =>
    selector({ showResponseModel: true }),
  useChatRuntimeStore: (selector: (state: object) => unknown) =>
    selector({ models: [] }),
  useExternalProvidersStore: (selector: (state: object) => unknown) =>
    selector({ providers: [] }),
}));

import { MessageResponseDetailsSheet } from "./message-response-details-sheet";

describe("MessageResponseDetailsSheet", () => {
  it("shows all typed platform details and redacts sensitive raw metadata", () => {
    render(
      <MessageResponseDetailsSheet open={true} onOpenChange={vi.fn()} />,
    );

    expect(screen.getByText("Rag Platform response")).toBeInTheDocument();
    expect(screen.getAllByText("Rag Platform").length).toBeGreaterThan(0);
    expect(screen.getByText("chat-1")).toBeInTheDocument();
    expect(screen.getAllByText("session-1").length).toBeGreaterThan(0);
    expect(screen.getAllByText("backend-message-1").length).toBeGreaterThan(0);
    expect(screen.getByText("Completed")).toBeInTheDocument();
    expect(screen.getByText("Reference chunks")).toBeInTheDocument();
    expect(screen.getByText("Reference documents")).toBeInTheDocument();

    const metadata = screen.getByLabelText("Response metadata");
    expect(within(metadata).getByText(/"promptTokens": 11/)).toBeInTheDocument();
    expect(within(metadata).getByText(/"authorization": "\[REDACTED\]"/)).toBeInTheDocument();
    expect(metadata).not.toHaveTextContent("must-not-render");
    expect(metadata).not.toHaveTextContent("must-also-not-render");
  });
});
