import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { TooltipProvider } from "@/components/ui/tooltip";

const mocks = vi.hoisted(() => ({
  setText: vi.fn(),
  send: vi.fn(),
  generateMindMap: vi.fn(),
  getRecommendations: vi.fn(),
  submitFeedback: vi.fn(),
}));

vi.mock("@assistant-ui/react", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@assistant-ui/react")>();
  const message = {
    id: "assistant-1",
    role: "assistant",
    content: [{ type: "text", text: "Answer" }],
    metadata: { custom: { platformMessageId: "turn-1" } },
  };
  const aui = {
    thread: () => ({
      getState: () => ({
        messages: [
          {
            id: "user-1",
            role: "user",
            content: [{ type: "text", text: "Question" }],
          },
          message,
        ],
      }),
    }),
    threadListItem: () => ({
      getState: () => ({ id: "session-1", remoteId: "session-1" }),
    }),
    composer: () => ({ setText: mocks.setText, send: mocks.send }),
  };
  return {
    ...actual,
    useAui: () => aui,
    useAuiState: (selector: (state: unknown) => unknown) =>
      selector({ threadListItem: { remoteId: "session-1" } }),
    useMessage: () => message,
  };
});

vi.mock("@/integrations/platform-backend", async (importOriginal) => {
  const actual =
    await importOriginal<typeof import("@/integrations/platform-backend")>();
  return { ...actual, isPlatformChatPersistenceEnabled: () => true };
});

vi.mock("@/features/chat/api/platform-chat-adapter", () => ({
  generatePlatformMindMapForChat: mocks.generateMindMap,
  getPlatformRecommendationsForChat: mocks.getRecommendations,
  submitPlatformMessageFeedbackForChat: mocks.submitFeedback,
}));

import {
  PlatformChatEnrichments,
  PlatformFeedbackActions,
} from "./platform-chat-enrichments";

describe("Rag Platform Phase 8 chat actions", () => {
  beforeEach(() => {
    mocks.setText.mockReset();
    mocks.send.mockReset();
    mocks.generateMindMap.mockReset();
    mocks.getRecommendations.mockReset();
    mocks.submitFeedback.mockReset();
    mocks.generateMindMap.mockResolvedValue({
      id: "root",
      label: "Root",
      children: [{ id: "child", label: "Child", children: [] }],
    });
    mocks.getRecommendations.mockResolvedValue(["Follow up?"]);
    mocks.submitFeedback.mockResolvedValue(undefined);
  });

  it("shows an accessible mindmap and keeps recommendation chips draft-only", async () => {
    render(
      <TooltipProvider>
        <PlatformChatEnrichments />
      </TooltipProvider>,
    );

    fireEvent.click(screen.getByRole("button", { name: "Mindmap" }));
    expect(await screen.findByRole("dialog")).toBeInTheDocument();
    expect(await screen.findByText("Child")).toBeInTheDocument();
    expect(mocks.generateMindMap).toHaveBeenCalledWith(
      "session-1",
      "Question",
      expect.any(AbortSignal),
    );
    Object.defineProperty(URL, "createObjectURL", {
      configurable: true,
      value: vi.fn(() => "blob:mindmap"),
    });
    Object.defineProperty(URL, "revokeObjectURL", {
      configurable: true,
      value: vi.fn(),
    });
    const anchorClick = vi
      .spyOn(HTMLAnchorElement.prototype, "click")
      .mockImplementation(() => undefined);
    fireEvent.click(screen.getByRole("button", { name: "JSON dışa aktar" }));
    expect(anchorClick).toHaveBeenCalledOnce();
    expect(URL.revokeObjectURL).toHaveBeenCalledWith("blob:mindmap");
    fireEvent.click(screen.getByRole("button", { name: "Close" }));

    fireEvent.click(screen.getByRole("button", { name: "Takip önerileri" }));
    const chip = await screen.findByRole("button", { name: "Follow up?" });
    fireEvent.click(chip);
    expect(mocks.setText).toHaveBeenCalledWith("Follow up?");
    expect(mocks.send).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: "Takip önerileri" }));
    expect(
      screen.queryByRole("button", { name: "Follow up?" }),
    ).not.toBeInTheDocument();
  });

  it("renders mindmap empty and permission-error states", async () => {
    mocks.generateMindMap.mockResolvedValueOnce(null);
    const { unmount } = render(
      <TooltipProvider>
        <PlatformChatEnrichments />
      </TooltipProvider>,
    );
    fireEvent.click(screen.getByRole("button", { name: "Mindmap" }));
    expect(
      await screen.findByText("Mindmap verisi bulunamadı."),
    ).toBeInTheDocument();
    unmount();

    mocks.generateMindMap.mockRejectedValueOnce(
      new Error("Bu işlem için izniniz yok."),
    );
    render(
      <TooltipProvider>
        <PlatformChatEnrichments />
      </TooltipProvider>,
    );
    fireEvent.click(screen.getByRole("button", { name: "Mindmap" }));
    expect(await screen.findByRole("alert")).toHaveTextContent(
      "Bu işlem için izniniz yok.",
    );
    expect(
      screen.getByRole("button", { name: "Yeniden dene" }),
    ).toBeInTheDocument();
  });

  it("persists positive and detailed negative feedback with the remote turn id", async () => {
    render(
      <TooltipProvider>
        <PlatformFeedbackActions />
      </TooltipProvider>,
    );

    fireEvent.click(screen.getByRole("button", { name: "Yanıt yararlı" }));
    await waitFor(() =>
      expect(mocks.submitFeedback).toHaveBeenCalledWith(
        "session-1",
        "turn-1",
        { thumbup: true },
        expect.any(AbortSignal),
      ),
    );

    fireEvent.click(
      screen.getByRole("button", { name: "Yanıt yararlı değil" }),
    );
    fireEvent.change(screen.getByPlaceholderText("İsteğe bağlı açıklama"), {
      target: { value: "Needs evidence" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Gönder" }));
    await waitFor(() =>
      expect(mocks.submitFeedback).toHaveBeenLastCalledWith(
        "session-1",
        "turn-1",
        { thumbup: false, feedback: "Needs evidence" },
        expect.any(AbortSignal),
      ),
    );
  });
});
