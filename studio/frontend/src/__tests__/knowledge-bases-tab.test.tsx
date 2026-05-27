import { KnowledgeBasesTab } from "@/features/settings/tabs/knowledge-bases-tab";
import { render, screen } from "@testing-library/react";
import React from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";

const { mockUsePreviewStore } = vi.hoisted(() => {
  const state = {
    target: null as unknown,
    status: "idle",
    close: vi.fn(),
  };
  const fn = vi.fn((selector?: (s: typeof state) => unknown) => {
    if (typeof selector === "function") return selector(state);
    return state;
  }) as ReturnType<typeof vi.fn> & { __state: typeof state };
  fn.__state = state;
  return { mockUsePreviewStore: fn };
});

vi.mock("@/features/rag/stores/preview-store", () => ({
  usePreviewStore: mockUsePreviewStore,
}));

vi.mock("@/features/rag/components/kb-list", () => ({
  KBList: () => React.createElement("div", { "data-testid": "kb-list" }),
}));

vi.mock("@/features/rag/components/kb-detail-panel", () => ({
  KBDetailPanel: () =>
    React.createElement("div", { "data-testid": "kb-detail-panel" }),
}));

vi.mock("@/features/rag/components/preview-panel", () => ({
  PreviewPanel: ({ open }: { open: boolean }) =>
    React.createElement("div", {
      "data-testid": "settings-preview-panel",
      "data-open": String(open),
    }),
}));

vi.mock("@/features/rag/components/thread-index-list", () => ({
  ThreadIndexList: () =>
    React.createElement("div", { "data-testid": "thread-index-list" }),
}));

vi.mock("@/features/rag/components/rag-defaults-section", () => ({
  RagDefaultsSection: () =>
    React.createElement("div", { "data-testid": "rag-defaults-section" }),
}));

beforeEach(() => {
  mockUsePreviewStore.__state.target = null;
  mockUsePreviewStore.__state.status = "idle";
  mockUsePreviewStore.mockImplementation(
    (selector?: (s: typeof mockUsePreviewStore.__state) => unknown) => {
      if (typeof selector === "function") {
        return selector(mockUsePreviewStore.__state);
      }
      return mockUsePreviewStore.__state;
    },
  );
});

describe("KnowledgeBasesTab preview host", () => {
  it("renders a preview panel when the preview store is active", () => {
    mockUsePreviewStore.__state.target = {
      documentId: "doc-abc",
      filename: "report.pdf",
    };
    mockUsePreviewStore.__state.status = "ready";

    render(React.createElement(KnowledgeBasesTab));

    const panel = screen.getByTestId("settings-preview-panel");
    expect(panel.getAttribute("data-open")).toBe("true");
  });
});
