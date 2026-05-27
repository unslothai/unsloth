import { fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import React from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";
const { mockOpen } = vi.hoisted(() => ({ mockOpen: vi.fn() }));

vi.mock("@/features/rag/stores/preview-store", () => ({
  usePreviewStore: (
    selector?: (state: { open: typeof mockOpen }) => unknown,
  ) => {
    const state = { open: mockOpen };
    return typeof selector === "function" ? selector(state) : state;
  },
}));

import { DocumentSourceBadge } from "@/components/assistant-ui/sources";

function source(overrides: Record<string, unknown> = {}) {
  return {
    kind: "document" as const,
    chunkId: "2",
    documentId: "doc-abc",
    backendChunkId: "chunk-xyz",
    filename: "report.pdf",
    page: "7",
    score: "0.85",
    text: "The margin rose to 18%.",
    ...overrides,
  };
}

beforeEach(() => {
  mockOpen.mockClear();
});

describe("DocumentSourceBadge preview routing", () => {
  it("opens preview with durable document and backend chunk IDs on click", async () => {
    render(React.createElement(DocumentSourceBadge, { source: source() }));

    await userEvent.click(
      screen.getByRole("button", { name: /open preview/i }),
    );

    expect(mockOpen).toHaveBeenCalledWith({
      documentId: "doc-abc",
      backendChunkId: "chunk-xyz",
    });
  });

  it("opens preview from Enter and Space", () => {
    render(React.createElement(DocumentSourceBadge, { source: source() }));
    const badge = screen.getByRole("button", { name: /open preview/i });

    fireEvent.keyDown(badge, { key: "Enter" });
    fireEvent.keyDown(badge, { key: " " });

    expect(mockOpen).toHaveBeenCalledTimes(2);
  });

  it("legacy source without durable IDs remains hover-only", async () => {
    render(
      React.createElement(DocumentSourceBadge, {
        source: source({ documentId: null, backendChunkId: null }),
      }),
    );

    expect(screen.queryByRole("button", { name: /open preview/i })).toBeNull();
    await userEvent.click(screen.getByText("[2]"));
    expect(mockOpen).not.toHaveBeenCalled();
  });

  it("applies brand-aligned interactive styling when preview is clickable", () => {
    render(React.createElement(DocumentSourceBadge, { source: source() }));
    const badge = screen.getByRole("button", { name: /open preview/i });

    expect(badge).toHaveClass("cursor-pointer");
    expect(badge).toHaveClass("hover:bg-chat-icon-bg-hover!");
    expect(badge).toHaveClass("focus-visible:ring-ring/50");
  });
});
