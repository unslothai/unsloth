import type { PreviewTarget } from "@/features/rag/api/rag-api";
import { PreviewTextView } from "@/features/rag/components/preview-text-view";
import { render, screen } from "@testing-library/react";
import React from "react";
import { describe, expect, it, vi } from "vitest";

vi.mock("@/features/auth", () => ({
  authFetch: vi.fn(),
}));

function target(overrides: Partial<PreviewTarget> = {}): PreviewTarget {
  return {
    documentId: "doc-abc",
    filename: "notes.txt",
    contentType: "text/plain",
    mediaKind: "text",
    byteSize: 100,
    status: "completed",
    kbId: "kb-1",
    threadId: null,
    chunkId: "chunk-1",
    chunkIndex: 0,
    targetPage: 2,
    snippet: "alpha\nhighlighted line\nomega",
    kind: "text",
    imageUrl: null,
    sourcePageIndex: null,
    pageCharStart: null,
    pageCharEnd: null,
    lineStart: null,
    lineEnd: null,
    pdfRegions: [],
    ...overrides,
  };
}

describe("PreviewTextView locator highlight fallback", () => {
  it("emphasizes the source excerpt when nullable locators are present", () => {
    render(
      React.createElement(PreviewTextView, {
        target: target({
          sourcePageIndex: 1,
          pageCharStart: 6,
          pageCharEnd: 22,
          lineStart: 2,
          lineEnd: 2,
        }),
      }),
    );

    expect(screen.getByText(/highlighted source excerpt/i)).toBeInTheDocument();
    expect(document.querySelector("mark")?.textContent).toContain(
      "highlighted line",
    );
  });

  it("keeps the source excerpt visible when locators are missing", () => {
    render(React.createElement(PreviewTextView, { target: target() }));

    expect(screen.getByText(/source excerpt/i)).toBeInTheDocument();
    expect(document.querySelector("mark")).toBeNull();
    expect(screen.getByText(/highlighted line/i)).toBeInTheDocument();
  });
});
