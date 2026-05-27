import type { PreviewTarget } from "@/features/rag/api/rag-api";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import React from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const { mockFetchPreviewTarget, mockFetchPreviewFileBlob } = vi.hoisted(() => ({
  mockFetchPreviewTarget:
    vi.fn<
      (documentId: string, chunkId?: string | null) => Promise<PreviewTarget>
    >(),
  mockFetchPreviewFileBlob:
    vi.fn<(documentId: string, signal?: AbortSignal) => Promise<Blob>>(),
}));

vi.mock("@/features/rag/api/rag-api", async (importOriginal) => {
  const original =
    await importOriginal<typeof import("@/features/rag/api/rag-api")>();
  return {
    ...original,
    fetchPreviewTarget: mockFetchPreviewTarget,
    fetchPreviewFileBlob: mockFetchPreviewFileBlob,
  };
});

vi.mock("react-pdf", () => ({
  Document: ({ children }: { children: React.ReactNode }) =>
    React.createElement("div", { "data-testid": "pdf-document" }, children),
  Page: () => React.createElement("div", { "data-testid": "pdf-page" }),
  pdfjs: { GlobalWorkerOptions: { workerSrc: "" } },
}));

import { PreviewPanel } from "@/features/rag/components/preview-panel";
import { usePreviewStore } from "@/features/rag/stores/preview-store";

function target(overrides: Partial<PreviewTarget> = {}): PreviewTarget {
  return {
    documentId: "doc-abc",
    filename: "report.html",
    contentType: "text/plain",
    mediaKind: "html",
    byteSize: 100,
    status: "completed",
    kbId: "kb-1",
    threadId: null,
    chunkId: "chunk-1",
    chunkIndex: 0,
    targetPage: 1,
    snippet: "safe extracted text",
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

beforeEach(() => {
  mockFetchPreviewTarget.mockReset();
  mockFetchPreviewFileBlob.mockReset();
  usePreviewStore.getState().close();
});

afterEach(() => {
  usePreviewStore.getState().close();
});

describe("preview a11y hardening", () => {
  it("Escape closes the preview and restores focus to the opener", async () => {
    const opener = document.createElement("button");
    opener.textContent = "Open preview";
    document.body.appendChild(opener);
    opener.focus();
    mockFetchPreviewTarget.mockResolvedValue(target());

    await usePreviewStore.getState().open({ documentId: "doc-abc" });
    render(React.createElement(PreviewPanel, { open: true }));
    expect(
      screen.getByRole("region", { name: /document preview/i }),
    ).toBeInTheDocument();

    fireEvent.keyDown(document, { key: "Escape" });

    await waitFor(() => {
      expect(usePreviewStore.getState().status).toBe("idle");
    });
    expect(document.activeElement).toBe(opener);
    opener.remove();
  });
});
