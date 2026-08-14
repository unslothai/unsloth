import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const mocks = vi.hoisted(() => ({
  listKnowledgeBases: vi.fn(),
  listDocuments: vi.fn(),
  upload: vi.fn(),
  parse: vi.fn(),
  stop: vi.fn(),
  remove: vi.fn(),
  rename: vi.fn(),
}));

vi.mock("@/features/rag/api/platform-dataset-adapter", () => ({
  listAllKnowledgeBases: mocks.listKnowledgeBases,
}));

vi.mock("@/integrations/platform-backend", async (importOriginal) => {
  const original = await importOriginal<
    typeof import("@/integrations/platform-backend")
  >();
  return {
    ...original,
    listDatasetDocuments: mocks.listDocuments,
    uploadDatasetDocuments: mocks.upload,
    parseDatasetDocuments: mocks.parse,
    stopDatasetDocuments: mocks.stop,
    deleteDatasetDocuments: mocks.remove,
    updateDatasetDocument: mocks.rename,
  };
});

import { nextDocumentPollDelay, useDocumentLibrary } from "./use-document-library";

const runningDocument = {
  id: "doc-1",
  datasetId: "dataset-1",
  name: "guide.pdf",
  thumbnail: null,
  sizeBytes: 10,
  sourceType: "local",
  location: null,
  tokenCount: 1,
  chunkCount: 0,
  progress: 0.25,
  progressMessage: "Parsing",
  processDuration: 0,
  suffix: "pdf",
  run: "1",
  backendStatus: "1",
  status: "running" as const,
  parserId: "naive",
  chunkMethod: "naive",
  pipelineId: null,
  pipelineName: null,
  createdAt: null,
  updatedAt: null,
};

function Harness() {
  const library = useDocumentLibrary();
  return <output data-testid="state">{library.documents[0]?.status ?? "empty"}</output>;
}

function PaginationHarness() {
  const library = useDocumentLibrary();
  return (
    <>
      <output data-testid="pagination-state">
        {library.page}/{library.totalPages}/{library.totalDocuments}
      </output>
      <button type="button" onClick={() => library.setPage(library.page + 1)}>
        next
      </button>
    </>
  );
}

describe("Phase 5 document polling state machine", () => {
  beforeEach(() => {
    mocks.listKnowledgeBases.mockResolvedValue([
      {
        id: "dataset-1",
        name: "Product docs",
        description: null,
        createdAt: null,
        updatedAt: null,
        documentCount: 1,
        embeddingModel: "embed",
        permission: "me",
        chunkMethod: "naive",
        pipelineId: null,
      },
    ]);
    mocks.listDocuments.mockReset();
    Object.defineProperty(document, "visibilityState", {
      configurable: true,
      value: "visible",
    });
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("uses bounded exponential backoff and stops after a terminal document state", async () => {
    vi.useFakeTimers();
    expect([1_500, 3_000, 6_000, 12_000, 15_000]).toEqual(
      [750, 1_500, 3_000, 6_000, 12_000].map(nextDocumentPollDelay),
    );
    mocks.listDocuments
      .mockResolvedValueOnce({ items: [runningDocument], total: 1 })
      .mockResolvedValueOnce({
        items: [{ ...runningDocument, status: "completed", run: "3", progress: 1 }],
        total: 1,
      });

    await act(async () => {
      render(<Harness />);
      await Promise.resolve();
      await Promise.resolve();
    });
    expect(screen.getByTestId("state")).toHaveTextContent("running");
    await act(async () => {
      await vi.advanceTimersByTimeAsync(1_500);
    });
    expect(screen.getByTestId("state")).toHaveTextContent("completed");
    await act(async () => {
      await vi.advanceTimersByTimeAsync(30_000);
    });
    expect(mocks.listDocuments).toHaveBeenCalledTimes(2);
  });

  it("pauses while hidden, resumes when visible and aborts the active request on unmount", async () => {
    vi.useFakeTimers();
    let activeSignal: AbortSignal | undefined;
    mocks.listDocuments
      .mockResolvedValueOnce({ items: [runningDocument], total: 1 })
      .mockImplementationOnce((_datasetId: string, _options: unknown, signal: AbortSignal) => {
        activeSignal = signal;
        return new Promise((_, reject) => {
          signal.addEventListener("abort", () => reject(new DOMException("aborted", "AbortError")));
        });
      });

    let view!: ReturnType<typeof render>;
    await act(async () => {
      view = render(<Harness />);
      await Promise.resolve();
      await Promise.resolve();
    });
    expect(screen.getByTestId("state")).toHaveTextContent("running");
    Object.defineProperty(document, "visibilityState", { configurable: true, value: "hidden" });
    act(() => document.dispatchEvent(new Event("visibilitychange")));
    await act(async () => vi.advanceTimersByTimeAsync(30_000));
    expect(mocks.listDocuments).toHaveBeenCalledTimes(1);

    Object.defineProperty(document, "visibilityState", { configurable: true, value: "visible" });
    act(() => document.dispatchEvent(new Event("visibilitychange")));
    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
    });
    expect(mocks.listDocuments).toHaveBeenCalledTimes(2);
    expect(activeSignal?.aborted).toBe(false);
    view.unmount();
    expect(activeSignal?.aborted).toBe(true);
  });

  it("requests and exposes server-backed pagination", async () => {
    mocks.listDocuments.mockResolvedValue({ items: [], total: 27 });

    render(<PaginationHarness />);

    await waitFor(() => {
      expect(mocks.listDocuments).toHaveBeenCalledWith(
        "dataset-1",
        { page: 1, pageSize: 10, keywords: "" },
        expect.any(AbortSignal),
      );
    });
    expect(screen.getByTestId("pagination-state")).toHaveTextContent("1/3/27");

    fireEvent.click(screen.getByRole("button", { name: "next" }));
    await waitFor(() => {
      expect(mocks.listDocuments).toHaveBeenLastCalledWith(
        "dataset-1",
        { page: 2, pageSize: 10, keywords: "" },
        expect.any(AbortSignal),
      );
    });
    expect(screen.getByTestId("pagination-state")).toHaveTextContent("2/3/27");
  });
});
