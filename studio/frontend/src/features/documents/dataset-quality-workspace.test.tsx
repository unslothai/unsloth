import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { http, HttpResponse } from "msw";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { TooltipProvider } from "@/components/ui/tooltip";
import { platformTestServer } from "@/integrations/platform-backend/__tests__/test-server";
import type { PlatformDocument } from "@/integrations/platform-backend";

import { DatasetQualityWorkspace } from "./dataset-quality-workspace";

const documentRow: PlatformDocument = {
  id: "doc-1",
  datasetId: "dataset-1",
  name: "guide.pdf",
  thumbnail: null,
  sizeBytes: 2048,
  sourceType: "local",
  location: null,
  tokenCount: 64,
  chunkCount: 1,
  progress: 1,
  progressMessage: null,
  processDuration: 2,
  suffix: "pdf",
  run: "3",
  backendStatus: "3",
  status: "completed",
  parserId: "naive",
  chunkMethod: "naive",
  pipelineId: null,
  pipelineName: null,
  createdAt: null,
  updatedAt: null,
};

const chunkDto = {
  chunk_id: "chunk-1",
  kb_id: "dataset-1",
  doc_id: "doc-1",
  docnm_kwd: "guide.pdf",
  content_with_weight: "Kaynak metin",
  available_int: 1,
  positions: [[4, 0, 0, 0, 0]],
  similarity: 0.81,
};

function renderWorkspace(mode: "chunks" | "retrieval", onPreview = vi.fn()) {
  return {
    onPreview,
    ...render(
      <TooltipProvider>
        <DatasetQualityWorkspace
          mode={mode}
          datasetId="dataset-1"
          datasetName="Ürün belgeleri"
          documents={[documentRow]}
          preferredDocumentId="doc-1"
          onPreview={onPreview}
        />
      </TooltipProvider>,
    ),
  };
}

describe("Phase 6 Documents chunk and retrieval UI", () => {
  beforeEach(() => {
    vi.stubEnv("VITE_RAG_PLATFORM_BASE_URL", "http://platform.test");
    vi.stubEnv("VITE_RAG_PLATFORM_API_PREFIX", "/api/v1");
    vi.stubGlobal(
      "ResizeObserver",
      class {
        observe() {}
        unobserve() {}
        disconnect() {}
      },
    );
    vi.stubGlobal(
      "IntersectionObserver",
      class {
        observe() {}
        unobserve() {}
        disconnect() {}
      },
    );
    vi.spyOn(HTMLElement.prototype, "offsetHeight", "get").mockReturnValue(530);
    vi.spyOn(HTMLElement.prototype, "offsetWidth", "get").mockReturnValue(800);
  });

  it("loads virtualized chunks and structure graphs and opens a page citation", async () => {
    platformTestServer.use(
      http.get(
        "http://platform.test/api/v1/datasets/dataset-1/documents/doc-1/chunks",
        () =>
          HttpResponse.json({
            code: 0,
            data: { chunks: [chunkDto], total: 1, doc: {} },
          }),
      ),
      http.get(
        "http://platform.test/api/v1/datasets/dataset-1/documents/doc-1/structure/graph",
        () =>
          HttpResponse.json({
            code: 0,
            data: {
              templates: [
                {
                  template_id: "timeline-1",
                  template_name: "Zaman çizelgesi",
                  kind: "timeline",
                  entities: [{ id: "event-1", name: "Başlangıç" }],
                  relations: [],
                },
              ],
            },
          }),
      ),
    );

    const { onPreview } = renderWorkspace("chunks");
    expect(await screen.findByText("Kaynak metin")).toBeInTheDocument();
    expect(await screen.findByText("Başlangıç")).toBeInTheDocument();
    expect(screen.getByText("Toplam chunk")).toBeInTheDocument();
    expect(screen.getByText("Bu sayfada etkin")).toBeInTheDocument();
    expect(screen.getByText("Backend bağlı")).toBeInTheDocument();
    expect(
      screen.getByLabelText("Virtualized chunk listesi"),
    ).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Chunk kaynağını aç" }));
    expect(onPreview).toHaveBeenCalledWith(documentRow, 4);
  });

  it("runs scoped retrieval and renders normalized scores with source actions", async () => {
    let retrievalBody: unknown;
    platformTestServer.use(
      http.post(
        "http://platform.test/api/v1/retrieval",
        async ({ request }) => {
          retrievalBody = await request.json();
          return HttpResponse.json({
            code: 0,
            data: { total: 1, chunks: [chunkDto], doc_aggs: [] },
          });
        },
      ),
    );

    const { onPreview } = renderWorkspace("retrieval");
    fireEvent.change(screen.getByLabelText("Retrieval sorgusu"), {
      target: { value: "Ürün nedir?" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Retrieval çalıştır" }));

    expect(await screen.findByText("81.0%")).toBeInTheDocument();
    expect(screen.getByText("Güçlü eşleşme")).toBeInTheDocument();
    expect(screen.getByText("En iyi skor")).toBeInTheDocument();
    expect(screen.getByText("Tamamlandı")).toBeInTheDocument();
    expect(retrievalBody).toMatchObject({
      question: "Ürün nedir?",
      dataset_ids: ["dataset-1"],
      document_ids: [],
      top_k: 10,
      similarity_threshold: 0.2,
      vector_similarity_weight: 0.3,
      highlight: true,
    });
    fireEvent.click(screen.getByRole("button", { name: "Kaynağı aç" }));
    expect(onPreview).toHaveBeenCalledWith(documentRow, 4);
  });

  it("treats a successful zero-result retrieval as empty, not as an error", async () => {
    platformTestServer.use(
      http.post("http://platform.test/api/v1/retrieval", () =>
        HttpResponse.json({
          code: 0,
          data: { total: 0, chunks: [], doc_aggs: [] },
        }),
      ),
    );
    renderWorkspace("retrieval");
    fireEvent.change(screen.getByLabelText("Retrieval sorgusu"), {
      target: { value: "bulunmayan" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Retrieval çalıştır" }));

    expect(await screen.findByText("Sonuç bulunamadı")).toBeInTheDocument();
    expect(screen.queryByText("İstek tamamlanamadı")).not.toBeInTheDocument();
  });

  it("aborts in-flight document requests during cleanup", async () => {
    const signals: AbortSignal[] = [];
    platformTestServer.use(
      http.get(
        "http://platform.test/api/v1/datasets/dataset-1/documents/doc-1/chunks",
        async ({ request }) => {
          signals.push(request.signal);
          await new Promise<void>((resolve) =>
            request.signal.addEventListener("abort", () => resolve(), {
              once: true,
            }),
          );
          return HttpResponse.json({ code: 0, data: { chunks: [], total: 0 } });
        },
      ),
      http.get(
        "http://platform.test/api/v1/datasets/dataset-1/documents/doc-1/structure/graph",
        async ({ request }) => {
          signals.push(request.signal);
          await new Promise<void>((resolve) =>
            request.signal.addEventListener("abort", () => resolve(), {
              once: true,
            }),
          );
          return HttpResponse.json({ code: 0, data: { templates: [] } });
        },
      ),
    );
    const view = renderWorkspace("chunks");
    await waitFor(() => expect(signals).toHaveLength(2));
    view.unmount();
    await waitFor(() =>
      expect(signals.every((signal) => signal.aborted)).toBe(true),
    );
  });
});
