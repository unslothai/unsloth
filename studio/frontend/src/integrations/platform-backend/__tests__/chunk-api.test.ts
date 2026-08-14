import { http, HttpResponse } from "msw";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  createDocumentChunk,
  deleteDocumentChunks,
  deleteDocumentStructureGraph,
  getDocumentChunk,
  getDocumentStructureGraph,
  listDocumentChunks,
  listDocumentChunksCompatibility,
  parseDatasetChunksCompatibility,
  retrievePlatformChunks,
  searchDatasetChunksCompatibility,
  setDocumentChunksEnabled,
  stopDatasetChunksCompatibility,
  updateDocumentChunk,
  updateDocumentChunkCompatibility,
} from "../chunk-api";
import {
  chunkPreviewDocument,
  mapPlatformChunk,
  mapPlatformRetrieval,
  mapPlatformStructureGraph,
} from "../chunk-types";
import { platformTestServer } from "./test-server";

const chunkDto = {
  chunk_id: "chunk-1",
  kb_id: "dataset-1",
  doc_id: "doc-1",
  docnm_kwd: "guide.pdf",
  content_with_weight: "Kaynak metin",
  important_kwd: ["rag", "retrieval"],
  question_kwd: ["Nasıl çalışır?"],
  available_int: 1,
  positions: [[7, 0, 10, 20, 30]],
  vector_similarity: 0.72,
  term_similarity: 0.41,
  rerank_score: 86,
  doc_type_kwd: "pdf",
};

function ok(data: unknown) {
  return HttpResponse.json({ code: 0, data });
}

describe("Rag Platform Phase 6 chunk and retrieval contracts", () => {
  beforeEach(() => {
    vi.stubEnv("VITE_RAG_PLATFORM_BASE_URL", "http://platform.test");
    vi.stubEnv("VITE_RAG_PLATFORM_API_PREFIX", "/api/v1");
  });

  afterEach(() => vi.unstubAllEnvs());

  it("maps backend aliases, scores, citations and graph rows defensively", () => {
    const chunk = mapPlatformChunk(chunkDto);
    expect(chunk).toMatchObject({
      id: "chunk-1",
      datasetId: "dataset-1",
      documentId: "doc-1",
      pageNumber: 7,
      enabled: true,
      normalizedScore: 0.86,
      importantKeywords: ["rag", "retrieval"],
    });
    expect(chunkPreviewDocument(chunk, "fallback")).toMatchObject({
      id: "doc-1",
      datasetId: "dataset-1",
      name: "guide.pdf",
      suffix: "pdf",
    });
    expect(
      mapPlatformRetrieval({ total: 1, chunks: [chunkDto], doc_aggs: [{}] })
        .documentAggregations,
    ).toHaveLength(1);
    expect(
      mapPlatformStructureGraph({
        templates: [
          {
            template_id: "timeline-1",
            template_name: "Zaman çizelgesi",
            kind: "timeline",
            entities: [{ entity_id: "e-1", label: "Başlangıç" }],
            relations: [{ from: "e-1", to: "e-2", label: "önce" }],
          },
        ],
      }).templates[0],
    ).toMatchObject({
      id: "timeline-1",
      entities: [{ id: "e-1", name: "Başlangıç" }],
      relations: [{ source: "e-1", target: "e-2", description: "önce" }],
    });
  });

  it("uses canonical list, detail and create routes with exact request fields", async () => {
    const calls: Array<{ method: string; path: string; body?: unknown }> = [];
    platformTestServer.use(
      http.get(
        "http://platform.test/api/v1/datasets/dataset-1/documents/doc-1/chunks",
        ({ request }) => {
          expect(Object.fromEntries(new URL(request.url).searchParams)).toEqual(
            {
              page: "2",
              page_size: "25",
              keywords: "retrieval",
              available: "false",
            },
          );
          return ok({ chunks: [chunkDto], doc: { id: "doc-1" }, total: 1 });
        },
      ),
      http.get(
        "http://platform.test/api/v1/datasets/dataset-1/documents/doc-1/chunks/chunk-1",
        ({ request }) => {
          calls.push({
            method: request.method,
            path: new URL(request.url).pathname,
          });
          return ok(chunkDto);
        },
      ),
      http.post(
        "http://platform.test/api/v1/datasets/dataset-1/documents/doc-1/chunks",
        async ({ request }) => {
          calls.push({
            method: request.method,
            path: new URL(request.url).pathname,
            body: await request.json(),
          });
          return ok({ chunk: chunkDto });
        },
      ),
    );

    const listed = await listDocumentChunks("dataset-1", "doc-1", {
      page: 2,
      pageSize: 25,
      keywords: " retrieval ",
      available: false,
    });
    await expect(
      getDocumentChunk("dataset-1", "doc-1", "chunk-1"),
    ).resolves.toMatchObject({
      id: "chunk-1",
    });
    await expect(
      createDocumentChunk("dataset-1", "doc-1", {
        content: "Kaynak metin",
        importantKeywords: ["rag"],
        questions: ["Nasıl çalışır?"],
      }),
    ).resolves.toMatchObject({ id: "chunk-1" });

    expect(listed).toMatchObject({ total: 1, document: { id: "doc-1" } });
    expect(calls[1]?.body).toEqual({
      content: "Kaynak metin",
      important_keywords: ["rag"],
      questions: ["Nasıl çalışır?"],
    });
  });

  it("keeps item update, bulk availability and deletion bodies distinct", async () => {
    const calls: Array<{ method: string; path: string; body: unknown }> = [];
    const capture = async (request: Request) => {
      calls.push({
        method: request.method,
        path: new URL(request.url).pathname,
        body: await request.json(),
      });
      return ok(true);
    };
    platformTestServer.use(
      http.patch(
        "http://platform.test/api/v1/datasets/dataset-1/documents/doc-1/chunks/chunk-1",
        ({ request }) => capture(request),
      ),
      http.patch(
        "http://platform.test/api/v1/datasets/dataset-1/documents/doc-1/chunks",
        ({ request }) => capture(request),
      ),
      http.delete(
        "http://platform.test/api/v1/datasets/dataset-1/documents/doc-1/chunks",
        ({ request }) => capture(request),
      ),
    );

    await updateDocumentChunk("dataset-1", "doc-1", "chunk-1", {
      content: "Yeni içerik",
      enabled: false,
    });
    await setDocumentChunksEnabled("dataset-1", "doc-1", ["chunk-1"], false);
    await deleteDocumentChunks("dataset-1", "doc-1", ["chunk-1"]);

    expect(calls.map((call) => call.body)).toEqual([
      { content: "Yeni içerik", available: false },
      { chunk_ids: ["chunk-1"], available: false },
      { chunk_ids: ["chunk-1"] },
    ]);
  });

  it("sends retrieval tuning fields and structure graph contracts exactly", async () => {
    const calls: Array<{ path: string; method: string; body: unknown }> = [];
    platformTestServer.use(
      http.post(
        "http://platform.test/api/v1/retrieval",
        async ({ request }) => {
          calls.push({
            path: new URL(request.url).pathname,
            method: request.method,
            body: await request.json(),
          });
          return ok({ total: 1, chunks: [chunkDto], doc_aggs: [] });
        },
      ),
      http.get(
        "http://platform.test/api/v1/datasets/dataset-1/documents/doc-1/structure/graph",
        ({ request }) => {
          expect(new URL(request.url).searchParams.get("keywords")).toBe(
            "ürün",
          );
          return ok({ templates: [] });
        },
      ),
      http.delete(
        "http://platform.test/api/v1/datasets/dataset-1/documents/doc-1/structure/graph",
        async ({ request }) => {
          calls.push({
            path: new URL(request.url).pathname,
            method: request.method,
            body: await request.json(),
          });
          return ok({ deleted: 3 });
        },
      ),
    );

    await expect(
      retrievePlatformChunks({
        datasetIds: ["dataset-1"],
        documentIds: ["doc-1"],
        question: " ürün nedir? ",
        page: 2,
        pageSize: 12,
        topK: 30,
        similarityThreshold: 0.35,
        vectorSimilarityWeight: 0.65,
        highlight: false,
        rerankId: " rerank-1 ",
      }),
    ).resolves.toMatchObject({ total: 1 });
    await expect(
      getDocumentStructureGraph("dataset-1", "doc-1", " ürün "),
    ).resolves.toEqual({ templates: [] });
    await expect(
      deleteDocumentStructureGraph("dataset-1", "doc-1", "timeline-1"),
    ).resolves.toBe(3);

    expect(calls[0]?.body).toEqual({
      question: "ürün nedir?",
      dataset_ids: ["dataset-1"],
      document_ids: ["doc-1"],
      page: 2,
      page_size: 12,
      top_k: 30,
      similarity_threshold: 0.35,
      vector_similarity_weight: 0.65,
      highlight: false,
      rerank_id: "rerank-1",
    });
    expect(calls[1]?.body).toEqual({ template_id: "timeline-1" });
  });

  it("contract-tests API-only compatibility aliases without exposing them as UI paths", async () => {
    const calls: Array<{ path: string; method: string; body: unknown }> = [];
    const capture = async (request: Request, data: unknown = true) => {
      calls.push({
        path: new URL(request.url).pathname,
        method: request.method,
        body: await request.json(),
      });
      return ok(data);
    };
    platformTestServer.use(
      http.post("http://platform.test/api/v1/chunk/list", ({ request }) =>
        capture(request, { chunks: [chunkDto], total: 1, doc: {} }),
      ),
      http.put(
        "http://platform.test/api/v1/datasets/dataset-1/documents/doc-1/chunks/chunk-1",
        ({ request }) => capture(request),
      ),
      http.post(
        "http://platform.test/api/v1/datasets/dataset-1/chunks",
        ({ request }) => capture(request),
      ),
      http.delete(
        "http://platform.test/api/v1/datasets/dataset-1/chunks",
        ({ request }) => capture(request),
      ),
      http.post(
        "http://platform.test/api/v1/datasets/dataset-1/search",
        ({ request }) =>
          capture(request, { chunks: [chunkDto], total: 1, doc_aggs: [] }),
      ),
    );

    await listDocumentChunksCompatibility("dataset-1", "doc-1", {
      page: 3,
      pageSize: 15,
    });
    await updateDocumentChunkCompatibility("dataset-1", "doc-1", "chunk-1", {
      content: "Uyumluluk",
    });
    await parseDatasetChunksCompatibility("dataset-1", ["doc-1"]);
    await stopDatasetChunksCompatibility("dataset-1", ["doc-1"]);
    await searchDatasetChunksCompatibility("dataset-1", {
      question: "test",
      documentIds: ["doc-1"],
      topK: 5,
    });

    expect(calls.map((call) => [call.method, call.path, call.body])).toEqual([
      [
        "POST",
        "/api/v1/chunk/list",
        { dataset_id: "dataset-1", doc_id: "doc-1", page: 3, size: 15 },
      ],
      [
        "PUT",
        "/api/v1/datasets/dataset-1/documents/doc-1/chunks/chunk-1",
        { content: "Uyumluluk" },
      ],
      [
        "POST",
        "/api/v1/datasets/dataset-1/chunks",
        { document_ids: ["doc-1"] },
      ],
      [
        "DELETE",
        "/api/v1/datasets/dataset-1/chunks",
        { document_ids: ["doc-1"] },
      ],
      [
        "POST",
        "/api/v1/datasets/dataset-1/search",
        {
          question: "test",
          page: 1,
          size: 5,
          doc_ids: ["doc-1"],
          top_k: 5,
          similarity_threshold: 0.2,
          vector_similarity_weight: 0.3,
        },
      ],
    ]);
  });
});
