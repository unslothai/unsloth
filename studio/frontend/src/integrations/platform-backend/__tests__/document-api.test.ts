import { http, HttpResponse } from "msw";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  cancelPlatformTask,
  createGenericDocument,
  deleteDatasetDocuments,
  deleteGenericDocument,
  downloadDatasetDocument,
  fetchDocumentArtifact,
  fetchDocumentImage,
  fetchDocumentPreview,
  getGenericDocument,
  ingestGenericDocuments,
  inspectDocumentUploads,
  listDatasetDocuments,
  listDocumentThumbnails,
  parseDatasetDocuments,
  stopPlatformTask,
  stopDatasetDocuments,
  updateDatasetDocument,
  updateGenericDocument,
  uploadDatasetDocuments,
} from "../document-api";
import {
  isInlineSafeContentType,
  mapPlatformDocument,
  platformDocumentStatus,
  validatePlatformDocumentFile,
} from "../document-types";
import { platformTestServer } from "./test-server";

const dto = {
  id: "doc-1",
  dataset_id: "dataset-1",
  name: "guide.pdf",
  size: 2048,
  token_count: 64,
  chunk_count: 3,
  progress: 0.5,
  progress_msg: "Parsing",
  suffix: "pdf",
  run: "1",
  status: "1",
  parser_id: "naive",
  create_time: 1_700_000_000_000,
};

describe("Rag Platform Phase 5 document contracts", () => {
  beforeEach(() => {
    vi.stubEnv("VITE_RAG_PLATFORM_BASE_URL", "http://platform.test");
    vi.stubEnv("VITE_RAG_PLATFORM_API_PREFIX", "/api/v1");
  });

  afterEach(() => vi.unstubAllEnvs());

  it("maps list status, counters and active-runtime pagination exactly", async () => {
    platformTestServer.use(
      http.get(
        "http://platform.test/api/v1/datasets/dataset-1/documents",
        ({ request }) => {
          expect(Object.fromEntries(new URL(request.url).searchParams)).toEqual({
            page: "3",
            page_size: "20",
            orderby: "update_time",
            desc: "true",
            keywords: "guide",
          });
          return HttpResponse.json({ code: 0, data: { total: 1, docs: [dto] } });
        },
      ),
    );

    const result = await listDatasetDocuments("dataset-1", {
      page: 3,
      pageSize: 20,
      keywords: " guide ",
    });
    expect(result.total).toBe(1);
    expect(result.items[0]).toMatchObject({
      id: "doc-1",
      name: "guide.pdf",
      status: "running",
      tokenCount: 64,
      chunkCount: 3,
      progress: 0.5,
    });
  });

  it("uploads multiple files with the multipart file key and retains partial success", async () => {
    platformTestServer.use(
      http.post(
        "http://platform.test/api/v1/datasets/dataset-1/documents",
        async ({ request }) => {
          expect(new URL(request.url).searchParams.get("type")).toBe("local");
          const form = await request.formData();
          expect(form.getAll("file")).toHaveLength(2);
          return HttpResponse.json({
            code: 100,
            data: [dto],
            message: "bad.docx: corrupted",
          });
        },
      ),
    );

    const result = await uploadDatasetDocuments("dataset-1", [
      new File(["pdf"], "guide.pdf", { type: "application/pdf" }),
      new File(["docx"], "bad.docx", {
        type: "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
      }),
    ]);
    expect(result.documents).toHaveLength(1);
    expect(result.partialFailure).toContain("corrupted");
  });

  it("uses separate patch, parse, stop and bulk-delete bodies", async () => {
    const calls: Array<{ path: string; method: string; body: unknown }> = [];
    const capture = async (request: Request) => {
      calls.push({
        path: new URL(request.url).pathname,
        method: request.method,
        body: await request.json(),
      });
      return HttpResponse.json({ code: 0, data: true });
    };
    platformTestServer.use(
      http.patch(
        "http://platform.test/api/v1/datasets/dataset-1/documents/doc-1",
        ({ request }) => capture(request),
      ),
      http.post(
        "http://platform.test/api/v1/datasets/dataset-1/documents/parse",
        ({ request }) => capture(request),
      ),
      http.post(
        "http://platform.test/api/v1/datasets/dataset-1/documents/stop",
        ({ request }) => capture(request),
      ),
      http.delete(
        "http://platform.test/api/v1/datasets/dataset-1/documents",
        ({ request }) => capture(request),
      ),
    );

    await updateDatasetDocument("dataset-1", "doc-1", { name: "renamed.pdf" });
    await parseDatasetDocuments("dataset-1", ["doc-1"]);
    await stopDatasetDocuments("dataset-1", ["doc-1"]);
    await deleteDatasetDocuments("dataset-1", ["doc-1"]);

    expect(calls.map((call) => call.body)).toEqual([
      { name: "renamed.pdf" },
      { document_ids: ["doc-1"] },
      { document_ids: ["doc-1"] },
      { ids: ["doc-1"] },
    ]);
  });

  it("returns authenticated preview/download blobs and records safe headers", async () => {
    platformTestServer.use(
      http.get(
        "http://platform.test/api/v1/documents/doc-1/preview",
        () =>
          new HttpResponse("preview", {
            headers: {
              "Content-Type": "text/plain; charset=utf-8",
              "Content-Disposition": 'inline; filename="guide.txt"',
            },
          }),
      ),
      http.get(
        "http://platform.test/api/v1/datasets/dataset-1/documents/doc-1",
        () =>
          new HttpResponse("download", {
            headers: {
              "Content-Type": "application/pdf",
              "Content-Disposition": 'attachment; filename="guide.pdf"',
              "X-Content-Type-Options": "nosniff",
            },
          }),
      ),
    );

    const preview = await fetchDocumentPreview("doc-1");
    expect(preview.contentType).toBe("text/plain; charset=utf-8");
    expect(typeof preview.blob.arrayBuffer).toBe("function");
    const download = await downloadDatasetDocument("dataset-1", "doc-1");
    expect(download.disposition).toBe('attachment; filename="guide.pdf"');
    expect(typeof download.blob.arrayBuffer).toBe("function");
  });

  it("classifies terminal states and rejects unsafe, empty, oversized and unsupported files", () => {
    expect(platformDocumentStatus("0")).toBe("pending");
    expect(platformDocumentStatus("1")).toBe("running");
    expect(platformDocumentStatus("2")).toBe("cancelled");
    expect(platformDocumentStatus("3")).toBe("completed");
    expect(platformDocumentStatus("4")).toBe("failed");
    expect(mapPlatformDocument(dto).status).toBe("running");

    expect(validatePlatformDocumentFile(new File([], "empty.pdf"))?.reason).toBe("empty");
    expect(
      validatePlatformDocumentFile(
        new File(["x"], "payload.exe", { type: "application/octet-stream" }),
      )?.reason,
    ).toBe("unsupported");
    const huge = new File(["x"], "huge.pdf");
    Object.defineProperty(huge, "size", { value: 129 * 1024 * 1024 });
    expect(validatePlatformDocumentFile(huge)?.reason).toBe("too-large");
    expect(isInlineSafeContentType("text/html")).toBe(false);
    expect(isInlineSafeContentType("image/png")).toBe(true);
  });

  it("keeps generic CRUD/ingest and dataset ownership contracts separate", async () => {
    const calls: Array<{ method: string; path: string; body: unknown }> = [];
    const generic = {
      id: "generic-1",
      name: "note.txt",
      kb_id: "dataset-1",
      parser_id: "naive",
      created_by: "user-1",
    };
    platformTestServer.use(
      http.post("http://platform.test/api/v1/documents", async ({ request }) => {
        calls.push({ method: request.method, path: new URL(request.url).pathname, body: await request.json() });
        return HttpResponse.json({ code: 0, data: generic });
      }),
      http.get("http://platform.test/api/v1/documents/generic-1", () =>
        HttpResponse.json({ data: generic }),
      ),
      http.put("http://platform.test/api/v1/documents/generic-1", async ({ request }) => {
        calls.push({ method: request.method, path: new URL(request.url).pathname, body: await request.json() });
        return HttpResponse.json({ code: 0, data: true });
      }),
      http.delete("http://platform.test/api/v1/documents/generic-1", ({ request }) => {
        calls.push({ method: request.method, path: new URL(request.url).pathname, body: null });
        return HttpResponse.json({ code: 0, data: true });
      }),
      http.post("http://platform.test/api/v1/documents/ingest", async ({ request }) => {
        calls.push({ method: request.method, path: new URL(request.url).pathname, body: await request.json() });
        return HttpResponse.json({ code: 0, data: true });
      }),
    );

    await createGenericDocument({
      name: "note.txt",
      kb_id: "dataset-1",
      parser_id: "naive",
      created_by: "user-1",
      type: "text",
      source: "local",
    });
    await expect(getGenericDocument("generic-1")).resolves.toMatchObject({ id: "generic-1" });
    await updateGenericDocument("generic-1", { name: "renamed.txt" });
    await ingestGenericDocuments(["generic-1"], "1");
    await deleteGenericDocument("generic-1");

    expect(calls.map((call) => call.body)).toEqual([
      {
        name: "note.txt",
        kb_id: "dataset-1",
        parser_id: "naive",
        created_by: "user-1",
        type: "text",
        source: "local",
      },
      { name: "renamed.txt" },
      { doc_ids: ["generic-1"], run: "1", delete: false, apply_kb: true },
      null,
    ]);
  });

  it("covers upload inspection, thumbnail/image/artifact blobs and both task-cancel forms", async () => {
    const taskBodies: unknown[] = [];
    platformTestServer.use(
      http.post("http://platform.test/api/v1/documents/upload", async ({ request }) => {
        expect((await request.formData()).getAll("file")).toHaveLength(2);
        return HttpResponse.json({ code: 0, data: [{ name: "a.pdf" }, { name: "b.txt" }] });
      }),
      http.get("http://platform.test/api/v1/thumbnails", ({ request }) => {
        expect(new URL(request.url).searchParams.getAll("doc_ids")).toEqual(["doc-1"]);
        return HttpResponse.json({ code: 0, data: { "doc-1": "/api/v1/documents/images/dataset-1-thumb.png" } });
      }),
      http.get("http://platform.test/api/v1/documents/images/dataset-1-thumb.png", () =>
        new HttpResponse(new Uint8Array([137, 80, 78, 71]), { headers: { "Content-Type": "image/png" } }),
      ),
      http.get("http://platform.test/api/v1/documents/artifact/report.pdf", () =>
        new HttpResponse("pdf", { headers: { "Content-Type": "application/pdf", "Content-Disposition": 'inline; filename="report.pdf"' } }),
      ),
      http.post("http://platform.test/api/v1/tasks/task-1/cancel", () =>
        HttpResponse.json({ code: 0, data: true }),
      ),
      http.patch("http://platform.test/api/v1/tasks/task-1", async ({ request }) => {
        taskBodies.push(await request.json());
        return HttpResponse.json({ code: 0, data: true });
      }),
    );

    await expect(inspectDocumentUploads([
      new File(["a"], "a.pdf"),
      new File(["b"], "b.txt"),
    ])).resolves.toHaveLength(2);
    await expect(listDocumentThumbnails(["doc-1"])).resolves.toEqual({
      "doc-1": "/api/v1/documents/images/dataset-1-thumb.png",
    });
    expect((await fetchDocumentImage("dataset-1-thumb.png")).contentType).toBe("image/png");
    expect((await fetchDocumentArtifact("report.pdf")).disposition).toContain("inline");
    await expect(cancelPlatformTask("task-1")).resolves.toBe(true);
    await expect(stopPlatformTask("task-1")).resolves.toBe(true);
    expect(taskBodies).toEqual([{ action: "stop" }]);
  });

  it("surfaces unauthorized media and aborts without leaking a token into the URL", async () => {
    let requestedUrl = "";
    platformTestServer.use(
      http.get("http://platform.test/api/v1/documents/secret/preview", ({ request }) => {
        requestedUrl = request.url;
        return HttpResponse.json({ code: 102, message: "document not found" });
      }),
    );
    await expect(fetchDocumentPreview("secret")).rejects.toMatchObject({ code: 102 });
    expect(requestedUrl).not.toContain("token=");

    const controller = new AbortController();
    controller.abort();
    await expect(fetchDocumentPreview("secret", controller.signal)).rejects.toMatchObject({
      code: "CLIENT_ABORTED",
    });
  });
});
