/**
 * Tests for preview-store object URL lifecycle (contracts §5, T4).
 *
 * Coverage:
 * - open() revokes the previous object URL before assigning a new one.
 * - close() revokes any live object URL.
 * - Opening doc B while doc A is loaded revokes doc A's URL.
 * - PDFs use a signed range URL, not a full blob download.
 * - Inline object URLs created ONLY for safe non-PDF kinds (text/image).
 * - Unsafe kinds (html/docx/unknown) skip blob fetch; previewBlobUrl = null.
 * - isInlineBlobAllowed matches the contracts §5.4 allowlist.
 * - __previewStoreInternals() verifies module-scoped cleanup.
 */

import type {
  PreviewMediaKind,
  PreviewTarget,
} from "@/features/rag/api/rag-api";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

// ── Mock rag-api BEFORE importing the store ───────────────────────────
// vi.hoisted initialises the mock refs before the vi.mock factory runs
// (Vitest hoists vi.mock to the top of the file).

const {
  mockFetchPreviewTarget,
  mockFetchPreviewFileBlob,
  mockFetchPreviewFileUrl,
} = vi.hoisted(() => ({
  mockFetchPreviewTarget:
    vi.fn<
      (documentId: string, chunkId?: string | null) => Promise<PreviewTarget>
    >(),
  mockFetchPreviewFileBlob:
    vi.fn<(documentId: string, signal?: AbortSignal) => Promise<Blob>>(),
  mockFetchPreviewFileUrl: vi.fn<
    (
      documentId: string,
      signal?: AbortSignal,
    ) => Promise<{ url: string; expiresAt: number }>
  >(),
}));

vi.mock("@/features/rag/api/rag-api", async (importOriginal) => {
  const original =
    await importOriginal<typeof import("@/features/rag/api/rag-api")>();
  return {
    ...original,
    fetchPreviewTarget: mockFetchPreviewTarget,
    fetchPreviewFileBlob: mockFetchPreviewFileBlob,
    fetchPreviewFileUrl: mockFetchPreviewFileUrl,
  };
});

// ── Import store AFTER mock registration ─────────────────────────────

import {
  __previewStoreInternals,
  isInlineBlobAllowed,
  usePreviewStore,
} from "@/features/rag/stores/preview-store";

// ── Mock URL.createObjectURL / revokeObjectURL ────────────────────────

let urlCounter = 0;

beforeEach(() => {
  urlCounter = 0;
  vi.spyOn(URL, "createObjectURL").mockImplementation(() => {
    return `blob:test/${++urlCounter}`;
  });
  vi.spyOn(URL, "revokeObjectURL").mockImplementation((_url: string) => {
    /* no-op */
  });
  mockFetchPreviewTarget.mockReset();
  mockFetchPreviewFileBlob.mockReset();
  mockFetchPreviewFileUrl.mockReset();
  // Reset store to idle per test
  usePreviewStore.getState().close();
});

afterEach(() => {
  vi.restoreAllMocks();
});

// ── Helpers ───────────────────────────────────────────────────────────

function makeTarget(overrides: Partial<PreviewTarget> = {}): PreviewTarget {
  return {
    documentId: "doc-abc",
    filename: "report.pdf",
    contentType: "application/pdf",
    mediaKind: "pdf",
    byteSize: 100,
    status: "completed",
    kbId: "kb-1",
    threadId: null,
    chunkId: null,
    chunkIndex: null,
    targetPage: null,
    snippet: null,
    kind: null,
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

function makePdfBlob(): Blob {
  return new Blob(["%PDF-1.4"], { type: "application/pdf" });
}

// ── Pure-logic: isInlineBlobAllowed (always green) ────────────────────

describe("isInlineBlobAllowed (contracts §5.4, pure logic)", () => {
  const safe: PreviewMediaKind[] = ["pdf", "text", "image"];
  const unsafe: PreviewMediaKind[] = ["html", "docx", "unknown"];

  it.each(safe)("mediaKind=%s is inline-safe", (mk) => {
    expect(isInlineBlobAllowed(mk)).toBe(true);
  });

  it.each(unsafe)("mediaKind=%s is NOT inline-safe (Risk #3)", (mk) => {
    expect(isInlineBlobAllowed(mk)).toBe(false);
  });
});

// ── Integration tests against real store ─────────────────────────────

describe("preview-store open/close lifecycle (contracts §5)", () => {
  it("open() for pdf stores a signed URL without creating an object URL", async () => {
    mockFetchPreviewTarget.mockResolvedValue(makeTarget({ mediaKind: "pdf" }));
    mockFetchPreviewFileUrl.mockResolvedValue({
      url: "http://127.0.0.1:8888/api/rag/documents/doc-abc/file-signed?token=signed",
      expiresAt: 1_700_000_000,
    });

    await usePreviewStore.getState().open({ documentId: "doc-abc" });

    expect(mockFetchPreviewFileBlob).not.toHaveBeenCalled();
    expect(URL.createObjectURL).not.toHaveBeenCalled();
    const { previewBlob, previewBlobUrl, previewFileUrl, status } =
      usePreviewStore.getState();
    expect(previewBlob).toBeNull();
    expect(previewBlobUrl).toBeNull();
    expect(previewFileUrl).toContain("/file-signed?token=signed");
    expect(status).toBe("ready");
  });

  it("open() passes backendChunkId through to fetchPreviewTarget", async () => {
    mockFetchPreviewTarget.mockResolvedValue(makeTarget({ mediaKind: "pdf" }));
    mockFetchPreviewFileUrl.mockResolvedValue({
      url: "/api/rag/documents/doc-abc/file-signed?token=signed",
      expiresAt: 1_700_000_000,
    });

    await usePreviewStore.getState().open({
      documentId: "doc-abc",
      backendChunkId: "chunk-xyz",
    });

    expect(mockFetchPreviewTarget).toHaveBeenCalledWith(
      "doc-abc",
      "chunk-xyz",
    );
  });

  it("close() revokes the live object URL and clears state", async () => {
    mockFetchPreviewTarget.mockResolvedValue(
      makeTarget({ mediaKind: "text", filename: "notes.txt" }),
    );
    mockFetchPreviewFileBlob.mockResolvedValue(makePdfBlob());

    await usePreviewStore.getState().open({ documentId: "doc-abc" });
    const blobUrl = usePreviewStore.getState().previewBlobUrl;
    expect(blobUrl).toMatch(/^blob:/);

    usePreviewStore.getState().close();

    expect(URL.revokeObjectURL).toHaveBeenCalledWith(blobUrl);
    const { previewBlob, previewBlobUrl, target, status } =
      usePreviewStore.getState();
    expect(previewBlob).toBeNull();
    expect(previewBlobUrl).toBeNull();
    expect(target).toBeNull();
    expect(status).toBe("idle");
  });

  it("opening doc B revokes doc A's URL before creating doc B's (contracts §5.1)", async () => {
    mockFetchPreviewTarget.mockResolvedValue(
      makeTarget({ mediaKind: "text", filename: "notes.txt" }),
    );
    mockFetchPreviewFileBlob.mockResolvedValue(makePdfBlob());

    await usePreviewStore.getState().open({ documentId: "doc-A" });
    const urlA = usePreviewStore.getState().previewBlobUrl;
    expect(urlA).toMatch(/^blob:/);

    await usePreviewStore.getState().open({ documentId: "doc-B" });

    expect(URL.revokeObjectURL).toHaveBeenCalledWith(urlA);
    const urlB = usePreviewStore.getState().previewBlobUrl;
    expect(urlB).not.toBe(urlA);
    expect(urlB).toMatch(/^blob:/);
  });

  it("html mediaKind skips blob fetch and sets previewBlobUrl = null (Risk #3)", async () => {
    mockFetchPreviewTarget.mockResolvedValue(
      makeTarget({ mediaKind: "html", filename: "evil.html" }),
    );

    await usePreviewStore.getState().open({ documentId: "doc-html" });

    expect(mockFetchPreviewFileBlob).not.toHaveBeenCalled();
    expect(mockFetchPreviewFileUrl).not.toHaveBeenCalled();
    expect(URL.createObjectURL).not.toHaveBeenCalled();
    const { previewBlob, previewBlobUrl, status } = usePreviewStore.getState();
    expect(previewBlob).toBeNull();
    expect(previewBlobUrl).toBeNull();
    expect(status).toBe("ready");
  });

  it("docx mediaKind skips blob fetch and sets previewBlobUrl = null (Risk #3)", async () => {
    mockFetchPreviewTarget.mockResolvedValue(
      makeTarget({ mediaKind: "docx", filename: "report.docx" }),
    );

    await usePreviewStore.getState().open({ documentId: "doc-docx" });

    expect(mockFetchPreviewFileBlob).not.toHaveBeenCalled();
    expect(mockFetchPreviewFileUrl).not.toHaveBeenCalled();
    expect(URL.createObjectURL).not.toHaveBeenCalled();
    expect(usePreviewStore.getState().previewBlob).toBeNull();
    expect(usePreviewStore.getState().previewBlobUrl).toBeNull();
  });

  it("unknown mediaKind skips blob fetch and sets previewBlobUrl = null", async () => {
    mockFetchPreviewTarget.mockResolvedValue(
      makeTarget({ mediaKind: "unknown", filename: "data.bin" }),
    );

    await usePreviewStore.getState().open({ documentId: "doc-bin" });

    expect(mockFetchPreviewFileBlob).not.toHaveBeenCalled();
    expect(mockFetchPreviewFileUrl).not.toHaveBeenCalled();
    expect(URL.createObjectURL).not.toHaveBeenCalled();
    expect(usePreviewStore.getState().previewBlob).toBeNull();
    expect(usePreviewStore.getState().previewBlobUrl).toBeNull();
  });

  it("close() when nothing is open does not throw", () => {
    expect(() => usePreviewStore.getState().close()).not.toThrow();
  });

  it("__previewStoreInternals shows no activeBlobUrl after close()", async () => {
    mockFetchPreviewTarget.mockResolvedValue(
      makeTarget({ mediaKind: "text", filename: "notes.txt" }),
    );
    mockFetchPreviewFileBlob.mockResolvedValue(makePdfBlob());

    await usePreviewStore.getState().open({ documentId: "doc-abc" });
    expect(__previewStoreInternals().activeBlobUrl).toMatch(/^blob:/);

    usePreviewStore.getState().close();
    expect(__previewStoreInternals().activeBlobUrl).toBeNull();
    expect(__previewStoreInternals().hasInflightController).toBe(false);
  });

  it("fetchPreviewTarget error sets status=error and clears target", async () => {
    mockFetchPreviewTarget.mockRejectedValue(new Error("404 not found"));

    await usePreviewStore.getState().open({ documentId: "missing" });

    const { status, error, target } = usePreviewStore.getState();
    expect(status).toBe("error");
    expect(error).toMatch(/404/);
    expect(target).toBeNull();
    expect(URL.createObjectURL).not.toHaveBeenCalled();
  });
});
