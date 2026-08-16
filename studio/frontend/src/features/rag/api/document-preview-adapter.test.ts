import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const mocks = vi.hoisted(() => ({
  fetchPlatform: vi.fn(),
  getLocalTarget: vi.fn(),
  getLocalFileUrl: vi.fn(),
}));

vi.mock("@/integrations/platform-backend", () => ({
  fetchDocumentPreview: mocks.fetchPlatform,
}));

vi.mock("./rag-api", () => ({
  getPreviewTarget: mocks.getLocalTarget,
  getDocumentFileUrl: mocks.getLocalFileUrl,
}));

import { resolveDocumentPreview } from "./document-preview-adapter";

describe("document preview adapter", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.stubGlobal("URL", {
      createObjectURL: vi.fn(() => "blob:platform-preview"),
      revokeObjectURL: vi.fn(),
    });
  });

  afterEach(() => vi.unstubAllGlobals());

  it("uses the authenticated platform preview route for platform citations", async () => {
    mocks.fetchPlatform.mockResolvedValue({
      blob: new Blob(["pdf"], { type: "application/pdf" }),
      contentType: "application/pdf",
      disposition: 'inline; filename="Guide.pdf"',
    });

    const resolved = await resolveDocumentPreview({
      documentId: "doc-1",
      filename: "Guide.pdf",
      page: 3,
      source: "platform",
    });

    expect(mocks.fetchPlatform).toHaveBeenCalledWith("doc-1", undefined);
    expect(mocks.getLocalTarget).not.toHaveBeenCalled();
    expect(resolved).toMatchObject({
      fileUrl: "blob:platform-preview",
      target: {
        documentId: "doc-1",
        filename: "Guide.pdf",
        mediaKind: "pdf",
        targetPage: 3,
      },
    });
    resolved.dispose();
    expect(URL.revokeObjectURL).toHaveBeenCalledWith("blob:platform-preview");
  });

  it("keeps local citations on the local preview contract", async () => {
    mocks.getLocalTarget.mockResolvedValue({
      documentId: "local-1",
      filename: "Local.pdf",
      mediaKind: "pdf",
      pdfRegions: [],
    });
    mocks.getLocalFileUrl.mockResolvedValue("http://local/file.pdf");

    const resolved = await resolveDocumentPreview({
      documentId: "local-1",
      source: "local",
    });

    expect(mocks.fetchPlatform).not.toHaveBeenCalled();
    expect(mocks.getLocalTarget).toHaveBeenCalledWith(
      "local-1",
      undefined,
      undefined,
    );
    expect(resolved.fileUrl).toBe("http://local/file.pdf");
  });
});
