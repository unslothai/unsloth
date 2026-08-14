import { render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

const mocks = vi.hoisted(() => ({
  preview: vi.fn(),
  thumbnails: vi.fn(),
  image: vi.fn(),
  artifact: vi.fn(),
}));

vi.mock("@/integrations/platform-backend", async (importOriginal) => {
  const original = await importOriginal<
    typeof import("@/integrations/platform-backend")
  >();
  return {
    ...original,
    fetchDocumentPreview: mocks.preview,
    listDocumentThumbnails: mocks.thumbnails,
    fetchDocumentImage: mocks.image,
    fetchDocumentArtifact: mocks.artifact,
  };
});

import { DocumentAssetDialog, DocumentInlinePreview } from "./document-asset-dialog";

const documentRow = {
  id: "doc-1",
  datasetId: "dataset-1",
  name: "guide.pdf",
  thumbnail: null,
  sizeBytes: 20,
  sourceType: "local",
  location: null,
  tokenCount: 4,
  chunkCount: 1,
  progress: 1,
  progressMessage: null,
  processDuration: 1,
  suffix: "pdf",
  run: "3",
  backendStatus: "3",
  status: "completed" as const,
  parserId: "naive",
  chunkMethod: "naive",
  pipelineId: null,
  pipelineName: null,
  createdAt: null,
  updatedAt: null,
};

describe("Phase 5 authenticated asset dialog", () => {
  beforeEach(() => {
    vi.stubGlobal("ResizeObserver", class { observe() {} unobserve() {} disconnect() {} });
    vi.stubGlobal("URL", {
      ...URL,
      createObjectURL: vi.fn(() => "blob:phase5-preview"),
      revokeObjectURL: vi.fn(),
    });
    mocks.preview.mockReset();
  });

  it("creates a PDF object URL only after authenticated fetch and revokes it on unmount", async () => {
    mocks.preview.mockResolvedValue({
      blob: new Blob(["pdf"], { type: "application/pdf" }),
      contentType: "application/pdf",
      disposition: 'inline; filename="guide.pdf"',
    });
    const view = render(
      <DocumentAssetDialog
        document={documentRow}
        mode="preview"
        open
        onOpenChange={() => undefined}
      />,
    );

    await waitFor(() =>
      expect(screen.getByTitle("guide.pdf önizlemesi")).toHaveAttribute(
        "src",
        "blob:phase5-preview",
      ),
    );
    expect(mocks.preview).toHaveBeenCalledWith("doc-1", expect.any(AbortSignal));
    expect(URL.createObjectURL).toHaveBeenCalledTimes(1);
    view.unmount();
    expect(URL.revokeObjectURL).toHaveBeenCalledWith("blob:phase5-preview");
  });

  it("shows permission/media failures instead of treating them as an empty preview", async () => {
    mocks.preview.mockRejectedValue(new Error("document not found"));
    render(
      <DocumentAssetDialog
        document={documentRow}
        mode="preview"
        open
        onOpenChange={() => undefined}
      />,
    );
    expect(await screen.findByText("İçerik açılamadı")).toBeInTheDocument();
    expect(screen.getByText("document not found")).toBeInTheDocument();
  });

  it("loads the selected document preview inline for the Hub split-detail view", async () => {
    mocks.preview.mockResolvedValue({
      blob: new Blob(["inline content"], { type: "text/plain" }),
      contentType: "text/plain",
      disposition: 'inline; filename="guide.txt"',
    });

    render(<DocumentInlinePreview document={{ ...documentRow, name: "guide.txt", suffix: "txt" }} />);

    expect(await screen.findByText("inline content")).toBeInTheDocument();
    expect(mocks.preview).toHaveBeenCalledWith("doc-1", expect.any(AbortSignal));
  });
});
