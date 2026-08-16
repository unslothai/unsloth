import { fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it } from "vitest";

import { useDocumentPreviewStore } from "@/features/rag/components/preview-store";
import { parseCitations } from "./citation-utils";
import { CitationBadge } from "./tool-ui-knowledge-base";

describe("Rag Platform Phase 8 citation preview", () => {
  beforeEach(() => useDocumentPreviewStore.getState().closePreview());

  it("preserves native document/chunk identity and opens the shared preview", () => {
    const [citation] = parseCitations([
      {
        id: "chunk-1",
        chunkId: "chunk-1",
        documentId: "doc-1",
        datasetId: "dataset-1",
        source: "platform",
        filename: "Guide.pdf",
        text: "Evidence",
        page: 3,
        score: 0.91,
      },
    ]);
    expect(citation).toMatchObject({
      chunkId: "chunk-1",
      documentId: "doc-1",
      datasetId: "dataset-1",
      source: "platform",
      page: 3,
    });

    render(<CitationBadge citation={citation} index={0} />);
    fireEvent.click(screen.getByRole("button", { name: /Guide\.pdf/ }));
    expect(useDocumentPreviewStore.getState()).toMatchObject({
      open: true,
      documentId: "doc-1",
      chunkId: "chunk-1",
      filename: "Guide.pdf",
      page: 3,
    });
  });
});
