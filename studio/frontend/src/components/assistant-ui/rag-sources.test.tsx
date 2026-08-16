import { fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { useDocumentPreviewStore } from "@/features/rag/components/preview-store";

const GUIDE_SOURCE_LABEL = /Guide\.pdf · p\.3/;

const mocks = vi.hoisted(() => ({
  message: {
    metadata: {},
    content: [],
  } as {
    metadata: unknown;
    content: Record<string, unknown>[];
  },
}));

vi.mock("@assistant-ui/react", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@assistant-ui/react")>()),
  useMessage: () => mocks.message,
}));

import { RagSourcesGroup } from "./rag-sources";

describe("Rag Platform reference sources", () => {
  beforeEach(() => {
    mocks.message = { metadata: {}, content: [] };
  });

  it("renders a persisted raw platform reference when citation metadata is absent", () => {
    mocks.message = {
      metadata: {
        custom: {
          platformReference: {
            chunks: [
              {
                id: "chunk-1",
                document_id: "doc-1",
                document_name: "Guide.pdf",
                dataset_id: "dataset-1",
                content: "Evidence",
                positions: [[3, 0]],
                similarity: 0.91,
              },
            ],
          },
        },
      },
      content: [],
    };

    render(<RagSourcesGroup />);

    expect(screen.getByText("Document Sources")).toBeInTheDocument();
    const sourceButton = screen.getByRole("button", {
      name: GUIDE_SOURCE_LABEL,
    });
    expect(sourceButton).toBeInTheDocument();
    fireEvent.click(sourceButton);
    expect(useDocumentPreviewStore.getState()).toMatchObject({
      documentId: "doc-1",
      source: "platform",
    });
  });

  it("routes legacy platform citations without a source marker to the platform preview", () => {
    mocks.message = {
      metadata: {
        custom: {
          platformCitations: [
            {
              id: "chunk-1",
              documentId: "doc-1",
              chunkId: "chunk-1",
              filename: "Guide.pdf",
              page: 3,
            },
          ],
        },
      },
      content: [],
    };

    render(<RagSourcesGroup />);

    fireEvent.click(
      screen.getByRole("button", { name: GUIDE_SOURCE_LABEL }),
    );
    expect(useDocumentPreviewStore.getState()).toMatchObject({
      documentId: "doc-1",
      source: "platform",
    });
  });
});
