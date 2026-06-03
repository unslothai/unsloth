import type { PreviewTarget } from "@/features/rag/api/rag-api";
import { PreviewPdfNativeView } from "@/features/rag/components/preview-pdf-native";
import { render, screen } from "@testing-library/react";
import React from "react";
import { afterEach, describe, expect, it, vi } from "vitest";

function makeTarget(overrides: Partial<PreviewTarget> = {}): PreviewTarget {
  return {
    documentId: "doc-1",
    filename: "report.pdf",
    contentType: "application/pdf",
    mediaKind: "pdf",
    byteSize: 100,
    status: "completed",
    kbId: "kb-1",
    threadId: null,
    chunkId: "c1",
    chunkIndex: 0,
    targetPage: 3,
    // pageCharStart/End span the second line of the snippet below.
    snippet: "Intro line\nThe cited sentence here\nOutro",
    kind: "text",
    sourcePageIndex: 2,
    pageCharStart: 11,
    pageCharEnd: 34,
    lineStart: null,
    lineEnd: null,
    pdfRegions: [],
    ...overrides,
  };
}

describe("PreviewPdfNativeView", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("renders a native PDF iframe jumped to the cited page from a signed URL", () => {
    render(
      React.createElement(PreviewPdfNativeView, {
        target: makeTarget(),
        file: "/api/rag/documents/doc-1/file-signed?token=abc",
      }),
    );
    const iframe = document.querySelector("iframe");
    expect(iframe).not.toBeNull();
    expect(iframe?.getAttribute("src")).toBe(
      "/api/rag/documents/doc-1/file-signed?token=abc#page=3&view=FitH",
    );
  });

  it("highlights the cited excerpt beside the viewer", () => {
    render(
      React.createElement(PreviewPdfNativeView, {
        target: makeTarget(),
        file: "/signed?token=x",
      }),
    );
    const mark = screen.getByText("The cited sentence here");
    expect(mark.tagName).toBe("MARK");
  });

  it("uses an object URL for a Blob and revokes it on unmount", () => {
    const createSpy = vi
      .spyOn(URL, "createObjectURL")
      .mockReturnValue("blob:obj");
    const revokeSpy = vi
      .spyOn(URL, "revokeObjectURL")
      .mockImplementation(() => undefined);

    const { unmount } = render(
      React.createElement(PreviewPdfNativeView, {
        target: makeTarget({ targetPage: 2 }),
        file: new Blob(["%PDF-1.4"], { type: "application/pdf" }),
      }),
    );

    const iframe = document.querySelector("iframe");
    expect(iframe?.getAttribute("src")).toBe("blob:obj#page=2&view=FitH");
    expect(createSpy).toHaveBeenCalled();

    unmount();
    expect(revokeSpy).toHaveBeenCalledWith("blob:obj");
  });
});
