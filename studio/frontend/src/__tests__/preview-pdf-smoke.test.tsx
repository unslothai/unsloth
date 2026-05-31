import type { PreviewTarget } from "@/features/rag/api/rag-api";
import type { PreviewPdfRegion } from "@/features/rag/api/rag-api";
import { PreviewPdfView } from "@/features/rag/components/preview-pdf-view";
import {
  act,
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@testing-library/react";
import React from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";

const SOURCE_EXCERPT_TEXT = /source excerpt/i;

/** The thumbnail rail also renders mocked `<Page>` elements, so tests
 *  targeting the main render scope through the `pdf-main-page` wrapper
 *  instead of taking the first `pdf-page`. */
async function findMainPdfPage(): Promise<HTMLElement> {
  const wrapper = await screen.findByTestId("pdf-main-page");
  return within(wrapper).getByTestId("pdf-page");
}
function getMainPdfPage(): HTMLElement {
  const wrapper = screen.getByTestId("pdf-main-page");
  return within(wrapper).getByTestId("pdf-page");
}

vi.mock("react-pdf", () => ({
  Document: ({
    children,
    file,
    onLoadSuccess,
  }: {
    children: React.ReactNode;
    file?: unknown;
    onLoadSuccess?: (result: { numPages: number }) => void;
  }) => {
    onLoadSuccess?.({ numPages: 1 });
    return React.createElement(
      "div",
      {
        "data-testid": "pdf-document",
        "data-file-kind": file instanceof Blob ? "blob" : typeof file,
        "data-file-url":
          file && typeof file === "object" && "url" in file
            ? String((file as { url: string }).url)
            : "",
      },
      children,
    );
  },
  Page: ({
    customTextRenderer,
    width,
    renderTextLayer,
  }: {
    customTextRenderer?: (item: { str: string }) => string;
    width?: number;
    renderTextLayer?: boolean;
  }) => {
    const html =
      customTextRenderer?.({ str: "target phrase" }) ?? "target phrase";
    return React.createElement("div", {
      "data-testid": "pdf-page",
      "data-width": String(width ?? ""),
      "data-render-text-layer": String(renderTextLayer),
      "data-rendered-html": html,
    });
  },
  pdfjs: { GlobalWorkerOptions: { workerSrc: "" } },
}));

function target(overrides: Partial<PreviewTarget> = {}): PreviewTarget {
  return {
    documentId: "doc-abc",
    filename: "report.pdf",
    contentType: "application/pdf",
    mediaKind: "pdf",
    byteSize: 100,
    status: "completed",
    kbId: "kb-1",
    threadId: null,
    chunkId: "chunk-1",
    chunkIndex: 0,
    targetPage: 1,
    snippet: "target phrase appears here",
    kind: "text",
    imageUrl: null,
    sourcePageIndex: 0,
    pageCharStart: 0,
    pageCharEnd: 13,
    lineStart: 1,
    lineEnd: 1,
    pdfRegions: [],
    ...overrides,
  };
}

beforeEach(() => {
  class ResizeObserverMock implements ResizeObserver {
    observe(_target: Element, _options?: ResizeObserverOptions) {
      // jsdom has no layout observer; only the API shape is needed.
    }
    unobserve(_target: Element) {
      // jsdom has no layout observer; only the API shape is needed.
    }
    disconnect() {
      // jsdom has no layout observer; only the API shape is needed.
    }
  }
  vi.stubGlobal("ResizeObserver", ResizeObserverMock);
});

describe("PreviewPdfView smoke", () => {
  it("renders a range URL source with text search and exact region overlay", async () => {
    render(
      React.createElement(PreviewPdfView, {
        target: target({
          pdfRegions: [
            {
              pageIndex: 0,
              pageNumber: 1,
              x: 0.1,
              y: 0.2,
              width: 0.3,
              height: 0.04,
              confidence: "exact",
              source: "pymupdf-search",
            },
          ],
        }),
        file: "http://127.0.0.1:8888/api/rag/documents/doc-abc/file-signed?token=signed",
      }),
    );

    expect(screen.getByTestId("pdf-document")).toHaveAttribute(
      "data-file-kind",
      "object",
    );
    expect(screen.getByTestId("pdf-document")).toHaveAttribute(
      "data-file-url",
      expect.stringContaining("/file-signed?token=signed"),
    );
    const page = await findMainPdfPage();
    await waitFor(() => {
      expect(page.getAttribute("data-render-text-layer")).toBe("true");
    });
    expect(page.getAttribute("data-rendered-html")).toBe("target phrase");
    expect(page.getAttribute("data-rendered-html")).not.toContain("<mark>");

    // Verify brand green highlight overlays
    const regionHighlight = screen.getByTestId("pdf-region-highlight");
    expect(regionHighlight).toBeInTheDocument();
    expect(regionHighlight).toHaveClass("bg-primary/20");
    expect(regionHighlight).toHaveClass("ring-primary/60");

    // Verify Tailwind v4 light-mode isolation reset wrapper. Post
    // thumbnail-rail refactor it lives INSIDE the Document, directly
    // wrapping the main-page block.
    const wrapper = screen.getByTestId("pdf-main-page").parentElement;
    expect(wrapper).toHaveClass("light");
    expect(wrapper).toHaveClass("bg-white");
    expect(wrapper).toHaveClass("text-slate-900");

    // Verify Shadcn toolbar elements and rounded-full pill groups
    const zoomInBtn = screen.getByRole("button", { name: "Zoom in" });
    expect(zoomInBtn).toHaveClass("rounded-full");
    expect(zoomInBtn.parentElement).toHaveClass(
      "bg-muted/40",
      "p-0.5",
      "shadow-xs",
    );

    // Source-excerpt card uses a neutral muted surface (no brand-coloured
    // left rail) so it doesn't visually compete in the panel.
    const excerptCard = screen.getByText(SOURCE_EXCERPT_TEXT).parentElement;
    expect(excerptCard).toHaveClass("border-border/60");
    expect(excerptCard).toHaveClass("bg-muted/30");
    expect(excerptCard).not.toHaveClass("border-l-primary");

    fireEvent.change(screen.getByLabelText("Search this PDF"), {
      target: { value: "phrase" },
    });
    await waitFor(() => {
      expect(
        getMainPdfPage().getAttribute("data-rendered-html"),
      ).toContain("<mark>phrase</mark>");
    });

    const beforeZoom = Number(page.getAttribute("data-width"));
    fireEvent.click(screen.getByRole("button", { name: "Zoom in" }));
    await waitFor(() => {
      expect(
        Number(getMainPdfPage().getAttribute("data-width")),
      ).toBeGreaterThan(beforeZoom);
    });
  });

  it("debounces ResizeObserver transitions to prevent infinite rendering loops", async () => {
    const resizeCallbacks: ResizeObserverCallback[] = [];
    class FakeResizeObserver implements ResizeObserver {
      constructor(callback: ResizeObserverCallback) {
        resizeCallbacks.push(callback);
      }
      observe(_target: Element, _options?: ResizeObserverOptions) {
        // jsdom has no layout observer; only the API shape is needed.
      }
      unobserve(_target: Element) {
        // jsdom has no layout observer; only the API shape is needed.
      }
      disconnect() {
        // jsdom has no layout observer; only the API shape is needed.
      }
    }
    vi.stubGlobal("ResizeObserver", FakeResizeObserver);

    render(
      React.createElement(PreviewPdfView, {
        target: target(),
        file: "http://127.0.0.1:8888/api/rag/documents/doc-abc/file-signed?token=signed",
      }),
    );

    // Initial render sets width synchronously on mount; capture it.
    const page = await findMainPdfPage();
    const initialWidth = Number(page.getAttribute("data-width"));

    // Fake timers AFTER finding elements, to avoid findByTestId timeout
    vi.useFakeTimers();

    // Set up HTMLDivElement.prototype.clientWidth mock
    const originalClientWidth = Object.getOwnPropertyDescriptor(
      HTMLDivElement.prototype,
      "clientWidth",
    );
    let clientWidthValue = 300;
    Object.defineProperty(HTMLDivElement.prototype, "clientWidth", {
      get() {
        return clientWidthValue;
      },
      configurable: true,
    });

    // Trigger resize callback after changing clientWidth
    clientWidthValue = 600;
    const resizeCallback = resizeCallbacks[0];
    if (!resizeCallback) {
      throw new Error("Expected ResizeObserver callback to be registered");
    }
    const resizeObserver: ResizeObserver = {
      observe() {
        // The callback under test ignores the observer arg.
      },
      unobserve() {
        // The callback under test ignores the observer arg.
      },
      disconnect() {
        // The callback under test ignores the observer arg.
      },
    };
    resizeCallback([], resizeObserver);

    // Width must NOT update immediately (100ms debounce)
    expect(Number(getMainPdfPage().getAttribute("data-width"))).toBe(
      initialWidth,
    );

    // Advance 100ms to fire the debounced callback and flush updates
    act(() => {
      vi.advanceTimersByTime(100);
      vi.runAllTimers();
    });

    // Now the width should have updated
    expect(
      Number(getMainPdfPage().getAttribute("data-width")),
    ).not.toBe(initialWidth);
    expect(Number(getMainPdfPage().getAttribute("data-width"))).toBe(572); // 600 - 28 (PDF_BODY_GUTTER_PX)

    // Clean up prototype descriptor
    if (originalClientWidth) {
      Object.defineProperty(
        HTMLDivElement.prototype,
        "clientWidth",
        originalClientWidth,
      );
    } else {
      Reflect.deleteProperty(HTMLDivElement.prototype, "clientWidth");
    }

    vi.useRealTimers();
  });

  it("renders only 'exact' confidence highlights and positions them with correct percentages", async () => {
    const nonExactRegion = {
      pageIndex: 0,
      pageNumber: 1,
      x: 0.5,
      y: 0.5,
      width: 0.2,
      height: 0.2,
      confidence: "fuzzy",
      source: "pymupdf-search",
    } as unknown as PreviewPdfRegion;

    render(
      React.createElement(PreviewPdfView, {
        target: target({
          pdfRegions: [
            {
              pageIndex: 0,
              pageNumber: 1,
              x: 0.15,
              y: 0.25,
              width: 0.35,
              height: 0.45,
              confidence: "exact",
              source: "pymupdf-search",
            },
            nonExactRegion,
          ],
        }),
        file: "http://127.0.0.1:8888/api/rag/documents/doc-abc/file-signed?token=signed",
      }),
    );

    await findMainPdfPage();

    const highlights = screen.getAllByTestId("pdf-region-highlight");
    expect(highlights.length).toBe(1);

    const exactHighlight = highlights[0];
    expect(exactHighlight.style.left).toBe("15%");
    expect(exactHighlight.style.top).toBe("25%");
    expect(exactHighlight.style.width).toBe("35%");
    expect(exactHighlight.style.height).toBe("45%");
  });

  it("uses stable scrollbar style classes in the PDF sidebar and page container to prevent shifting", async () => {
    render(
      React.createElement(PreviewPdfView, {
        target: target(),
        file: "http://127.0.0.1:8888/api/rag/documents/doc-abc/file-signed?token=signed",
      }),
    );

    const mainPageWrapper = await screen.findByTestId("pdf-main-page");

    // pdf-main-page → light-wrapper → scrollContainer
    const scrollContainer =
      mainPageWrapper.parentElement?.parentElement ?? null;
    expect(scrollContainer).toHaveClass("preview-scrollbar");
    expect(scrollContainer).toHaveClass("overflow-y-scroll");
    expect(scrollContainer).toHaveClass("overflow-x-auto");

    const sidebar = screen.getByRole("button", {
      name: "Go to page 1",
    }).parentElement;
    expect(sidebar).toHaveClass("preview-scrollbar");
    expect(sidebar).toHaveClass("overflow-y-auto");
  });

  it("highlights search terms using the custom text renderer with the mark wrapper", async () => {
    render(
      React.createElement(PreviewPdfView, {
        target: target({
          snippet: "this snippet contains some special keyword",
        }),
        file: "http://127.0.0.1:8888/api/rag/documents/doc-abc/file-signed?token=signed",
      }),
    );

    await findMainPdfPage();

    fireEvent.change(screen.getByLabelText("Search this PDF"), {
      target: { value: "phrase" },
    });

    await waitFor(() => {
      expect(
        getMainPdfPage().getAttribute("data-rendered-html"),
      ).toContain("<mark>phrase</mark>");
    });
  });
});
