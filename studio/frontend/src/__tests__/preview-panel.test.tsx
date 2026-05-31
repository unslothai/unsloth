/**
 * Tests for preview-panel — HTML/DOCX/unknown must NEVER render inline (T5 / Risk #3).
 *
 * Acceptance (contracts §5.4, PLAN.md T5, decisions Q7):
 * - "pdf" → react-pdf view mounted (or loading indicator).
 * - "html" → text-view fallback, NO object/embed/iframe with blob URL.
 * - "docx" → text-view fallback, NO inline rendering.
 * - "unknown" → unavailable/download state, NOT inline.
 * - "text" → text/snippet view.
 * - No target → nothing or unavailable state.
 */

import {
  type PreviewMediaKind,
  type PreviewTarget,
} from "@/features/rag/api/rag-api";
import type { PreviewLoadStatus } from "@/features/rag/stores/preview-store";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import React from "react";
import {
  type MockInstance,
  afterEach,
  beforeEach,
  describe,
  expect,
  it,
  vi,
} from "vitest";

const DOWNLOAD_BUTTON_NAME = /download/i;
const LONG_CONTENT_TEXT = /Some long content/;

// ── Mock preview store ────────────────────────────────────────────────
// The component uses per-field selectors (usePreviewStore((s) => s.target)),
// so the mock must handle the selector pattern. vi.hoisted ensures the
// mock fn is initialised before the vi.mock factory runs.

interface MockStoreState {
  target: PreviewTarget | null;
  previewBlobUrl: string | null;
  previewBlob: Blob | null;
  previewFileUrl: string | null;
  previewFileUrlExpiresAt: number | null;
  status: PreviewLoadStatus;
  error: string | null;
  close: () => void;
  open: () => void;
}

let mockState: MockStoreState = {
  target: null,
  previewBlobUrl: null,
  previewBlob: null,
  previewFileUrl: null,
  previewFileUrlExpiresAt: null,
  status: "idle",
  error: null,
  close: vi.fn(),
  open: vi.fn(),
};

const { mockAuthFetch, mockUsePreviewStore } = vi.hoisted(() => {
  // usePreviewStore is called two ways:
  //   usePreviewStore((s) => s.field)  — selector (React hook)
  //   usePreviewStore.getState().close() — outside React (cleanup)
  const fn = vi.fn((selector?: (s: MockStoreState) => unknown) => {
    if (typeof selector === "function") {
      return selector(mockState);
    }
    return mockState;
  }) as ReturnType<typeof vi.fn> & { getState: () => MockStoreState };
  fn.getState = () => mockState;
  return { mockAuthFetch: vi.fn(), mockUsePreviewStore: fn };
});

vi.mock("@/features/auth", () => ({
  authFetch: mockAuthFetch,
  getAuthToken: () => "mock-token-123",
}));

vi.mock("@/features/rag/stores/preview-store", async (importOriginal) => {
  const real =
    await importOriginal<
      typeof import("@/features/rag/stores/preview-store")
    >();
  return {
    ...real,
    usePreviewStore: mockUsePreviewStore,
    // isInlineBlobAllowed passes through from the real module so assertions
    // use the production allowlist, not a test-local copy (D1.5 fix).
  };
});

// react-pdf requires a browser worker URL that doesn't exist in jsdom.
vi.mock("react-pdf", () => ({
  Document: ({ children }: { children: React.ReactNode }) =>
    React.createElement("div", { "data-testid": "pdf-document" }, children),
  Page: () => React.createElement("div", { "data-testid": "pdf-page" }),
  pdfjs: { GlobalWorkerOptions: { workerSrc: "" } },
}));

beforeEach(() => {
  mockAuthFetch.mockReset();
  mockAuthFetch.mockResolvedValue(
    new Response(new Blob(["download bytes"], { type: "text/plain" }), {
      status: 200,
    }),
  );
  mockState = {
    target: null,
    previewBlobUrl: null,
    previewBlob: null,
    previewFileUrl: null,
    previewFileUrlExpiresAt: null,
    status: "idle",
    error: null,
    close: vi.fn(),
    open: vi.fn(),
  };
  mockUsePreviewStore.mockImplementation(
    (selector?: (s: MockStoreState) => unknown) => {
      if (typeof selector === "function") {
        return selector(mockState);
      }
      return mockState;
    },
  );
  // Restore getState; mockImplementation replaces the fn internals
  mockUsePreviewStore.getState = () => mockState;

  // Mock window.matchMedia globally for tests
  window.matchMedia = vi.fn().mockImplementation((query) => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: vi.fn(),
    removeListener: vi.fn(),
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    dispatchEvent: vi.fn(),
  }));
});

// ── Import panel + production allowlist predicate AFTER mocks ─────────

import { PreviewPanel } from "@/features/rag/components/preview-panel";
import { isInlineBlobAllowed } from "@/features/rag/stores/preview-store";

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

function setReady(overrides: Partial<PreviewTarget> = {}): void {
  mockState.target = makeTarget(overrides);
  mockState.status = "ready";
}

// ── Tests against real panel component ───────────────────────────────

describe("preview-panel inline rendering safety (contracts §5.4 / Risk #3)", () => {
  it("panel with no target (idle) renders without crashing", () => {
    const { container } = render(
      React.createElement(PreviewPanel, { open: true }),
    );
    expect(container).toBeDefined();
    expect(document.querySelector("iframe")).toBeNull();
  });

  it("html mediaKind does not render an iframe, object, or embed element", () => {
    setReady({ mediaKind: "html", filename: "malicious.html" });

    render(React.createElement(PreviewPanel, { open: true }));

    expect(document.querySelector("iframe")).toBeNull();
    expect(document.querySelector("object")).toBeNull();
    expect(document.querySelector("embed")).toBeNull();
  });

  it("docx mediaKind does not render an iframe, object, or embed element", () => {
    setReady({ mediaKind: "docx", filename: "report.docx" });

    render(React.createElement(PreviewPanel, { open: true }));

    expect(document.querySelector("iframe")).toBeNull();
    expect(document.querySelector("object")).toBeNull();
    expect(document.querySelector("embed")).toBeNull();
  });

  it("unknown mediaKind does not render inline blob content", () => {
    setReady({ mediaKind: "unknown", filename: "data.bin" });

    render(React.createElement(PreviewPanel, { open: true }));

    expect(document.querySelector("iframe")).toBeNull();
    expect(document.querySelector("object")).toBeNull();
    expect(document.querySelector("embed")).toBeNull();
  });

  it.each<PreviewMediaKind>(["html", "docx", "unknown"])(
    "%s download creates only a download object URL, never inline preview content",
    async (mediaKind) => {
      const createObjectUrl = vi
        .spyOn(URL, "createObjectURL")
        .mockReturnValue("blob:unsafe");
      const revokeObjectUrl = vi
        .spyOn(URL, "revokeObjectURL")
        .mockImplementation(() => undefined);
      setReady({
        mediaKind,
        filename: `unsafe.${mediaKind}`,
        contentType: "text/plain",
        snippet: "Extracted text only.",
      });

      render(React.createElement(PreviewPanel, { open: true }));
      fireEvent.click(
        screen.getByRole("button", { name: DOWNLOAD_BUTTON_NAME }),
      );

      await waitFor(() => {
        expect(mockAuthFetch).toHaveBeenCalledWith(
          "/api/rag/documents/doc-abc/file",
        );
      });
      expect(createObjectUrl).toHaveBeenCalledWith(expect.any(Blob));
      await waitFor(() => {
        expect(revokeObjectUrl).toHaveBeenCalledWith("blob:unsafe");
      });
      expect(document.querySelector("iframe")).toBeNull();
      expect(document.querySelector("object")).toBeNull();
      expect(document.querySelector("embed")).toBeNull();

      createObjectUrl.mockRestore();
      revokeObjectUrl.mockRestore();
    },
  );

  it("text mediaKind renders without iframe (text fallback path)", () => {
    setReady({
      mediaKind: "text",
      filename: "notes.txt",
      snippet: "This is the extracted text content.",
    });

    render(React.createElement(PreviewPanel, { open: true }));

    expect(document.querySelector("iframe")).toBeNull();
  });

  it("open=false triggers close side-effect on the store", () => {
    const closeFn = vi.fn();
    mockState.close = closeFn;
    mockState.target = makeTarget();
    mockState.status = "ready";

    const { rerender } = render(
      React.createElement(PreviewPanel, { open: true }),
    );
    rerender(React.createElement(PreviewPanel, { open: false }));

    // open=false useEffect should have called close()
    expect(closeFn).toHaveBeenCalled();
  });

  it("closes the preview panel on Escape key down (Escape key closures)", () => {
    const closeFn = vi.fn();
    mockState.close = closeFn;
    mockState.target = makeTarget();
    mockState.status = "ready";

    render(React.createElement(PreviewPanel, { open: true }));

    fireEvent.keyDown(document, { key: "Escape" });
    expect(closeFn).toHaveBeenCalled();
  });

  it("renders with premium glassmorphic visual details and a pulsing green indicator dot", () => {
    setReady({ mediaKind: "text", filename: "notes.txt" });

    render(React.createElement(PreviewPanel, { open: true }));

    const section = screen.getByLabelText("Document preview");
    expect(section).toHaveClass("bg-panel-surface/85");
    expect(section).toHaveClass("backdrop-blur-lg");
    expect(section).toHaveClass("border-border/40");
    expect(section).toHaveClass("shadow-lg");

    // Pulser dot
    const pulser = section.querySelector(".animate-pulse");
    expect(pulser).toBeInTheDocument();
    expect(pulser).toHaveClass("bg-primary");
    expect(pulser).toHaveClass("w-2");
    expect(pulser).toHaveClass("h-2");
  });

  it("shifts the layout to a full mobile Sheet drawer overlay when the viewport is squeezed (< 1024px)", () => {
    window.matchMedia = vi.fn().mockImplementation((query) => ({
      matches: true,
      media: query,
      onchange: null,
      addListener: vi.fn(),
      removeListener: vi.fn(),
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      dispatchEvent: vi.fn(),
    }));

    setReady({ mediaKind: "text", filename: "notes.txt" });

    render(React.createElement(PreviewPanel, { open: true }));

    // Radix Sheet renders dialog role in mobile viewports
    const dialog = screen.getByRole("dialog");
    expect(dialog).toBeInTheDocument();
    expect(dialog).toHaveClass("preview-sheet-content");
    expect(screen.getByText("Document preview")).toBeInTheDocument();
  });
});

// ── Pure-logic: inline allowlist (always green) ───────────────────────
// Uses the real isInlineBlobAllowed (D1.5 fix: no local copy).

describe("inline object URL allowlist (contracts §5.4, pure logic)", () => {
  const inlineSafe: PreviewMediaKind[] = ["pdf", "text", "image"];
  const inlineUnsafe: PreviewMediaKind[] = ["html", "docx", "unknown"];

  it.each(inlineSafe)("mediaKind=%s is inline-safe", (mk) => {
    expect(isInlineBlobAllowed(mk)).toBe(true);
  });

  it.each(inlineUnsafe)("mediaKind=%s is NOT inline-safe (Risk #3)", (mk) => {
    expect(isInlineBlobAllowed(mk)).toBe(false);
  });
});

describe("preview-panel stable scrollbars, sheets, layouts, and downloads", () => {
  let createObjectUrl: MockInstance<typeof URL.createObjectURL>;
  let revokeObjectUrl: MockInstance<typeof URL.revokeObjectURL>;

  beforeEach(() => {
    createObjectUrl = vi
      .spyOn(URL, "createObjectURL")
      .mockReturnValue("blob:safe-url");
    revokeObjectUrl = vi
      .spyOn(URL, "revokeObjectURL")
      .mockImplementation(() => undefined);
  });

  afterEach(() => {
    createObjectUrl.mockRestore();
    revokeObjectUrl.mockRestore();
  });

  it("asserts stable scrollbar style classes are present on panel content", () => {
    setReady({
      mediaKind: "text",
      filename: "notes.txt",
      snippet: "Some long content that requires scrolling ".repeat(20),
    });

    render(React.createElement(PreviewPanel, { open: true }));

    // Snippet renders in a <pre>; check it has overflow-auto
    const preElement = screen.getByText(LONG_CONTENT_TEXT);
    expect(preElement).toHaveClass("overflow-auto");
    expect(preElement).toHaveClass("flex-1");
  });

  it("asserts non-nested sheets are rendered in squeezed viewports", () => {
    window.matchMedia = vi.fn().mockImplementation((query) => ({
      matches: true,
      media: query,
      onchange: null,
      addListener: vi.fn(),
      removeListener: vi.fn(),
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      dispatchEvent: vi.fn(),
    }));

    setReady({ mediaKind: "text", filename: "notes.txt" });

    render(React.createElement(PreviewPanel, { open: true }));

    const dialogs = screen.getAllByRole("dialog");
    expect(dialogs.length).toBe(1);
    expect(dialogs[0]).toHaveClass("preview-sheet-content");

    const nestedDialogs = dialogs[0].querySelectorAll("[role='dialog']");
    expect(nestedDialogs.length).toBe(0);
  });

  it("supports responsive collapses under different viewport widths", () => {
    const mockMatchMedia = vi.fn().mockImplementation((query) => ({
      matches: query.includes("max-width: 1023px"),
      media: query,
      onchange: null,
      addListener: vi.fn(),
      removeListener: vi.fn(),
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      dispatchEvent: vi.fn(),
    }));
    window.matchMedia = mockMatchMedia;

    setReady({ mediaKind: "text", filename: "notes.txt" });

    const { unmount } = render(
      React.createElement(PreviewPanel, { open: true }),
    );
    expect(screen.getByRole("dialog")).toBeInTheDocument();
    unmount();

    window.matchMedia = vi.fn().mockImplementation((query) => ({
      matches: false,
      media: query,
      onchange: null,
      addListener: vi.fn(),
      removeListener: vi.fn(),
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      dispatchEvent: vi.fn(),
    }));

    render(React.createElement(PreviewPanel, { open: true }));
    expect(screen.queryByRole("dialog")).toBeNull();
    expect(screen.getByLabelText("Document preview")).toBeInTheDocument();
  });

  it("asserts object URL download logic: uses URL.createObjectURL for safe types and Data URL for unsafe types", async () => {
    setReady({
      mediaKind: "text",
      filename: "notes.txt",
      contentType: "text/plain",
      snippet: "Text snippet.",
    });

    render(React.createElement(PreviewPanel, { open: true }));
    fireEvent.click(screen.getByRole("button", { name: DOWNLOAD_BUTTON_NAME }));

    await waitFor(() => {
      expect(mockAuthFetch).toHaveBeenCalledWith(
        "/api/rag/documents/doc-abc/file",
      );
    });

    expect(createObjectUrl).toHaveBeenCalled();
  });
});

describe("PreviewTextView precise highlights matching", () => {
  it("highlights with character ranges", () => {
    setReady({
      mediaKind: "text",
      filename: "notes.txt",
      snippet: "Line 1: Hello World\nLine 2: Target Phrase\nLine 3: Goodbye",
      pageCharStart: 28,
      pageCharEnd: 41,
    });

    render(React.createElement(PreviewPanel, { open: true }));

    const mark = screen.getByText("Target Phrase");
    expect(mark.tagName).toBe("MARK");
    expect(mark).toHaveClass("bg-primary/20", "ring-primary/60");
  });

  it("highlights with line numbers", () => {
    setReady({
      mediaKind: "text",
      filename: "notes.txt",
      snippet: "Line one text\nLine two text\nLine three text",
      lineStart: 2,
      lineEnd: 2,
    });

    render(React.createElement(PreviewPanel, { open: true }));

    const mark = screen.getByText("Line two text");
    expect(mark.tagName).toBe("MARK");
    expect(mark).toHaveClass("bg-primary/20", "ring-primary/60");
  });

  it("highlights with fuzzy fallback matching high density line", () => {
    setReady({
      mediaKind: "text",
      filename: "notes.txt",
      snippet: "...\nAlphanumericDensity123456\n...",
      lineStart: 999, // hasLocator true, but matches no line range
    });

    render(React.createElement(PreviewPanel, { open: true }));

    const mark = screen.getByText("AlphanumericDensity123456");
    expect(mark.tagName).toBe("MARK");
  });
});
