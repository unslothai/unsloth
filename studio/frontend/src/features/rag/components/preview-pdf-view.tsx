// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { copyToClipboard } from "@/lib/copy-to-clipboard";
import { cn } from "@/lib/utils";
import {
  ChevronLeftIcon,
  ChevronRightIcon,
  CopyIcon,
  LoaderIcon,
  RotateCcwIcon,
  SearchIcon,
  ZoomInIcon,
  ZoomOutIcon,
} from "lucide-react";
import {
  type CSSProperties,
  type FC,
  useCallback,
  useEffect,
  useId,
  useMemo,
  useRef,
  useState,
} from "react";
import { Document, Page, pdfjs } from "react-pdf";
import "react-pdf/dist/Page/AnnotationLayer.css";
import "react-pdf/dist/Page/TextLayer.css";
import type { PreviewPdfRegion, PreviewTarget } from "../api/rag-api";
import { PreviewUnavailable } from "./preview-unavailable";

// Configure pdfjs worker in the same module where react-pdf is used,
// per the react-pdf README. `import.meta.url` resolves to the JS bundle
// containing this module, and Vite (+ Tauri) rewrites the URL during
// build so the worker is co-located with the chunk.
pdfjs.GlobalWorkerOptions.workerSrc = new URL(
  "pdfjs-dist/build/pdf.worker.min.mjs",
  import.meta.url,
).toString();

type PreviewPdfFile = Blob | string;

interface PreviewPdfViewProps {
  target: PreviewTarget;
  file: PreviewPdfFile;
}

type LoadSuccess = { numPages: number };
type PdfLightThemeStyle = CSSProperties & Record<`--${string}`, string>;

const RESIZE_DEBOUNCE_MS = 100;
const MIN_PDF_WIDTH = 280;
// Body has p-2 (8px each side) + stable scrollbar gutter (~10px) + a tiny
// breathing margin so the page render doesn't kiss the scrollbar.
const PDF_BODY_GUTTER_PX = 28;
const PDF_THUMBNAIL_WIDTH = 64;

const PDF_LIGHT_THEME_STYLE: PdfLightThemeStyle = {
  "--background": "oklch(1 0 0)",
  "--foreground": "oklch(0.2686 0 0)",
  "--card": "oklch(1 0 0)",
  "--card-foreground": "oklch(0.1281 0.0179 169.2764)",
  "--popover": "oklch(1 0 0)",
  "--popover-foreground": "oklch(0.1281 0.0179 169.2764)",
  "--primary": "#17b88b",
  "--primary-foreground": "oklch(1 0 0)",
  "--secondary": "oklch(0.9596 0.0275 167.8295)",
  "--secondary-foreground": "oklch(0.2868 0.0649 159.9823)",
  "--muted": "oklch(0.9702 0 0)",
  "--muted-foreground": "oklch(0.5486 0 0)",
  "--accent": "oklch(0.9596 0.0275 167.8295)",
  "--accent-foreground": "oklch(0.2868 0.0649 159.9823)",
  "--border": "oklch(0.9208 0.0101 164.8536)",
  "--input": "oklch(0.9208 0.0101 164.8536)",
  "--ring": "#17b88b",
  colorScheme: "light",
};

function escapeHtml(value: string): string {
  return value
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function markFirstMatch(text: string, needle: string): string | null {
  const trimmed = needle.trim();
  if (trimmed.length < 2) {
    return null;
  }
  const lower = text.toLowerCase();
  const start = lower.indexOf(trimmed.toLowerCase());
  if (start < 0) {
    return null;
  }
  const end = start + trimmed.length;
  return `${escapeHtml(text.slice(0, start))}<mark>${escapeHtml(
    text.slice(start, end),
  )}</mark>${escapeHtml(text.slice(end))}`;
}

// Keep text-layer highlighting opt-in. Citation snippets render in the card
// below; using them here would mark common words across unrelated PDF text.
function highlightPdfText(text: string, searchTerm: string): string {
  const trimmed = searchTerm.trim();
  if (trimmed.length < 2) {
    return escapeHtml(text);
  }
  const searchHit = markFirstMatch(text, trimmed);
  if (searchHit) {
    return searchHit;
  }
  return escapeHtml(text);
}

function regionIsOnPage(region: PreviewPdfRegion, pageNumber: number): boolean {
  if (region.confidence !== "exact") {
    return false;
  }
  if (region.pageNumber != null) {
    return region.pageNumber === pageNumber;
  }
  return region.pageIndex === pageNumber - 1;
}

interface PdfThumbnailProps {
  pageNumber: number;
  active: boolean;
  onSelect: (pageNumber: number) => void;
}

/** Lazy thumbnail rendered via IntersectionObserver — only mounts the
 *  inner <Page> when scrolled into view (or close to it), so large PDFs
 *  stay responsive even when the rail caps at 80 buttons. */
const PdfThumbnail: FC<PdfThumbnailProps> = ({
  pageNumber,
  active,
  onSelect,
}) => {
  const buttonRef = useRef<HTMLButtonElement | null>(null);
  const [shouldRender, setShouldRender] = useState(false);

  useEffect(() => {
    if (shouldRender) {
      return;
    }
    const el = buttonRef.current;
    if (!el || typeof IntersectionObserver === "undefined") {
      // Fallback for jsdom / older browsers: render eagerly.
      setShouldRender(true);
      return;
    }
    const observer = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (entry.isIntersecting) {
            setShouldRender(true);
            observer.disconnect();
            return;
          }
        }
      },
      { rootMargin: "320px" },
    );
    observer.observe(el);
    return () => observer.disconnect();
  }, [shouldRender]);

  useEffect(() => {
    const el = buttonRef.current;
    if (!active || !el || typeof el.scrollIntoView !== "function") {
      return;
    }
    el.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }, [active]);

  return (
    <button
      ref={buttonRef}
      type="button"
      onClick={() => onSelect(pageNumber)}
      aria-label={`Go to page ${pageNumber}`}
      aria-current={active ? "page" : undefined}
      className={cn(
        "mb-1.5 flex w-full flex-col items-center gap-0.5 rounded-md p-1 outline-none transition-colors",
        "focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-1",
        active
          ? "bg-secondary/70 text-secondary-foreground"
          : "hover:bg-muted/60",
      )}
    >
      <div
        className={cn(
          "overflow-hidden rounded-sm border bg-white shadow-xs",
          active
            ? "border-primary/70 ring-1 ring-primary/40"
            : "border-border/60",
        )}
        style={{
          width: PDF_THUMBNAIL_WIDTH,
          minHeight: Math.round(PDF_THUMBNAIL_WIDTH * 1.3),
        }}
      >
        {shouldRender ? (
          <Page
            pageNumber={pageNumber}
            width={PDF_THUMBNAIL_WIDTH}
            renderTextLayer={false}
            renderAnnotationLayer={false}
            loading={
              <div
                className="flex h-full w-full animate-pulse items-center justify-center bg-muted/40"
                style={{
                  minHeight: Math.round(PDF_THUMBNAIL_WIDTH * 1.3),
                }}
              />
            }
            error={
              <div
                className="flex h-full w-full items-center justify-center text-[8px] text-muted-foreground"
                style={{
                  minHeight: Math.round(PDF_THUMBNAIL_WIDTH * 1.3),
                }}
              >
                ?
              </div>
            }
            className="pointer-events-none [&_canvas]:!h-auto [&_canvas]:!w-full"
          />
        ) : null}
      </div>
      <span
        className={cn(
          "tabular-nums text-[10px]",
          active ? "font-semibold" : "text-muted-foreground",
        )}
      >
        {pageNumber}
      </span>
    </button>
  );
};

export const PreviewPdfView: FC<PreviewPdfViewProps> = ({ target, file }) => {
  const [numPages, setNumPages] = useState<number | null>(null);
  const [pageNumber, setPageNumber] = useState<number>(target.targetPage ?? 1);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [zoom, setZoom] = useState(1);
  const [searchTerm, setSearchTerm] = useState("");
  const [copied, setCopied] = useState(false);
  const containerRef = useRef<HTMLDivElement | null>(null);
  const observerRef = useRef<ResizeObserver | null>(null);
  const resizeTimeoutRef = useRef<number | null>(null);
  const lastMeasuredWidthRef = useRef<number | null>(null);
  const lastResetKeyRef = useRef<string | null>(null);
  const [width, setWidth] = useState<number | null>(null);
  const searchInputId = useId();

  const sourceKey =
    typeof file === "string"
      ? file
      : `${target.documentId}:${target.chunkId ?? ""}:${file.size}:${file.type}`;

  const documentFile = useMemo(() => {
    return typeof file === "string" ? { url: file } : file;
  }, [file]);
  const resetKey = `${sourceKey}:${target.targetPage ?? ""}`;

  useEffect(() => {
    if (lastResetKeyRef.current === resetKey) {
      return;
    }
    lastResetKeyRef.current = resetKey;
    setNumPages(null);
    setLoadError(null);
    setPageNumber(target.targetPage ?? 1);
    setZoom(1);
    setSearchTerm("");
    setCopied(false);
  }, [resetKey, target.targetPage]);

  const measureWidth = useCallback(() => {
    const el = containerRef.current;
    if (!el) {
      return;
    }
    const next = Math.max(MIN_PDF_WIDTH, el.clientWidth - PDF_BODY_GUTTER_PX);
    if (lastMeasuredWidthRef.current === next) {
      return;
    }
    lastMeasuredWidthRef.current = next;
    setWidth(next);
  }, []);

  // Callback ref instead of useRef + mount effect: the scroll container
  // lives INSIDE <Document>, so it only enters the DOM after the PDF
  // loads. Attaching the ResizeObserver the instant the node mounts
  // (rather than on the component's mount effect, when the node is still
  // absent) is what keeps the main page from rendering at width 0 — the
  // thin white strip regression.
  const attachContainer = useCallback(
    (node: HTMLDivElement | null) => {
      if (observerRef.current) {
        observerRef.current.disconnect();
        observerRef.current = null;
      }
      if (resizeTimeoutRef.current !== null) {
        window.clearTimeout(resizeTimeoutRef.current);
        resizeTimeoutRef.current = null;
      }
      containerRef.current = node;
      if (!node) {
        return;
      }
      measureWidth();
      const observer = new ResizeObserver(() => {
        if (resizeTimeoutRef.current !== null) {
          window.clearTimeout(resizeTimeoutRef.current);
        }
        resizeTimeoutRef.current = window.setTimeout(
          measureWidth,
          RESIZE_DEBOUNCE_MS,
        );
      });
      observer.observe(node);
      observerRef.current = observer;
    },
    [measureWidth],
  );

  useEffect(() => {
    if (!copied) {
      return;
    }
    const id = window.setTimeout(() => setCopied(false), 1200);
    return () => window.clearTimeout(id);
  }, [copied]);

  const handleLoadSuccess = useCallback(({ numPages }: LoadSuccess) => {
    setNumPages(numPages);
    setLoadError(null);
  }, []);

  const handleLoadError = useCallback((err: Error) => {
    setLoadError(err.message || "Failed to load PDF");
  }, []);

  const goPrev = useCallback(() => {
    setPageNumber((p) => Math.max(1, p - 1));
  }, []);

  const goNext = useCallback(() => {
    setPageNumber((p) =>
      numPages == null ? p + 1 : Math.min(numPages, p + 1),
    );
  }, [numPages]);

  const textRenderer = useCallback(
    ({ str }: { str: string }) => highlightPdfText(str, searchTerm),
    [searchTerm],
  );

  const currentRegions = useMemo(
    () =>
      (target.pdfRegions ?? []).filter((region) =>
        regionIsOnPage(region, pageNumber),
      ),
    [target.pdfRegions, pageNumber],
  );

  const visiblePageNumbers = useMemo(() => {
    if (!numPages) {
      return [];
    }
    const maxButtons = 80;
    if (numPages <= maxButtons) {
      return Array.from({ length: numPages }, (_, index) => index + 1);
    }
    const half = Math.floor(maxButtons / 2);
    let start = Math.max(1, pageNumber - half);
    const end = Math.min(numPages, start + maxButtons - 1);
    start = Math.max(1, end - maxButtons + 1);
    return Array.from({ length: end - start + 1 }, (_, index) => start + index);
  }, [numPages, pageNumber]);

  const pageWidth = width == null ? null : Math.round(width * zoom);
  const excerptKey = `${sourceKey}:${target.chunkId ?? ""}:${
    target.targetPage ?? ""
  }:${pageNumber}`;

  const copyExcerpt = useCallback(() => {
    copyToClipboard(target.snippet ?? "").then(setCopied);
  }, [target.snippet]);

  if (loadError) {
    return (
      <PreviewUnavailable
        filename={target.filename}
        reason={loadError}
        variant="error"
      />
    );
  }

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-border/60 px-3 py-2 text-xs">
        <span
          className="min-w-0 flex-1 truncate font-semibold font-heading"
          title={target.filename}
        >
          {target.filename}
        </span>
        <div className="flex shrink-0 items-center gap-2">
          {/* Navigation Pill Group */}
          <div className="flex items-center rounded-full border border-border/60 bg-muted/40 p-0.5 shadow-xs">
            <Button
              variant="ghost"
              size="icon"
              onClick={goPrev}
              disabled={pageNumber <= 1}
              aria-label="Previous page"
              className="h-7 w-7 rounded-full hover:bg-background/80"
            >
              <ChevronLeftIcon className="size-3.5" />
            </Button>
            <span className="min-w-12 text-center tabular-nums text-[10px] font-medium text-muted-foreground">
              {numPages == null
                ? `${pageNumber}/?`
                : `${pageNumber}/${numPages}`}
            </span>
            <Button
              variant="ghost"
              size="icon"
              onClick={goNext}
              disabled={numPages != null && pageNumber >= numPages}
              aria-label="Next page"
              className="h-7 w-7 rounded-full hover:bg-background/80"
            >
              <ChevronRightIcon className="size-3.5" />
            </Button>
          </div>

          {/* Zoom Pill Group */}
          <div className="flex items-center rounded-full border border-border/60 bg-muted/40 p-0.5 shadow-xs">
            <Button
              variant="ghost"
              size="icon"
              onClick={() => setZoom((value) => Math.max(0.6, value - 0.1))}
              aria-label="Zoom out"
              className="h-7 w-7 rounded-full hover:bg-background/80"
            >
              <ZoomOutIcon className="size-3.5" />
            </Button>
            <Button
              variant="ghost"
              size="icon"
              onClick={() => setZoom(1)}
              aria-label="Reset zoom"
              className="h-7 w-7 rounded-full hover:bg-background/80"
            >
              <RotateCcwIcon className="size-3.5" />
            </Button>
            <Button
              variant="ghost"
              size="icon"
              onClick={() => setZoom((value) => Math.min(2.5, value + 0.1))}
              aria-label="Zoom in"
              className="h-7 w-7 rounded-full hover:bg-background/80"
            >
              <ZoomInIcon className="size-3.5" />
            </Button>
          </div>

          {/* Copy Excerpt Pill Group */}
          <div className="flex items-center rounded-full border border-border/60 bg-muted/40 p-0.5 shadow-xs">
            <Button
              variant="ghost"
              size="icon"
              onClick={copyExcerpt}
              disabled={!target.snippet}
              aria-label={
                copied ? "Copied source excerpt" : "Copy source excerpt"
              }
              className="h-7 w-7 rounded-full hover:bg-background/80"
            >
              <CopyIcon className="size-3.5" />
            </Button>
          </div>
        </div>
        <label
          htmlFor={searchInputId}
          className="flex min-w-48 max-w-full flex-1 items-center gap-1 rounded-md border border-border/60 bg-background px-2"
        >
          <SearchIcon className="size-3.5 shrink-0 text-muted-foreground" />
          <Input
            id={searchInputId}
            value={searchTerm}
            onChange={(event) => setSearchTerm(event.target.value)}
            placeholder="Search this PDF"
            aria-label="Search this PDF"
            className="h-7 border-0 bg-transparent px-0 text-xs shadow-none focus-visible:ring-0"
          />
        </label>
      </div>

      {target.snippet ? (
        <div
          key={excerptKey}
          className="m-2 rounded-lg border border-border/60 bg-muted/30 p-3 shadow-xs text-[11px] leading-relaxed text-foreground/80 transition-all duration-300 animate-in fade-in"
        >
          <p className="mb-1 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground/80">
            Source Excerpt
            {target.targetPage != null ? ` · Page ${target.targetPage}` : ""}
          </p>
          <p className="line-clamp-4 whitespace-pre-wrap font-sans text-muted-foreground">
            {target.snippet}
          </p>
        </div>
      ) : null}

      <Document
        file={documentFile}
        onLoadSuccess={handleLoadSuccess}
        onLoadError={handleLoadError}
        loading={
          <div className="flex h-full items-center justify-center gap-2 text-xs text-muted-foreground">
            <LoaderIcon className="size-3.5 animate-spin" />
            Loading PDF...
          </div>
        }
        error={
          <PreviewUnavailable
            filename={target.filename}
            reason="The PDF could not be opened."
            variant="error"
          />
        }
        className="flex min-h-0 flex-1"
      >
        <div className="preview-scrollbar w-[88px] shrink-0 overflow-y-auto border-r border-border/60 bg-muted/20 p-1.5">
          {visiblePageNumbers.map((page) => (
            <PdfThumbnail
              key={page}
              pageNumber={page}
              active={page === pageNumber}
              onSelect={setPageNumber}
            />
          ))}
        </div>
        <div
          ref={attachContainer}
          className="preview-scrollbar flex-1 overflow-y-scroll overflow-x-auto bg-muted/20 p-2 [scrollbar-gutter:stable]"
        >
          <div
            className="light [color-scheme:light] bg-white text-slate-900 rounded-md p-1 shadow-sm border border-border/30 [&_mark]:bg-primary/20 [&_mark]:text-slate-900 [&_mark]:ring-1 [&_mark]:ring-primary/60 [&_mark]:rounded-xs flex min-w-fit flex-col items-center"
            style={PDF_LIGHT_THEME_STYLE}
          >
            {pageWidth != null ? (
              <div
                data-testid="pdf-main-page"
                className="relative inline-block"
              >
                <Page
                  pageNumber={pageNumber}
                  width={pageWidth}
                  customTextRenderer={textRenderer}
                  renderTextLayer={true}
                  renderAnnotationLayer={false}
                  loading={
                    <div className="py-4 text-xs text-muted-foreground">
                      Rendering page...
                    </div>
                  }
                  className="shadow-sm"
                />
                {currentRegions.map((region, index) => (
                  <div
                    key={`${region.pageIndex}-${region.x}-${region.y}-${index}`}
                    data-testid="pdf-region-highlight"
                    className="pointer-events-none absolute rounded-sm bg-primary/20 ring-1 ring-primary/60"
                    style={{
                      left: `${region.x * 100}%`,
                      top: `${region.y * 100}%`,
                      width: `${region.width * 100}%`,
                      height: `${region.height * 100}%`,
                    }}
                  />
                ))}
              </div>
            ) : null}
          </div>
        </div>
      </Document>
    </div>
  );
};
