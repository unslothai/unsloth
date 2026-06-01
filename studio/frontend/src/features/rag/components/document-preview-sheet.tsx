// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"use client";

import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";
import { Document, Page, pdfjs } from "react-pdf";
import {
  ChevronLeftIcon,
  ChevronRightIcon,
  FileTextIcon,
  LoaderIcon,
  ZoomInIcon,
  ZoomOutIcon,
} from "lucide-react";

import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { getDocumentFileUrl, getPreviewTarget } from "../api/rag-api";
import type { PdfRegion, PreviewTarget } from "../types/rag";
import { useDocumentPreviewStore } from "./preview-store";

// Serve the pdf.js worker from the app origin (Vite dev, prod, Tauri).
pdfjs.GlobalWorkerOptions.workerSrc = new URL(
  "pdfjs-dist/build/pdf.worker.min.mjs",
  import.meta.url,
).toString();

/** Highlight rects for a page; coords normalized 0..1 of the page box. */
function RegionOverlay({ regions }: { regions: PdfRegion[] }) {
  if (regions.length === 0) return null;
  return (
    <>
      {regions.map((r, i) => (
        <div
          key={i}
          className="pointer-events-none absolute rounded-sm bg-amber-300/35 ring-1 ring-amber-500/70 mix-blend-multiply"
          style={{
            left: `${r.x * 100}%`,
            top: `${r.y * 100}%`,
            width: `${r.width * 100}%`,
            height: `${r.height * 100}%`,
          }}
        />
      ))}
    </>
  );
}

// Zoom multiplies fit-to-panel width: 1 = fit, >1 enlarges (page scrolls), <1
// shrinks. Stepped so the buttons and wheel agree.
const ZOOM_MIN = 0.5;
const ZOOM_MAX = 3;
const ZOOM_STEP = 0.25;

const clampZoom = (z: number) =>
  Math.min(ZOOM_MAX, Math.max(ZOOM_MIN, Number(z.toFixed(2))));

function PdfPreview({
  fileUrl,
  initialPage,
  regions,
}: {
  fileUrl: string;
  initialPage: number;
  regions: PdfRegion[];
}) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [width, setWidth] = useState(0);
  const [numPages, setNumPages] = useState(0);
  const [page, setPage] = useState(initialPage);
  const [scale, setScale] = useState(1);
  const [error, setError] = useState<string | null>(null);
  const [grabbing, setGrabbing] = useState(false);
  const [scrollable, setScrollable] = useState(false);
  const panRef = useRef<{
    x: number;
    y: number;
    left: number;
    top: number;
  } | null>(null);

  // Reset to the cited page when a new citation reuses this viewer.
  useEffect(() => setPage(initialPage), [initialPage, fileUrl]);

  // Reset each newly opened document to fit-to-panel.
  useEffect(() => setScale(1), [fileUrl]);

  // Track available width so the page scales to the panel.
  useLayoutEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const measure = () => setWidth(el.clientWidth);
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  // Wheel zooms (scroll up = in, down = out); drag-to-pan moves a zoomed page.
  // Native non-passive listener so preventDefault stops the panel from scrolling.
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      setScale((s) => clampZoom(s - Math.sign(e.deltaY) * ZOOM_STEP));
    };
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, []);

  const onLoad = useCallback(
    ({ numPages: n }: { numPages: number }) => {
      setNumPages(n);
      setPage((p) => Math.min(Math.max(p, 1), n));
    },
    [],
  );

  const zoomBy = useCallback(
    (delta: number) => setScale((s) => clampZoom(s + delta)),
    [],
  );

  // Does the page overflow the panel (so panning/scrolling does anything)?
  // Re-checked on layout changes and once the canvas finishes rendering.
  const recheckScrollable = useCallback(() => {
    const el = containerRef.current;
    if (!el) return;
    setScrollable(
      el.scrollWidth > el.clientWidth + 1 ||
        el.scrollHeight > el.clientHeight + 1,
    );
  }, []);

  useEffect(() => {
    recheckScrollable();
  }, [recheckScrollable, width, scale, page, numPages]);

  // Grab-to-pan: hold and drag to move an overflowing page. Listen on window so
  // the drag keeps tracking when the cursor leaves the panel.
  useEffect(() => {
    if (!grabbing) return;
    const onMove = (e: MouseEvent) => {
      const el = containerRef.current;
      const start = panRef.current;
      if (!el || !start) return;
      el.scrollLeft = start.left - (e.clientX - start.x);
      el.scrollTop = start.top - (e.clientY - start.y);
    };
    const onUp = () => {
      setGrabbing(false);
      panRef.current = null;
    };
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
    return () => {
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
    };
  }, [grabbing]);

  const onPanStart = useCallback(
    (e: React.MouseEvent) => {
      const el = containerRef.current;
      if (!el || e.button !== 0 || !scrollable) return;
      panRef.current = {
        x: e.clientX,
        y: e.clientY,
        left: el.scrollLeft,
        top: el.scrollTop,
      };
      setGrabbing(true);
      e.preventDefault(); // stop canvas image-drag / selection
    },
    [scrollable],
  );

  // Only regions on the current page.
  const pageRegions = regions.filter(
    (r) => r.pageNumber === page || r.pageIndex === page - 1,
  );

  if (error) {
    return (
      <div className="p-6 text-sm text-muted-foreground">
        Could not render this PDF ({error}).
      </div>
    );
  }

  return (
    <div className="flex h-full flex-col">
      <div
        ref={containerRef}
        onMouseDown={onPanStart}
        className={cn(
          "flex-1 overflow-auto bg-muted/30 px-4 py-3",
          grabbing
            ? "cursor-grabbing select-none"
            : scrollable
              ? "cursor-grab"
              : "",
        )}
      >
        <Document
          file={fileUrl}
          onLoadSuccess={onLoad}
          onLoadError={(e) => setError(e.message)}
          loading={
            <div className="flex items-center gap-2 p-6 text-sm text-muted-foreground">
              <LoaderIcon className="size-3.5 animate-spin" /> Loading PDF…
            </div>
          }
        >
          {width > 0 && (
            // min-w-fit lets the row grow past the panel when zoomed so the
            // centered page stays reachable on both sides while scrolling.
            <div className="flex min-w-fit justify-center">
              <div className="relative w-fit shadow-sm">
                <Page
                  pageNumber={page}
                  width={(width - 8) * scale}
                  renderTextLayer={false}
                  renderAnnotationLayer={false}
                  onRenderSuccess={recheckScrollable}
                />
                <RegionOverlay regions={pageRegions} />
              </div>
            </div>
          )}
        </Document>
      </div>
      <div className="grid grid-cols-[1fr_auto_1fr] items-center border-t px-3 py-2 text-xs">
        <div className="flex items-center gap-0.5 justify-self-start">
          <Button
            variant="ghost"
            size="icon"
            className="size-7"
            disabled={scale <= ZOOM_MIN}
            onClick={() => zoomBy(-ZOOM_STEP)}
            aria-label="Zoom out"
          >
            <ZoomOutIcon className="size-4" />
          </Button>
          <button
            type="button"
            onClick={() => setScale(1)}
            className="w-11 text-center tabular-nums text-muted-foreground hover:text-foreground"
            aria-label="Reset zoom"
          >
            {Math.round(scale * 100)}%
          </button>
          <Button
            variant="ghost"
            size="icon"
            className="size-7"
            disabled={scale >= ZOOM_MAX}
            onClick={() => zoomBy(ZOOM_STEP)}
            aria-label="Zoom in"
          >
            <ZoomInIcon className="size-4" />
          </Button>
        </div>
        {numPages > 1 ? (
          <div className="flex items-center gap-3 justify-self-center">
            <Button
              variant="ghost"
              size="icon"
              className="size-7"
              disabled={page <= 1}
              onClick={() => setPage((p) => Math.max(1, p - 1))}
              aria-label="Previous page"
            >
              <ChevronLeftIcon className="size-4" />
            </Button>
            <span className="tabular-nums text-muted-foreground">
              Page {page} / {numPages}
            </span>
            <Button
              variant="ghost"
              size="icon"
              className="size-7"
              disabled={page >= numPages}
              onClick={() => setPage((p) => Math.min(numPages, p + 1))}
              aria-label="Next page"
            >
              <ChevronRightIcon className="size-4" />
            </Button>
          </div>
        ) : (
          <span />
        )}
        <span aria-hidden />
      </div>
    </div>
  );
}

/**
 * Shared preview panel. `openPreview` points it at a document + chunk; it
 * resolves the page + highlight regions and renders the PDF (or chunk text).
 */
export function DocumentPreviewSheet() {
  const { open, documentId, chunkId, filename, page, closePreview } =
    useDocumentPreviewStore();
  const [target, setTarget] = useState<PreviewTarget | null>(null);
  const [fileUrl, setFileUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!open || !documentId) return;
    let cancelled = false;
    setLoading(true);
    setError(null);
    setTarget(null);
    setFileUrl(null);
    (async () => {
      try {
        const t = await getPreviewTarget(documentId, chunkId ?? undefined);
        if (cancelled) return;
        setTarget(t);
        if (t.mediaKind === "pdf") {
          const url = await getDocumentFileUrl(documentId);
          if (!cancelled) setFileUrl(url);
        }
      } catch (e) {
        if (!cancelled) setError(e instanceof Error ? e.message : String(e));
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [open, documentId, chunkId]);

  const headerName = target?.filename ?? filename ?? "Document";
  const headerPage = target?.targetPage ?? page ?? null;

  return (
    <Sheet open={open} onOpenChange={(o) => !o && closePreview()}>
      <SheetContent
        side="right"
        className="flex w-full flex-col gap-0 p-0 sm:max-w-[44rem]"
      >
        <SheetHeader className="gap-1 border-b p-4">
          {/* pr-10 reserves room for the sheet's absolute close button so a long
          filename + page label never run under it. */}
          <SheetTitle className="flex items-center gap-2 pr-10 text-sm">
            <FileTextIcon className="size-4 shrink-0" />
            <span className="min-w-0 truncate">{headerName}</span>
            {headerPage != null && (
              <span className="shrink-0 text-muted-foreground">
                · page {headerPage}
              </span>
            )}
          </SheetTitle>
        </SheetHeader>

        <div className="min-h-0 flex-1">
          {loading ? (
            <div className="flex items-center gap-2 p-6 text-sm text-muted-foreground">
              <LoaderIcon className="size-3.5 animate-spin" /> Resolving source…
            </div>
          ) : error ? (
            <div className="p-6 text-sm text-muted-foreground">
              Could not open this document ({error}).
            </div>
          ) : target && target.mediaKind === "pdf" && fileUrl ? (
            <PdfPreview
              fileUrl={fileUrl}
              initialPage={target.targetPage ?? 1}
              regions={target.pdfRegions ?? []}
            />
          ) : target?.text ? (
            <div className="h-full overflow-auto p-5">
              <p className="whitespace-pre-wrap break-words text-sm leading-relaxed text-foreground/90">
                {target.text}
              </p>
            </div>
          ) : (
            <div className="p-6 text-sm text-muted-foreground">
              No preview available for this source.
            </div>
          )}
        </div>
      </SheetContent>
    </Sheet>
  );
}
