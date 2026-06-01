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
} from "lucide-react";

import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import { Button } from "@/components/ui/button";
import { getDocumentFileUrl, getPreviewTarget } from "../api/rag-api";
import type { PdfRegion, PreviewTarget } from "../types/rag";
import { useDocumentPreviewStore } from "./preview-store";

// Resolve the pdf.js worker through Vite's ?url asset handling so the worker
// is bundled and served from the app origin (works under the Vite dev server,
// the production build, and the Tauri webview alike).
pdfjs.GlobalWorkerOptions.workerSrc = new URL(
  "pdfjs-dist/build/pdf.worker.min.mjs",
  import.meta.url,
).toString();

/** Overlay rectangles for a page; coords are normalized 0..1 of the page box. */
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
  const [error, setError] = useState<string | null>(null);

  // Reset to the cited page whenever a new citation opens this same viewer.
  useEffect(() => setPage(initialPage), [initialPage, fileUrl]);

  // Track the available width so the page scales to the panel.
  useLayoutEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const measure = () => setWidth(el.clientWidth);
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const onLoad = useCallback(
    ({ numPages: n }: { numPages: number }) => {
      setNumPages(n);
      setPage((p) => Math.min(Math.max(p, 1), n));
    },
    [],
  );

  // Show only the regions that live on the page currently in view.
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
        className="flex-1 overflow-auto bg-muted/30 px-4 py-3"
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
            <div className="relative mx-auto w-fit shadow-sm">
              <Page
                pageNumber={page}
                width={width - 8}
                renderTextLayer={false}
                renderAnnotationLayer={false}
              />
              <RegionOverlay regions={pageRegions} />
            </div>
          )}
        </Document>
      </div>
      {numPages > 1 && (
        <div className="flex items-center justify-center gap-3 border-t px-4 py-2 text-xs">
          <Button
            variant="ghost"
            size="icon"
            className="size-7"
            disabled={page <= 1}
            onClick={() => setPage((p) => Math.max(1, p - 1))}
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
          >
            <ChevronRightIcon className="size-4" />
          </Button>
        </div>
      )}
    </div>
  );
}

/**
 * Single shared document preview panel. Citation badges call
 * `useDocumentPreviewStore.openPreview` to point it at a document + chunk;
 * this resolves the page + highlight regions and renders the source PDF (or
 * the chunk text for non-PDF formats).
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
          <SheetTitle className="flex items-center gap-2 text-sm">
            <FileTextIcon className="size-4 shrink-0" />
            <span className="truncate">{headerName}</span>
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
