// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Spinner } from "@/components/ui/spinner";
import { useT } from "@/i18n";
import { useVirtualizer } from "@tanstack/react-virtual";
import { createContext, useCallback, useContext, useEffect, useRef, useState } from "react";
import { Document, Page, pdfjs, usePageContext } from "react-pdf";
import "react-pdf/dist/Page/TextLayer.css";
import { useWidth } from "./use-width";

pdfjs.GlobalWorkerOptions.workerSrc = new URL(
  "pdfjs-dist/build/pdf.worker.min.mjs",
  import.meta.url,
).toString();

const MAX_PAGE_WIDTH = 880;
const PAGE_GAP = 16;
const EDGE = 24;
const MAX_CANVAS_PIXELS = 32 * 1024 * 1024;
const MAX_CANVAS_SIDE = 16384;
// PDF.js skips larger images before decoding; a 600dpi letter/A4 scan still fits.
const PDF_OPTIONS = { maxImageSize: 64 * 1024 * 1024 };
const MIN_PIXEL_RATIO = 0.1;
const MAX_TEXT_ITEMS = 20_000;
const MAX_PAGE_OPERATIONS = 1_000_000;
const MAX_PDF_PAGES = 10_000;
const clampAspect = (aspect: number) => Math.min(Math.max(aspect, 0.05), 20);

type PdfDocument = {
  getPage(page: number): Promise<{
    getViewport(options: { scale: number }): { width: number; height: number };
    streamTextContent(): ReadableStream<{ items: unknown[] }>;
  }>;
};

const overflowed = new WeakMap<object, Set<number>>();
const PageOverflow = createContext<() => void>(() => {});

function PdfCanvas() {
  const context = usePageContext();
  const onOverflow = useContext(PageOverflow);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const page = context?.page;
  const scale = context?.scale ?? 1;
  const rotate = context?.rotate ?? 0;
  const ratio = context?.devicePixelRatio ?? 1;
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!page || !canvas) return;
    const viewport = page.getViewport({ scale: scale * ratio, rotation: rotate });
    const shown = page.getViewport({ scale, rotation: rotate });
    canvas.width = viewport.width;
    canvas.height = viewport.height;
    canvas.style.width = `${Math.floor(shown.width)}px`;
    canvas.style.height = `${Math.floor(shown.height)}px`;
    let over = false;
    const task = page.render({
      canvas,
      canvasContext: canvas.getContext("2d", { alpha: false })!,
      viewport,
      annotationMode: pdfjs.AnnotationMode.ENABLE,
      operationsFilter: (index) => {
        if (index < MAX_PAGE_OPERATIONS) return true;
        if (!over) {
          over = true;
          queueMicrotask(() => task.cancel());
        }
        return false;
      },
    });
    task.promise.catch(() => {
      if (!over) return;
      page.cleanup();
      onOverflow();
    });
    return () => {
      task.cancel();
      page.cleanup();
      canvas.width = 0;
      canvas.height = 0;
    };
  }, [page, scale, rotate, ratio, onOverflow]);
  return <canvas ref={canvasRef} className="block select-none" />;
}

function PdfPage({ pdf, index, width, aspect }: { pdf: PdfDocument; index: number; width: number; aspect: number }) {
  const t = useT();
  const [own, setOwn] = useState<{ index: number; aspect: number } | null>(null);
  useEffect(() => {
    let live = true;
    pdf.getPage(index + 1).then(
      (page) => {
        const viewport = page.getViewport({ scale: 1 });
        if (live) setOwn({ index, aspect: viewport.height / viewport.width });
      },
      () => live && setOwn({ index, aspect: Number.NaN }),
    );
    return () => {
      live = false;
    };
  }, [pdf, index]);
  const [selectable, setSelectable] = useState<number | null>(null);
  useEffect(() => {
    let live = true;
    let reader: ReadableStreamDefaultReader<{ items: unknown[] }> | undefined;
    void (async () => {
      reader = (await pdf.getPage(index + 1)).streamTextContent().getReader();
      let items = 0;
      for (let chunk = await reader.read(); !chunk.done; chunk = await reader.read()) {
        items += chunk.value.items.length;
        if (!live || items > MAX_TEXT_ITEMS) return;
      }
      if (live) setSelectable(index);
    })()
      .catch(() => {})
      .finally(() => void reader?.cancel().catch(() => {}));
    return () => {
      live = false;
    };
  }, [pdf, index]);
  const [tooLong, setTooLong] = useState<number | null>(null);
  const markOverflow = useCallback(() => {
    const pages = overflowed.get(pdf) ?? new Set<number>();
    overflowed.set(pdf, pages.add(index));
    setTooLong(index);
  }, [pdf, index]);
  const pageAspect = own?.index === index ? own.aspect : null;
  if (pageAspect === null) return <div style={{ height: width * aspect }} />;
  const height = width * pageAspect;
  const ratio = Math.min(
    window.devicePixelRatio || 1,
    Math.sqrt(MAX_CANVAS_PIXELS / (width * height)),
    MAX_CANVAS_SIDE / width,
    MAX_CANVAS_SIDE / height,
  );
  if (!(ratio >= MIN_PIXEL_RATIO) || tooLong === index || overflowed.get(pdf)?.has(index)) {
    return (
      <p className="flex items-center justify-center text-sm text-muted-foreground" style={{ height: width * clampAspect(pageAspect) }}>
        {t("library.preview.cannotPreview")}
      </p>
    );
  }
  return (
    <PageOverflow.Provider value={markOverflow}>
      <Page
        pageNumber={index + 1}
        width={width}
        devicePixelRatio={ratio}
        renderMode="custom"
        customRenderer={PdfCanvas}
        renderAnnotationLayer={false}
        renderTextLayer={selectable === index}
        loading={<div style={{ height }} />}
      />
    </PageOverflow.Provider>
  );
}

function PdfPages({
  pdf,
  pages,
  width,
  aspect,
  scrollElement,
}: {
  pdf: PdfDocument;
  pages: number;
  width: number;
  aspect: number;
  scrollElement: HTMLElement | null;
}) {
  // eslint-disable-next-line react-hooks/incompatible-library
  const virtualizer = useVirtualizer({
    count: pages,
    getScrollElement: () => scrollElement,
    estimateSize: () => width * aspect + PAGE_GAP,
    paddingStart: EDGE,
    paddingEnd: EDGE - PAGE_GAP,
    overscan: 2,
  });
  return (
    <div className="relative mx-auto" style={{ width, height: virtualizer.getTotalSize() }}>
      {virtualizer.getVirtualItems().map((item) => (
        <div
          key={item.key}
          data-index={item.index}
          ref={virtualizer.measureElement}
          className="absolute top-0 left-0 w-full"
          style={{ transform: `translateY(${item.start}px)`, paddingBottom: PAGE_GAP }}
        >
          <div className="bg-white shadow-sm ring-1 ring-black/5" style={{ minHeight: width * aspect }}>
            <PdfPage pdf={pdf} index={item.index} width={width} aspect={aspect} />
          </div>
        </div>
      ))}
    </div>
  );
}

export default function PdfView({
  file,
  scale,
  firstPageOnly = false,
}: {
  file: Blob;
  scale: number;
  firstPageOnly?: boolean;
}) {
  const t = useT();
  const [container, setContainer] = useState<HTMLDivElement | null>(null);
  const available = useWidth(container);
  const [pdf, setPdf] = useState<PdfDocument | null>(null);
  const [pages, setPages] = useState(0);
  const [aspect, setAspect] = useState(1.294);
  const [failed, setFailed] = useState<Blob | null>(null);
  const width = Math.max(200, Math.min(available, MAX_PAGE_WIDTH)) * scale;

  if (failed === file) {
    return <p className="m-auto text-sm text-muted-foreground">{t("library.preview.cannotPreview")}</p>;
  }
  return (
    <div ref={setContainer} className="size-full overflow-auto bg-muted/60">
      <Document
        file={file}
        options={PDF_OPTIONS}
        onLoadSuccess={(document) => {
          setPdf(document);
          setPages(document.numPages);
          void document.getPage(1).then((page) => {
            const viewport = page.getViewport({ scale: 1 });
            setAspect(clampAspect(viewport.height / viewport.width));
          });
        }}
        onLoadError={() => setFailed(file)}
        loading={<Spinner className="mx-auto mt-24 size-6" />}
      >
        {available > 0 && pdf && (
          <PdfPages
            key={`${width}:${aspect}`}
            pdf={pdf}
            pages={firstPageOnly ? Math.min(pages, 1) : Math.min(pages, MAX_PDF_PAGES)}
            width={width}
            aspect={aspect}
            scrollElement={container}
          />
        )}
        {available > 0 && pdf && !firstPageOnly && pages > MAX_PDF_PAGES && (
          <p className="pb-6 text-center text-ui-12 text-muted-foreground">{t("library.preview.documentTruncated")}</p>
        )}
      </Document>
    </div>
  );
}
