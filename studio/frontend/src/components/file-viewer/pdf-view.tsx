// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Spinner } from "@/components/ui/spinner";
import { useT } from "@/i18n";
import { useVirtualizer } from "@tanstack/react-virtual";
import { useEffect, useState } from "react";
import { Document, Page, pdfjs } from "react-pdf";
import "react-pdf/dist/Page/TextLayer.css";
import { useWidth } from "./use-width";

pdfjs.GlobalWorkerOptions.workerSrc = new URL(
  "pdfjs-dist/build/pdf.worker.min.mjs",
  import.meta.url,
).toString();

const MAX_PAGE_WIDTH = 880;
const PAGE_GAP = 16;
const EDGE = 24;
// A page's canvas stays within these, drawn at a lower resolution past them: an extreme MediaBox
// would otherwise ask for more pixels than a browser holds.
const MAX_CANVAS_PIXELS = 32 * 1024 * 1024;
const MAX_CANVAS_SIDE = 16384;
// Below this resolution a page is too blurred to read, and shows as unpreviewable.
const MIN_PIXEL_RATIO = 0.1;
// Layout keeps even an extreme first page to a sane height until each page is measured.
const clampAspect = (aspect: number) => Math.min(Math.max(aspect, 0.05), 20);

type PdfDocument = {
  getPage(page: number): Promise<{ getViewport(options: { scale: number }): { width: number; height: number } }>;
};

/** One page, at its own shape and a resolution its canvas can hold. */
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
  const pageAspect = own?.index === index ? own.aspect : null;
  if (pageAspect === null) return <div style={{ height: width * aspect }} />;
  const height = width * pageAspect;
  const ratio = Math.min(
    window.devicePixelRatio || 1,
    Math.sqrt(MAX_CANVAS_PIXELS / (width * height)),
    MAX_CANVAS_SIDE / width,
    MAX_CANVAS_SIDE / height,
  );
  if (!(ratio >= MIN_PIXEL_RATIO)) {
    return (
      <p className="flex items-center justify-center text-sm text-muted-foreground" style={{ height: width * clampAspect(pageAspect) }}>
        {t("library.preview.cannotPreview")}
      </p>
    );
  }
  return (
    <Page
      pageNumber={index + 1}
      width={width}
      devicePixelRatio={ratio}
      renderAnnotationLayer={false}
      loading={<div style={{ height }} />}
    />
  );
}

/** Only the pages near the viewport are mounted, so a document of many thousands opens as fast as a short one. */
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

export default function PdfView({ file, scale }: { file: Blob; scale: number }) {
  const t = useT();
  const [container, setContainer] = useState<HTMLDivElement | null>(null);
  const available = useWidth(container);
  const [pdf, setPdf] = useState<PdfDocument | null>(null);
  const [pages, setPages] = useState(0);
  const [aspect, setAspect] = useState(1.294);
  // The file that failed, so a different one passed in later gets its own try.
  const [failed, setFailed] = useState<Blob | null>(null);
  const width = Math.max(200, Math.min(available, MAX_PAGE_WIDTH)) * scale;

  if (failed === file) {
    return <p className="m-auto text-sm text-muted-foreground">{t("library.preview.cannotPreview")}</p>;
  }
  return (
    <div ref={setContainer} className="size-full overflow-auto bg-muted/60">
      <Document
        file={file}
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
        {/* Keyed on the size, so the virtualizer measures afresh. */}
        {available > 0 && pdf && (
          <PdfPages
            key={`${width}:${aspect}`}
            pdf={pdf}
            pages={pages}
            width={width}
            aspect={aspect}
            scrollElement={container}
          />
        )}
      </Document>
    </div>
  );
}
