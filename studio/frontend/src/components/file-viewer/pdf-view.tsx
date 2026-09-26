// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Spinner } from "@/components/ui/spinner";
import { useT } from "@/i18n";
import { useVirtualizer } from "@tanstack/react-virtual";
import { useState } from "react";
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

/** Only the pages near the viewport are mounted, so a document of many thousands opens as fast as a short one. */
function PdfPages({
  pages,
  width,
  aspect,
  scrollElement,
}: {
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
            <Page
              pageNumber={item.index + 1}
              width={width}
              renderAnnotationLayer={false}
              loading={<div style={{ height: width * aspect }} />}
            />
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
        onLoadSuccess={(pdf) => {
          setPages(pdf.numPages);
          void pdf.getPage(1).then((page) => {
            const viewport = page.getViewport({ scale: 1 });
            setAspect(viewport.height / viewport.width);
          });
        }}
        onLoadError={() => setFailed(file)}
        loading={<Spinner className="mx-auto mt-24 size-6" />}
      >
        {/* Keyed on the size, so the virtualizer measures afresh. */}
        {available > 0 && (
          <PdfPages key={`${width}:${aspect}`} pages={pages} width={width} aspect={aspect} scrollElement={container} />
        )}
      </Document>
    </div>
  );
}
