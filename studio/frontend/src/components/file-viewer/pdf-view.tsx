// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Spinner } from "@/components/ui/spinner";
import { useT } from "@/i18n";
import { useEffect, useRef, useState } from "react";
import { Document, Page, pdfjs } from "react-pdf";
import "react-pdf/dist/Page/TextLayer.css";
import { useWidth } from "./use-width";

pdfjs.GlobalWorkerOptions.workerSrc = new URL(
  "pdfjs-dist/build/pdf.worker.min.mjs",
  import.meta.url,
).toString();

const MAX_PAGE_WIDTH = 880;

/** A page renders once it comes near the viewport; until then it holds its space. */
function LazyPage({
  pageNumber,
  width,
  aspect,
  root,
}: {
  pageNumber: number;
  width: number;
  aspect: number;
  root: HTMLElement | null;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const [seen, setSeen] = useState(pageNumber === 1);
  useEffect(() => {
    const element = ref.current;
    if (!element || seen) return;
    const observer = new IntersectionObserver(
      (entries) => entries.some((entry) => entry.isIntersecting) && setSeen(true),
      { root, rootMargin: "1200px 0px" },
    );
    observer.observe(element);
    return () => observer.disconnect();
  }, [root, seen]);
  return (
    <div
      ref={ref}
      className="relative mx-auto bg-white shadow-sm ring-1 ring-black/5"
      style={{ width, minHeight: seen ? undefined : width * aspect }}
    >
      {seen && (
        <Page
          pageNumber={pageNumber}
          width={width}
          renderAnnotationLayer={false}
          loading={<div style={{ height: width * aspect }} />}
        />
      )}
    </div>
  );
}

export default function PdfView({ file, scale }: { file: Blob; scale: number }) {
  const t = useT();
  const [container, setContainer] = useState<HTMLDivElement | null>(null);
  const available = useWidth(container);
  const [pages, setPages] = useState(0);
  const [aspect, setAspect] = useState(1.294);
  const [error, setError] = useState<string | null>(null);
  const width = Math.max(200, Math.min(available, MAX_PAGE_WIDTH)) * scale;

  if (error) {
    return <p className="m-auto text-sm text-muted-foreground">{t("library.preview.cannotPreview")}</p>;
  }
  return (
    <div ref={setContainer} className="size-full overflow-auto bg-muted/60 py-6">
      <Document
        file={file}
        onLoadSuccess={(pdf) => {
          setPages(pdf.numPages);
          void pdf.getPage(1).then((page) => {
            const viewport = page.getViewport({ scale: 1 });
            setAspect(viewport.height / viewport.width);
          });
        }}
        onLoadError={(err) => setError(err.message)}
        loading={<Spinner className="mx-auto mt-24 size-6" />}
        className="flex min-w-fit flex-col gap-4"
      >
        {available > 0 &&
          Array.from({ length: pages }, (_, index) => (
            <LazyPage
              key={index}
              pageNumber={index + 1}
              width={width}
              aspect={aspect}
              root={container}
            />
          ))}
      </Document>
    </div>
  );
}
