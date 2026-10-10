// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect, useState } from "react";
import { pdfjs } from "react-pdf";

export type PdfWorker = {
  worker: InstanceType<typeof pdfjs.PDFWorker>;
  loaded: Set<{ destroy(): Promise<void> }>;
};

/** Own worker per viewer: unpdf's globalThis.pdfjsWorker is another PDF.js version, which PDF.js would adopt. */
export function usePdfWorker(enabled: boolean): PdfWorker | null {
  const [state, setState] = useState<PdfWorker | null>(null);
  useEffect(() => {
    if (!enabled) return;
    const port = new Worker(pdfjs.GlobalWorkerOptions.workerSrc, {
      type: "module",
    });
    const next: PdfWorker = {
      worker: pdfjs.PDFWorker.create({ port }),
      loaded: new Set(),
    };
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setState(next);
    return () => {
      setState(null);
      // PDF.js frees a document's fonts only once the worker answers its Terminate.
      void Promise.allSettled(
        [...next.loaded].map((doc) => doc.destroy()),
      ).then(() => {
        next.worker.destroy();
        port.terminate();
      });
    };
  }, [enabled]);
  return state;
}
