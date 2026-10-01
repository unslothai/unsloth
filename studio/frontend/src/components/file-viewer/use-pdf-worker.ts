// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect, useState } from "react";
import { pdfjs } from "react-pdf";

export type PdfWorker = {
  worker: InstanceType<typeof pdfjs.PDFWorker>;
  loaded: Set<{ destroy(): Promise<void> }>;
};

/**
 * One worker per viewer. unpdf sets globalThis.pdfjsWorker to its own PDF.js build, which PDF.js
 * then adopts and fails on (version mismatch). Passing a worker skips that lookup.
 */
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
    // External resource owned by this effect.
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
