// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import type { PreviewFigure, PreviewTarget } from "../types/rag";

// Global store for the shared preview Sheet, so any citation drives the one viewer
// without prop-drilling.
interface DocumentPreviewState {
  open: boolean;
  documentId: string | null;
  /** Chunk to highlight; null opens at page 1. */
  chunkId: string | null;
  filename: string | null;
  page: number | null;
  /** Inline-resolved target (no backend fetch); used by the markdown kind so
   * extracted documents render without a server-side documentId. */
  inlineTarget: PreviewTarget | null;
  openPreview: (args: {
    documentId: string;
    chunkId?: string | null;
    filename?: string | null;
    page?: number | null;
    /** When set with mediaKind "markdown", the sheet renders it directly. */
    mediaKind?: PreviewTarget["mediaKind"];
    markdown?: string | null;
    figures?: PreviewFigure[];
  }) => void;
  closePreview: () => void;
}

export const useDocumentPreviewStore = create<DocumentPreviewState>((set) => ({
  open: false,
  documentId: null,
  chunkId: null,
  filename: null,
  page: null,
  inlineTarget: null,
  openPreview: ({ documentId, chunkId, filename, page, mediaKind, markdown, figures }) =>
    set({
      open: true,
      documentId,
      chunkId: chunkId ?? null,
      filename: filename ?? null,
      page: page ?? null,
      inlineTarget:
        mediaKind === "markdown"
          ? {
              documentId,
              filename: filename ?? "Document",
              mediaKind: "markdown",
              pdfRegions: [],
              markdown: markdown ?? "",
              figures: figures ?? [],
            }
          : null,
    }),
  closePreview: () => set({ open: false, inlineTarget: null }),
}));
