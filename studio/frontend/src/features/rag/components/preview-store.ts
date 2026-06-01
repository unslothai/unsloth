// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";

/**
 * Global store for the shared document preview Sheet: a citation badge calls
 * `openPreview` with its document + chunk, and the Sheet resolves and renders
 * it. Global so any citation can open the one viewer without prop-drilling.
 */
interface DocumentPreviewState {
  open: boolean;
  documentId: string | null;
  /** Chunk to highlight; null opens the document at its first page. */
  chunkId: string | null;
  /** Filename for the header, shown before the target resolves. */
  filename: string | null;
  /** Page hint for the header, shown before the target resolves. */
  page: number | null;
  openPreview: (args: {
    documentId: string;
    chunkId?: string | null;
    filename?: string | null;
    page?: number | null;
  }) => void;
  closePreview: () => void;
}

export const useDocumentPreviewStore = create<DocumentPreviewState>((set) => ({
  open: false,
  documentId: null,
  chunkId: null,
  filename: null,
  page: null,
  openPreview: ({ documentId, chunkId, filename, page }) =>
    set({
      open: true,
      documentId,
      chunkId: chunkId ?? null,
      filename: filename ?? null,
      page: page ?? null,
    }),
  closePreview: () => set({ open: false }),
}));
