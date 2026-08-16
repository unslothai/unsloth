


import { create } from "zustand";

// Global store for the shared preview Sheet, so any citation drives the one viewer
// without prop-drilling.
interface DocumentPreviewState {
  open: boolean;
  documentId: string | null;
  /** Chunk to highlight; null opens at page 1. */
  chunkId: string | null;
  filename: string | null;
  page: number | null;
  source: "platform" | "local" | null;
  openPreview: (args: {
    documentId: string;
    chunkId?: string | null;
    filename?: string | null;
    page?: number | null;
    source?: "platform" | "local" | null;
  }) => void;
  closePreview: () => void;
}

export const useDocumentPreviewStore = create<DocumentPreviewState>((set) => ({
  open: false,
  documentId: null,
  chunkId: null,
  filename: null,
  page: null,
  source: null,
  openPreview: ({ documentId, chunkId, filename, page, source }) =>
    set({
      open: true,
      documentId,
      chunkId: chunkId ?? null,
      filename: filename ?? null,
      page: page ?? null,
      source: source ?? null,
    }),
  closePreview: () => set({ open: false }),
}));
