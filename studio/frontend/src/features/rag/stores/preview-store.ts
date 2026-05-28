// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import {
  type PreviewMediaKind,
  type PreviewTarget,
  fetchPreviewFileBlob,
  fetchPreviewFileUrl,
  fetchPreviewTarget,
} from "../api/rag-api";

/** mediaKinds that may safely back an inline object URL (e.g. PDF.js
 *  worker, plain text, raster image). HTML, DOCX, and unknown are
 *  forced through the extracted-text fallback per contracts §5.4 +
 *  Risk #3 (unsafe HTML inline rendering). */
const INLINE_BLOB_ALLOWLIST: ReadonlySet<PreviewMediaKind> = new Set([
  "pdf",
  "text",
  "image",
]);

export function isInlineBlobAllowed(mediaKind: PreviewMediaKind): boolean {
  return INLINE_BLOB_ALLOWLIST.has(mediaKind);
}

/** What the panel should mount for the current target. Computed from
 *  `target.mediaKind` so the panel never has to re-derive it. */
export type PreviewLoadStatus = "idle" | "loading" | "ready" | "error";

export interface PreviewRequest {
  /** Durable `rag_documents.id`. The only field required to open. */
  documentId: string;
  /** Durable `rag_chunks.id`. Optional — absence means document-row
   *  preview (contracts §1.3 + decision Q2: snippet/targetPage stay
   *  null, no first-chunk fallback). */
  backendChunkId?: string | null;
}

interface PreviewState {
  /** Currently-open preview, or null when closed. */
  target: PreviewTarget | null;
  /** Object URL for the original file blob (PDF / text / image only).
   *  Null for docx / html / unknown (extracted-text fallback) and
   *  while the fetch is still in flight. */
  previewBlobUrl: string | null;
  /** Original fetched file blob for text/image fallback previews. PDFs
   *  prefer `previewFileUrl` so PDF.js can issue range requests. */
  previewBlob: Blob | null;
  /** Short-lived signed URL for PDF.js range requests. */
  previewFileUrl: string | null;
  previewFileUrlExpiresAt: number | null;
  /** `target`-fetch + `blob`-fetch combined status. */
  status: PreviewLoadStatus;
  /** Last error message, if `status === "error"`. */
  error: string | null;
  /** Open key uniquely identifying the current request — used by tests
   *  and by consumers that need to react to "the open call changed
   *  underneath me" (e.g. re-fetch after stale closure). */
  openKey: number;

  /** Open or replace the current preview. If a previous preview is
   *  open, its object URL is revoked and its in-flight fetch is
   *  aborted before the new request begins. */
  open: (req: PreviewRequest) => Promise<void>;
  /** Close the current preview. Revokes the object URL and aborts any
   *  in-flight fetch. Safe to call when nothing is open. */
  close: () => void;
}

// State that can't live inside the zustand object without being
// part of the React render cycle. Kept module-scoped because the
// preview store is a singleton.
let activeAbortController: AbortController | null = null;
let activeBlobUrl: string | null = null;
let activeOpenKey = 0;
let restoreFocusElement: HTMLElement | null = null;

function revokeActiveBlobUrl(): void {
  if (activeBlobUrl) {
    URL.revokeObjectURL(activeBlobUrl);
    activeBlobUrl = null;
  }
}

function abortActive(): void {
  if (activeAbortController) {
    activeAbortController.abort();
    activeAbortController = null;
  }
}

export const usePreviewStore = create<PreviewState>((set) => ({
  target: null,
  previewBlobUrl: null,
  previewBlob: null,
  previewFileUrl: null,
  previewFileUrlExpiresAt: null,
  status: "idle",
  error: null,
  openKey: 0,

  async open(req) {
    // Single-slot invariant (contracts §5.1): tear down whatever was
    // there before assigning the new target. revoke → abort → reset.
    revokeActiveBlobUrl();
    abortActive();

    activeOpenKey += 1;
    const myKey = activeOpenKey;
    const controller = new AbortController();
    activeAbortController = controller;
    const activeElement = document.activeElement;
    restoreFocusElement =
      activeElement instanceof HTMLElement ? activeElement : null;

    set({
      target: null,
      previewBlobUrl: null,
      previewBlob: null,
      previewFileUrl: null,
      previewFileUrlExpiresAt: null,
      status: "loading",
      error: null,
      openKey: myKey,
    });

    let target: PreviewTarget;
    try {
      target = await fetchPreviewTarget(
        req.documentId,
        req.backendChunkId ?? null,
      );
    } catch (err) {
      if (myKey !== activeOpenKey) return; // superseded
      if (activeAbortController === controller) activeAbortController = null;
      set({
        status: "error",
        error: err instanceof Error ? err.message : String(err),
        openKey: myKey,
      });
      return;
    }

    if (myKey !== activeOpenKey) {
      // The user opened a different document while we were waiting.
      return;
    }

    // DOCX: fetch the raw bytes so the panel can render a DOMPurify-
    // sanitized inline preview (docx-preview). We deliberately keep
    // previewBlobUrl = null — the bytes flow through `previewBlob` to the
    // renderer only; no object URL is created, so the "open original
    // inline" path stays disabled (Risk #3) and Download remains the only
    // way to get the raw file.
    if (target.mediaKind === "docx") {
      let docxBlob: Blob;
      try {
        docxBlob = await fetchPreviewFileBlob(req.documentId, controller.signal);
      } catch (err) {
        if (controller.signal.aborted || myKey !== activeOpenKey) return;
        if (activeAbortController === controller) activeAbortController = null;
        set({
          target,
          previewBlobUrl: null,
          previewBlob: null,
          previewFileUrl: null,
          previewFileUrlExpiresAt: null,
          status: "error",
          error: err instanceof Error ? err.message : String(err),
          openKey: myKey,
        });
        return;
      }
      if (myKey !== activeOpenKey) return;
      if (activeAbortController === controller) activeAbortController = null;
      set({
        target,
        previewBlobUrl: null,
        previewBlob: docxBlob,
        previewFileUrl: null,
        previewFileUrlExpiresAt: null,
        status: "ready",
        error: null,
        openKey: myKey,
      });
      return;
    }

    // For mediaKinds outside the allowlist (html / unknown), skip the
    // blob fetch entirely — the panel mounts the extracted-text fallback
    // (contracts §5.4 + Risk #3).
    if (!isInlineBlobAllowed(target.mediaKind)) {
      if (activeAbortController === controller) activeAbortController = null;
      set({
        target,
        previewBlobUrl: null,
        previewBlob: null,
        previewFileUrl: null,
        previewFileUrlExpiresAt: null,
        status: "ready",
        error: null,
        openKey: myKey,
      });
      return;
    }

    if (target.mediaKind === "pdf") {
      try {
        const previewFile = await fetchPreviewFileUrl(
          req.documentId,
          controller.signal,
        );
        if (myKey !== activeOpenKey) return;
        if (activeAbortController === controller) activeAbortController = null;
        set({
          target,
          previewBlobUrl: null,
          previewBlob: null,
          previewFileUrl: previewFile.url,
          previewFileUrlExpiresAt: previewFile.expiresAt,
          status: "ready",
          error: null,
          openKey: myKey,
        });
      } catch (err) {
        if (controller.signal.aborted || myKey !== activeOpenKey) return;
        if (activeAbortController === controller) activeAbortController = null;
        set({
          target,
          previewBlobUrl: null,
          previewBlob: null,
          previewFileUrl: null,
          previewFileUrlExpiresAt: null,
          status: "error",
          error: err instanceof Error ? err.message : String(err),
          openKey: myKey,
        });
      }
      return;
    }

    let blob: Blob;
    try {
      blob = await fetchPreviewFileBlob(req.documentId, controller.signal);
    } catch (err) {
      if (controller.signal.aborted || myKey !== activeOpenKey) return;
      if (activeAbortController === controller) activeAbortController = null;
      set({
        target,
        previewBlobUrl: null,
        previewBlob: null,
        previewFileUrl: null,
        previewFileUrlExpiresAt: null,
        status: "error",
        error: err instanceof Error ? err.message : String(err),
        openKey: myKey,
      });
      return;
    }

    if (myKey !== activeOpenKey) {
      // Superseded between target fetch and blob fetch — drop the bytes.
      return;
    }

    const objectUrl = URL.createObjectURL(blob);
    activeBlobUrl = objectUrl;
    if (activeAbortController === controller) activeAbortController = null;
    set({
      target,
      previewBlobUrl: objectUrl,
      previewBlob: blob,
      previewFileUrl: null,
      previewFileUrlExpiresAt: null,
      status: "ready",
      error: null,
      openKey: myKey,
    });
  },

  close() {
    revokeActiveBlobUrl();
    abortActive();
    activeOpenKey += 1; // poison any in-flight fetch that lands after this
    const focusTarget = restoreFocusElement;
    restoreFocusElement = null;
    set({
      target: null,
      previewBlobUrl: null,
      previewBlob: null,
      previewFileUrl: null,
      previewFileUrlExpiresAt: null,
      status: "idle",
      error: null,
      openKey: activeOpenKey,
    });
    if (focusTarget?.isConnected) {
      focusTarget.focus();
    }
  },
}));

/** Test-only inspector: returns whether the module-scoped blob URL is
 *  still live. Used by `preview-store.test.ts` to assert
 *  URL.revokeObjectURL was paired with URL.createObjectURL. */
export function __previewStoreInternals(): {
  activeBlobUrl: string | null;
  hasInflightController: boolean;
} {
  return {
    activeBlobUrl,
    hasInflightController: activeAbortController !== null,
  };
}
