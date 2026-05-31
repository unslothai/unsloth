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

/** mediaKinds that may safely back an inline object URL (PDF.js worker,
 *  plain text, raster image). HTML/DOCX/unknown go through the
 *  extracted-text fallback per contracts §5.4 + Risk #3 (unsafe HTML
 *  inline rendering). */
const INLINE_BLOB_ALLOWLIST: ReadonlySet<PreviewMediaKind> = new Set([
  "pdf",
  "text",
  "image",
]);

export function isInlineBlobAllowed(mediaKind: PreviewMediaKind): boolean {
  return INLINE_BLOB_ALLOWLIST.has(mediaKind);
}

/** What the panel mounts for the current target, derived from
 *  `target.mediaKind` so the panel never re-derives it. */
export type PreviewLoadStatus = "idle" | "loading" | "ready" | "error";

export interface PreviewRequest {
  /** Durable `rag_documents.id`. The only field required to open. */
  documentId: string;
  /** Durable `rag_chunks.id`. Optional; absence means document-row
   *  preview (contracts §1.3 + Q2: snippet/targetPage stay null, no
   *  first-chunk fallback). */
  backendChunkId?: string | null;
}

interface PreviewState {
  /** Currently-open preview, or null when closed. */
  target: PreviewTarget | null;
  /** Object URL for the original file blob (PDF/text/image only). Null
   *  for docx/html/unknown (extracted-text fallback) and while the
   *  fetch is in flight. */
  previewBlobUrl: string | null;
  /** Fetched file blob for text/image fallback previews. PDFs prefer
   *  `previewFileUrl` so PDF.js can issue range requests. */
  previewBlob: Blob | null;
  /** Short-lived signed URL for PDF.js range requests. */
  previewFileUrl: string | null;
  previewFileUrlExpiresAt: number | null;
  /** `target`-fetch + `blob`-fetch combined status. */
  status: PreviewLoadStatus;
  /** Last error message, if `status === "error"`. */
  error: string | null;
  /** Key uniquely identifying the current request — used by tests and
   *  consumers reacting to "the open call changed underneath me" (e.g.
   *  re-fetch after stale closure). */
  openKey: number;

  /** Open or replace the current preview. A previously-open preview's
   *  object URL is revoked and its in-flight fetch aborted before the
   *  new request begins. */
  open: (req: PreviewRequest) => Promise<void>;
  /** Close the current preview: revoke the object URL and abort any
   *  in-flight fetch. Safe to call when nothing is open. */
  close: () => void;
}

// State kept out of the zustand object to stay off the React render
// cycle. Module-scoped because the preview store is a singleton.
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
    // Single-slot invariant (contracts §5.1): tear down the previous
    // target before assigning the new one. revoke → abort → reset.
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

    // For mediaKinds outside the allowlist (docx/html/unknown), skip the
    // blob fetch — the panel mounts the extracted-text fallback
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

/** Test-only inspector: whether the module-scoped blob URL is still
 *  live. Used by `preview-store.test.ts` to assert revokeObjectURL was
 *  paired with createObjectURL. */
export function __previewStoreInternals(): {
  activeBlobUrl: string | null;
  hasInflightController: boolean;
} {
  return {
    activeBlobUrl,
    hasInflightController: activeAbortController !== null,
  };
}
