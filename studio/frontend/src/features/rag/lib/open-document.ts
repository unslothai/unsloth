// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { MAX_DOCUMENT_PREVIEW_BYTES } from "@/components/file-viewer";
import { openFileInBrowser } from "@/features/browser";
import { isTauri } from "@/lib/api-base";
import { toast } from "@/lib/toast";
import { getDocumentFileUrl } from "../api/rag-api";

/**
 * Opens an indexed document's file outside Studio: the OS opener on Desktop (its webview has no
 * window.open), else a new tab opened inside the click, since popup blockers refuse one opened
 * after the URL request.
 */
export function openRagDocumentExternally(
  documentId: string,
  filename: string,
): void {
  const failed = () => toast.error(`Could not open ${filename}`);
  if (isTauri) {
    getDocumentFileUrl(documentId)
      .then(async (url) => {
        const { openUrl } = await import("@tauri-apps/plugin-opener");
        await openUrl(url);
      })
      .catch(failed);
    return;
  }
  const tab = window.open("about:blank", "_blank");
  if (!tab) {
    toast.error(`Your browser blocked the new tab for ${filename}`, {
      description: "Allow popups for Studio and try again.",
    });
    return;
  }
  tab.opener = null;
  getDocumentFileUrl(documentId)
    .then((url) => tab.location.replace(url))
    .catch(() => {
      tab.close();
      failed();
    });
}

/** For a file Studio can't preview: a toast whose action opens it outside Studio. */
export function offerRagDocumentOutside(
  documentId: string,
  filename: string,
): void {
  toast.message(`${filename} is too large to preview in Studio`, {
    action: {
      label: "Open",
      onClick: () => openRagDocumentExternally(documentId, filename),
    },
  });
}

// Per document, only the latest click opens: downloads can finish out of order.
const latestOpen = new Map<string, number>();
let opens = 0;

/** Opens an indexed document's file in Studio's browser, a PDF at `page`; past the preview limit,
 *  offers it outside. */
export function openRagDocumentInBrowser(
  documentId: string,
  filename: string,
  page?: number,
): void {
  const ticket = ++opens;
  latestOpen.set(documentId, ticket);
  getDocumentFileUrl(documentId)
    // Signed URL: no bearer, so no credentials either.
    .then((url) => fetch(url, { credentials: "omit" }))
    .then((response) => {
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      // The file view refuses these anyway: don't buffer up to the upload limit to say so.
      if (
        Number(response.headers.get("content-length")) >
        MAX_DOCUMENT_PREVIEW_BYTES
      ) {
        void response.body?.cancel();
        return null;
      }
      return response.blob();
    })
    .then((blob) => {
      if (latestOpen.get(documentId) !== ticket) {
        return;
      }
      if (blob === null || blob.size > MAX_DOCUMENT_PREVIEW_BYTES) {
        offerRagDocumentOutside(documentId, filename);
        return;
      }
      openFileInBrowser({
        blob,
        name: filename,
        contentType: blob.type,
        key: `rag:${documentId}`,
        page,
      });
    })
    .catch(() => {
      if (latestOpen.get(documentId) === ticket) {
        toast.error(`Could not open ${filename}`);
      }
    });
}
