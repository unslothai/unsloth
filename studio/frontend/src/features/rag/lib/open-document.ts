// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { openFileInBrowser } from "@/features/browser";
import { toast } from "@/lib/toast";
import { getDocumentFileUrl } from "../api/rag-api";

/** Opens an indexed document's original file in Studio's browser, for formats the attachment viewer can't show. */
export function openRagDocumentInBrowser(
  documentId: string,
  filename: string,
): void {
  void getDocumentFileUrl(documentId)
    // Signed URL: no bearer, so no credentials either.
    .then((url) => fetch(url, { credentials: "omit" }))
    .then((response) => {
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      return response.blob();
    })
    .then((blob) =>
      openFileInBrowser({
        blob,
        name: filename,
        contentType: blob.type,
        key: `rag:${documentId}`,
      }),
    )
    .catch(() => toast.error(`Could not open ${filename}`));
}
