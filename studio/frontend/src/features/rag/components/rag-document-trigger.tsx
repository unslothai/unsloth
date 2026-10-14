// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { LocalFileDialog } from "@/components/assistant-ui/attachment-document-dialog";
import { MAX_DOCUMENT_PREVIEW_BYTES } from "@/components/file-viewer";
import { useBrowserPanelAvailable } from "@/features/browser/panel-availability";
import { toast } from "@/lib/toast";
import { type ReactElement, cloneElement } from "react";
import { getDocumentFileUrl } from "../api/rag-api";
import { openRagDocumentInBrowser } from "../lib/open-document";
import { asViewerFile, fetchSourceFile, isViewableSource } from "./source-file";

/**
 * Opens an indexed document from `children` (a button): in Studio's browser, which renders every
 * indexed format. Without the browser panel (mobile), viewable files use the attachment viewer and
 * the rest a new tab.
 */
export function RagDocumentTrigger({
  documentId,
  filename,
  children,
}: {
  documentId: string;
  filename: string;
  children: ReactElement;
}) {
  const browser = useBrowserPanelAvailable();
  if (browser) {
    return cloneElement(children as ReactElement<{ onClick?: () => void }>, {
      onClick: () => openRagDocumentInBrowser(documentId, filename),
    });
  }
  if (isViewableSource(filename)) {
    const load = async (): Promise<Blob> => {
      const file = await fetchSourceFile(
        { id: documentId, filename },
        MAX_DOCUMENT_PREVIEW_BYTES,
      );
      if (!file) {
        throw new Error(`Couldn't preview ${filename}`);
      }
      return asViewerFile(file);
    };
    return (
      <LocalFileDialog name={filename.split("/").pop() ?? filename} load={load}>
        {children}
      </LocalFileDialog>
    );
  }
  return cloneElement(children as ReactElement<{ onClick?: () => void }>, {
    onClick: () => {
      // Opened inside the click: popup blockers refuse a tab opened after the URL request.
      const tab = window.open("about:blank", "_blank");
      if (tab) {
        tab.opener = null;
      }
      getDocumentFileUrl(documentId)
        .then((url) => {
          if (tab) {
            tab.location.href = url;
          }
        })
        .catch(() => {
          tab?.close();
          toast.error(`Could not open ${filename}`);
        });
    },
  });
}
