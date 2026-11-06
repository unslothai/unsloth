// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { LocalFileDialog } from "@/components/assistant-ui/attachment-document-dialog";
import { MAX_DOCUMENT_PREVIEW_BYTES } from "@/components/file-viewer";
import { canShowFile } from "@/features/browser/file-kind";
import { useBrowserPanelAvailable } from "@/features/browser/panel-availability";
import { useBrowserPrefsStore } from "@/features/browser/prefs-store";
import { type ReactElement, cloneElement } from "react";
import {
  offerRagDocumentOutside,
  openRagDocumentExternally,
  openRagDocumentInBrowser,
} from "../lib/open-document";
import { asViewerFile, fetchSourceFile, isViewableSource } from "./source-file";

/**
 * Opens an indexed document from `children` (a button): in Studio's browser when it can show the
 * format and files open there. Otherwise viewable files use the attachment viewer, the rest a new tab.
 * A PDF opens at `page` (1-based) in either viewer.
 */
export function RagDocumentTrigger({
  documentId,
  filename,
  page,
  children,
}: {
  documentId: string;
  filename: string;
  page?: number;
  children: ReactElement;
}) {
  const panel = useBrowserPanelAvailable();
  const filesInBrowser = useBrowserPrefsStore((s) => s.openFilesInBrowser);
  // A non-empty type, so only the name decides: canShowFile accepts any file with no type.
  const browser =
    panel &&
    filesInBrowser &&
    canShowFile(filename, "application/octet-stream");
  if (browser) {
    return cloneElement(children as ReactElement<{ onClick?: () => void }>, {
      onClick: () => openRagDocumentInBrowser(documentId, filename, page),
    });
  }
  if (isViewableSource(filename)) {
    const load = async (): Promise<Blob> => {
      const file = await fetchSourceFile(
        { id: documentId, filename },
        MAX_DOCUMENT_PREVIEW_BYTES,
      );
      if (!file) {
        // Past the viewer's limit (uploads go up to 200 MB): still let it open outside Studio.
        offerRagDocumentOutside(documentId, filename);
        throw new Error(`Couldn't preview ${filename}`);
      }
      return asViewerFile(file);
    };
    return (
      <LocalFileDialog
        name={filename.split("/").pop() ?? filename}
        load={load}
        page={page}
      >
        {children}
      </LocalFileDialog>
    );
  }
  return cloneElement(children as ReactElement<{ onClick?: () => void }>, {
    onClick: () => openRagDocumentExternally(documentId, filename),
  });
}
