// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { LocalFileDialog } from "@/components/assistant-ui/attachment-document-dialog";
import { MAX_DOCUMENT_PREVIEW_BYTES } from "@/components/file-viewer";
import { DocumentStatusChip } from "./document-status-chip";
import { asViewerFile, fetchSourceFile, isViewableSource } from "./source-file";
import type { TrackedDocument } from "./use-rag-documents";

/** A Sources tab chip whose name opens the file in the attachment viewer, as the chat with files cards do. */
export function ProjectSourceChip({
  doc,
  onRemove,
  onRetry,
}: {
  doc: TrackedDocument;
  onRemove?: () => void;
  onRetry?: () => void;
}) {
  const ready = doc.status === "completed" && !doc.id.startsWith("pending_");
  const viewable = isViewableSource(doc.filename);
  const name = doc.filename.split("/").pop() ?? doc.filename;
  // Read on open; the dialog keeps the latest one in a ref, so a new identity per render is fine.
  const load = async (): Promise<Blob> => {
    const file = await fetchSourceFile(doc, MAX_DOCUMENT_PREVIEW_BYTES);
    if (!file) {
      throw new Error(`Couldn't preview ${doc.filename}`);
    }
    return asViewerFile(file);
  };
  return (
    <DocumentStatusChip
      filename={doc.filename}
      status={doc.status}
      progress={doc.progress}
      stage={doc.stage}
      error={doc.error}
      onRemove={onRemove}
      onRetry={onRetry}
      openTrigger={
        ready && viewable
          ? (trigger) => (
              <LocalFileDialog name={name} load={load}>
                {trigger}
              </LocalFileDialog>
            )
          : undefined
      }
    />
  );
}
