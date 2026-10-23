// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { DocumentStatusChip } from "./document-status-chip";
import { RagDocumentTrigger } from "./rag-document-trigger";
import type { TrackedDocument } from "./use-rag-documents";

/** A Sources tab chip whose name opens the file, as the chat with files cards do. */
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
        ready
          ? (trigger) => (
              <RagDocumentTrigger documentId={doc.id} filename={doc.filename}>
                {trigger}
              </RagDocumentTrigger>
            )
          : undefined
      }
    />
  );
}
