// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useCallback, useEffect, useRef, useState } from "react";
import { PaperclipIcon } from "lucide-react";
import { useAui } from "@assistant-ui/react";
import { useChatRuntimeStore } from "@/features/chat/stores/chat-runtime-store";
import { toast } from "@/lib/toast";
import { listThreadDocuments } from "../api/rag-api";
import { RAG_UPLOAD_ACCEPT } from "../types/rag";
import { DocumentStatusChip } from "./document-status-chip";
import { useRagDocuments } from "./use-rag-documents";

/**
 * Per-thread document strip above the composer (Docs pill on, thread source).
 * Uploads attach to the thread; chips track indexing status live via SSE.
 */
export function ThreadDocumentsBar({ threadId }: { threadId: string | null }) {
  const ragEnabled = useChatRuntimeStore((s) => s.ragEnabled);
  const ragSource = useChatRuntimeStore((s) => s.ragSource);
  const setActiveThreadId = useChatRuntimeStore((s) => s.setActiveThreadId);
  const aui = useAui();
  const fileInputRef = useRef<HTMLInputElement>(null);

  // A fresh chat has no thread id until the first message. To let the user
  // attach documents before sending, materialize the thread id on demand and
  // keep it here; the first message reuses the same id (see append() in
  // runtime-provider), so uploaded docs stay with the conversation.
  const [materializedId, setMaterializedId] = useState<string | null>(null);
  const effectiveThreadId = threadId ?? materializedId;
  useEffect(() => {
    if (threadId) setMaterializedId(null);
  }, [threadId]);

  const lister = useCallback(
    () =>
      effectiveThreadId
        ? listThreadDocuments(effectiveThreadId)
        : Promise.resolve([]),
    [effectiveThreadId],
  );
  const { documents, uploading, upload, remove } = useRagDocuments(
    effectiveThreadId && ragEnabled && ragSource.type === "thread"
      ? { type: "thread", threadId: effectiveThreadId }
      : null,
    lister,
  );

  // Open the picker synchronously to preserve the user gesture; materialize the
  // thread id in parallel so the upload scope is live by the time onChange fires.
  const handleAddDocs = useCallback(() => {
    if (!effectiveThreadId) {
      void aui
        .threadListItem()
        .initialize()
        .then(({ remoteId }) => {
          setMaterializedId(remoteId);
          setActiveThreadId(remoteId);
        })
        .catch(() => toast.error("Couldn't start a chat for these documents"));
    }
    fileInputRef.current?.click();
  }, [aui, effectiveThreadId, setActiveThreadId]);

  // Only surface for the thread-document source; KB sources are managed in
  // the KB dialog, not per-thread.
  if (!ragEnabled || ragSource.type !== "thread") return null;

  return (
    <div className="mb-2 flex w-full flex-row items-start gap-1.5 px-1.5 pt-0.5 pb-1">
      <button
        type="button"
        onClick={handleAddDocs}
        disabled={uploading}
        className="composer-pill-btn shrink-0"
        aria-label="Attach documents to this thread"
        title="Attach documents for retrieval"
      >
        <PaperclipIcon className="size-3.5" />
        <span>Add docs</span>
      </button>
      <input
        ref={fileInputRef}
        type="file"
        multiple
        accept={RAG_UPLOAD_ACCEPT}
        className="hidden"
        onChange={(e) => {
          if (e.target.files?.length) void upload(e.target.files);
          e.target.value = "";
        }}
      />
      {/* Cap the chip area so a large document set scrolls instead of growing
          up and over the conversation. */}
      <div className="flex max-h-24 flex-1 flex-row flex-wrap items-center gap-1.5 overflow-y-auto">
        {documents.map((doc) => (
          <DocumentStatusChip
            key={doc.id}
            filename={doc.filename}
            status={doc.status}
            progress={doc.progress}
            error={doc.error}
            onRemove={
              doc.id.startsWith("pending_")
                ? undefined
                : () => void remove(doc.id)
            }
          />
        ))}
      </div>
    </div>
  );
}
