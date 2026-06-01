// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useCallback, useRef } from "react";
import { PaperclipIcon } from "lucide-react";
import { useChatRuntimeStore } from "@/features/chat/stores/chat-runtime-store";
import { listThreadDocuments } from "../api/rag-api";
import { RAG_UPLOAD_ACCEPT } from "../types/rag";
import { DocumentStatusChip } from "./document-status-chip";
import { useRagDocuments } from "./use-rag-documents";

/**
 * Per-thread document strip shown above the composer when the Docs pill is
 * on and the retrieval source is "this thread's documents". Uploads attach
 * to the current thread; chips track indexing status live via SSE.
 */
export function ThreadDocumentsBar({ threadId }: { threadId: string | null }) {
  const ragEnabled = useChatRuntimeStore((s) => s.ragEnabled);
  const ragSource = useChatRuntimeStore((s) => s.ragSource);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const lister = useCallback(
    () => (threadId ? listThreadDocuments(threadId) : Promise.resolve([])),
    [threadId],
  );
  const { documents, uploading, upload, remove } = useRagDocuments(
    threadId && ragEnabled && ragSource.type === "thread"
      ? { type: "thread", threadId }
      : null,
    lister,
  );

  // Only surface for the thread-document source; KB sources are managed in
  // the KB dialog, not per-thread.
  if (!ragEnabled || ragSource.type !== "thread") return null;

  return (
    <div className="mb-2 flex w-full flex-row flex-wrap items-center gap-1.5 px-1.5 pt-0.5 pb-1">
      <button
        type="button"
        onClick={() => fileInputRef.current?.click()}
        disabled={!threadId || uploading}
        className="composer-pill-btn"
        aria-label="Attach documents to this thread"
        title={
          threadId
            ? "Attach documents for retrieval"
            : "Send a message first to attach documents"
        }
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
      {documents.map((doc) => (
        <DocumentStatusChip
          key={doc.id}
          filename={doc.filename}
          status={doc.status}
          progress={doc.progress}
          error={doc.error}
          onRemove={
            doc.id.startsWith("pending_") ? undefined : () => void remove(doc.id)
          }
        />
      ))}
    </div>
  );
}
