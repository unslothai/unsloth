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

  // A fresh chat has no thread id until the first message. To attach docs before
  // sending, materialize the id on demand; the first message reuses it (see
  // append() in runtime-provider), so uploaded docs stay with the conversation.
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

  // Resolve the thread id, materializing on first use. Ref-deduped so a
  // double-click can't start two threads.
  const initPromiseRef = useRef<Promise<string | null> | null>(null);
  const ensureThreadId = useCallback((): Promise<string | null> => {
    if (effectiveThreadId) return Promise.resolve(effectiveThreadId);
    if (initPromiseRef.current) return initPromiseRef.current;
    const pending = aui
      .threadListItem()
      .initialize()
      .then(({ remoteId }) => {
        setMaterializedId(remoteId);
        setActiveThreadId(remoteId);
        return remoteId;
      })
      .catch(() => {
        toast.error("Couldn't start a chat for these documents");
        return null;
      })
      .finally(() => {
        initPromiseRef.current = null;
      });
    initPromiseRef.current = pending;
    return pending;
  }, [aui, effectiveThreadId, setActiveThreadId]);

  // Open the picker only once the thread id is ready, so the upload scope is
  // live when onChange fires; otherwise a fast first selection uploads into a
  // null scope and silently attaches nothing.
  const handleAddDocs = useCallback(async () => {
    const id = await ensureThreadId();
    if (id) fileInputRef.current?.click();
  }, [ensureThreadId]);

  // Only for the thread-document source; KB sources are managed in the KB dialog.
  if (!ragEnabled || ragSource.type !== "thread") return null;

  return (
    <div className="mb-2 flex w-full flex-row items-start gap-1.5 px-1.5 pt-0.5 pb-1">
      <button
        type="button"
        onClick={() => void handleAddDocs()}
        disabled={uploading}
        className="composer-pill-btn shrink-0"
        aria-label="Attach documents to this thread"
        title="Attach documents for retrieval"
      >
        <PaperclipIcon className="size-3.5" />
        <span>Add Files</span>
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
      {/* Cap the chip area so a large set scrolls instead of overflowing the
          conversation. */}
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
