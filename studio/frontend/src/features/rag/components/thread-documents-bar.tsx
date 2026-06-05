// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useCallback, useEffect, useRef, useState } from "react";
import { LibraryBigIcon, PaperclipIcon } from "lucide-react";
import { useAui } from "@assistant-ui/react";
import { useChatRuntimeStore } from "@/features/chat/stores/chat-runtime-store";
import { useRagToolAvailable } from "@/features/chat/hooks/use-rag-tool-available";
import { toast } from "@/lib/toast";
import { listKnowledgeBases, listThreadDocuments } from "../api/rag-api";
import { RAG_UPLOAD_ACCEPT } from "../types/rag";
import { DocumentStatusChip } from "./document-status-chip";
import { useRagDocuments } from "./use-rag-documents";

/**
 * Read-only chip shown above the composer when retrieval comes from a knowledge base
 * instead of this thread's uploads, so a KB source isn't invisible.
 */
function KnowledgeBaseSourceChip({ kbId }: { kbId: string }) {
  const [name, setName] = useState<string | null>(null);
  useEffect(() => {
    let cancelled = false;
    listKnowledgeBases()
      .then((rows) => {
        if (!cancelled) setName(rows.find((kb) => kb.id === kbId)?.name ?? null);
      })
      .catch(() => {
        if (!cancelled) setName(null);
      });
    return () => {
      cancelled = true;
    };
  }, [kbId]);
  return (
    <div className="mb-2 flex w-full flex-row items-center gap-1.5 px-1.5 pt-0.5 pb-1">
      <span
        className="composer-pill-btn shrink-0 cursor-default"
        title="This chat retrieves from a knowledge base. Change the source in RAG retrieval settings."
      >
        <LibraryBigIcon className="size-3.5" />
        <span>{name ? `Knowledge base: ${name}` : "Knowledge base"}</span>
      </span>
    </div>
  );
}

/** Per-thread document strip above the composer; chips track indexing status live via SSE. */
export function ThreadDocumentsBar({ threadId }: { threadId: string | null }) {
  const ragEnabled = useChatRuntimeStore((s) => s.ragEnabled);
  const ragAvailable = useRagToolAvailable();
  const ragSource = useChatRuntimeStore((s) => s.ragSource);
  const aui = useAui();
  const fileInputRef = useRef<HTMLInputElement>(null);

  // A fresh chat has no thread id until the first message. Materialize one on demand
  // so docs can attach before sending; append() in runtime-provider reuses it. Track
  // the id locally only: pushing it to global activeThreadId would, in a project, flip
  // ProjectLanding into its pendingNewThreadId branch and swap the composer for a fresh
  // <Thread>, remounting this bar mid-upload and dropping the just-attached chips.
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

  // Resolve the thread id, materializing on first use. Ref-deduped so a double-click
  // can't start two threads.
  const initPromiseRef = useRef<Promise<string | null> | null>(null);
  const ensureThreadId = useCallback((): Promise<string | null> => {
    if (effectiveThreadId) return Promise.resolve(effectiveThreadId);
    if (initPromiseRef.current) return initPromiseRef.current;
    const pending = aui
      .threadListItem()
      .initialize()
      .then(({ remoteId }) => {
        setMaterializedId(remoteId);
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
  }, [aui, effectiveThreadId]);

  // Just open the picker synchronously so the click's user activation survives. Do
  // NOT materialize the thread here: that fires setActiveThreadId while the native
  // dialog sits open, which can remount the composer and orphan this <input> so the
  // file selection lands on a detached node. Materialize in onChange instead.
  const handleAddDocs = useCallback(() => {
    fileInputRef.current?.click();
  }, []);

  // Show only when the RAG pill is effectively on: enabled AND a local tool-capable
  // model is loaded (matches the pill's disabled gate).
  if (!ragEnabled || !ragAvailable) return null;
  // A KB source uploads via the KB dialog, not here; show which KB is active.
  if (ragSource.type === "kb") {
    return <KnowledgeBaseSourceChip kbId={ragSource.kbId} />;
  }

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
        <span>Add Files</span>
      </button>
      <input
        ref={fileInputRef}
        type="file"
        multiple
        accept={RAG_UPLOAD_ACCEPT}
        className="hidden"
        onChange={(e) => {
          const files = Array.from(e.target.files ?? []);
          e.target.value = "";
          if (files.length === 0) return;
          // Pass the thread id as a promise so upload() flips its in-flight guard
          // synchronously, before materialization re-renders us. On the first click
          // the hook's `scope` is still null, so we upload to the resolved id.
          void upload(
            files,
            ensureThreadId().then((id) =>
              id ? ({ type: "thread", threadId: id } as const) : null,
            ),
          );
        }}
      />
      {/* Cap the chip area so a large set scrolls. */}
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
