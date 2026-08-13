// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useCallback, useEffect, useRef, useState } from "react";
import { HugeiconsIcon } from "@hugeicons/react";
import {
  AttachmentIcon,
  FileDatabaseIcon,
  Folder02Icon,
} from "@hugeicons/core-free-icons";
import { Tick02Icon } from "@/lib/tick-icon";
import { useAui } from "@assistant-ui/react";
import { cn } from "@/lib/utils";
import {
  type ProjectAttachmentTarget,
  useChatRuntimeStore,
} from "@/features/chat/stores/chat-runtime-store";
import {
  chatHistoryClearBoundary,
  ChatThreadDeletedError,
  ensureStoredChatThread,
  isThreadIncognito,
} from "@/features/chat";
import {
  useNativeAttachmentTargetKey,
  useNativeIntentStore,
} from "@/features/native-intents";
import { toast } from "@/lib/toast";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  invalidateProjectSources,
  listKnowledgeBases,
  listProjectDocuments,
  listThreadDocuments,
} from "../api/rag-api";
import { RAG_UPLOAD_ACCEPT, isLinkedFolderManaged } from "../types/rag";
import { DocumentStatusChip } from "./document-status-chip";
import { useRagDocuments } from "./use-rag-documents";

// Read-only chip shown when retrieval comes from a KB, so the source isn't invisible.
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
    <div className="mb-2 flex w-full flex-row items-center gap-1.5 pl-0.5 pr-1.5 pt-0.5 pb-1">
      <span
        className="composer-pill-btn shrink-0 cursor-default"
        title="This chat retrieves from a knowledge base. Change the source in RAG retrieval settings."
      >
        <HugeiconsIcon
          icon={FileDatabaseIcon}
          strokeWidth={2}
          className="size-3.5"
        />
        <span>{name ? `Knowledge base: ${name}` : "Knowledge base"}</span>
      </span>
    </div>
  );
}

/**
* Confirm a thread is stored before documents are indexed against it. An id reaches this
* component before its row write lands, from a cached initialize() or from activeThreadId, and
* upload_thread_document does not check the thread itself. A transport failure is not proof the
* row is missing, so only a definitive miss blocks the upload.
*/
async function requireStoredThread(threadId: string): Promise<void> {
  if (isThreadIncognito(threadId)) return;
  let stored: Awaited<ReturnType<typeof ensureStoredChatThread>>;
  try {
    stored = await ensureStoredChatThread(threadId);
  } catch (error) {
    // A backend tombstone is an answer, not an indeterminate transport failure: indexing
    // against it would leave documents under a thread that can never come back.
    if (error instanceof ChatThreadDeletedError) {
      throw error;
    }
    return;
  }
  if (!stored) {
    throw new Error(`Thread ${threadId} was not persisted`);
  }
}

/** The composer's attach control. Wording and glyph follow the active target, so
 * a project chat says up front where the file is going. */
function AttachFilesButton({
  disabled,
  compact,
  sharesWithProject,
  onClick,
}: {
  disabled: boolean;
  /** Icon-only once documents are attached, to leave the chips room. */
  compact: boolean;
  sharesWithProject: boolean;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      className={cn(
        "composer-pill-btn shrink-0 -translate-y-px !text-foreground/80",
        // Square button so the rounded-full hover reads as a circle.
        compact && "size-8 justify-center px-0",
      )}
      aria-label={
        sharesWithProject
          ? "Attach documents to this project"
          : "Attach documents to this thread"
      }
      title={
        sharesWithProject
          ? "Attach documents for retrieval, shared with every chat in this project"
          : "Attach documents for retrieval in this chat"
      }
    >
      <HugeiconsIcon
        icon={sharesWithProject ? Folder02Icon : AttachmentIcon}
        strokeWidth={2}
        className="size-3.5"
      />
      {compact ? null : (
        <span>
          {sharesWithProject
            ? "Add files for this project"
            : "Add files to chat with"}
        </span>
      )}
    </button>
  );
}

/** Picks whether new attachments go to the project (shared with every chat in it)
 * or to this chat alone. Only a project chat has the choice. */
function AttachmentTargetMenu({
  disabled,
  sharesWithProject,
  onSelect,
}: {
  disabled: boolean;
  sharesWithProject: boolean;
  onSelect: (target: ProjectAttachmentTarget) => void;
}) {
  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild={true}>
        <button
          type="button"
          disabled={disabled}
          aria-label="Choose where attached files go"
          title="Choose where attached files go"
          className="composer-pill-btn shrink-0 -translate-y-px !text-foreground/60 px-2"
        >
          <span className="text-ui-11">
            {sharesWithProject ? "Project" : "This chat"}
          </span>
        </button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="start" className="unsloth-plus-menu w-64">
        <DropdownMenuLabel>New files go to</DropdownMenuLabel>
        <DropdownMenuItem onSelect={() => onSelect("project")}>
          <HugeiconsIcon
            icon={sharesWithProject ? Tick02Icon : Folder02Icon}
            strokeWidth={1.75}
            className="size-icon"
          />
          <span className="flex flex-col">
            <span>The project</span>
            <span className="text-ui-11 text-muted-foreground">
              Every chat in this project can use them
            </span>
          </span>
        </DropdownMenuItem>
        <DropdownMenuItem onSelect={() => onSelect("thread")}>
          <HugeiconsIcon
            icon={sharesWithProject ? AttachmentIcon : Tick02Icon}
            strokeWidth={1.75}
            className="size-icon"
          />
          <span className="flex flex-col">
            <span>This chat only</span>
            <span className="text-ui-11 text-muted-foreground">
              Other chats in the project won't see them
            </span>
          </span>
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}

export function ThreadDocumentsBar({
  threadId,
  onIndexingChange,
}: {
  threadId: string | null;
  onIndexingChange?: (active: boolean) => void;
}) {
  const ragEnabled = useChatRuntimeStore((s) => s.ragEnabled);
  const ragSource = useChatRuntimeStore((s) => s.ragSource);
  const setRagSource = useChatRuntimeStore((s) => s.setRagSource);
  const setRagEnabled = useChatRuntimeStore((s) => s.setRagEnabled);
  // activeProjectId is set by the chat page from the thread's own row, so it is
  // null for every chat with no project.
  const activeProjectId = useChatRuntimeStore((s) => s.activeProjectId);
  const projectAttachmentTarget = useChatRuntimeStore(
    (s) => s.projectAttachmentTarget,
  );
  const setProjectAttachmentTarget = useChatRuntimeStore(
    (s) => s.setProjectAttachmentTarget,
  );
  const aui = useAui();
  // A KB-scoped chat uploads through the KB dialog, so neither scope applies there.
  const projectId = ragSource.type === "kb" ? null : activeProjectId;
  const sharesWithProject =
    projectId !== null && projectAttachmentTarget === "project";
  const fileInputRef = useRef<HTMLInputElement>(null);

  // A fresh chat has no thread id until the first message; materialize one on demand
  // so docs can attach (append() in runtime-provider reuses it). Track it locally:
  // pushing to global activeThreadId would, in a project, remount this bar mid-upload
  // (ProjectLanding's pendingNewThreadId branch) and drop the just-attached chips.
  const [materializedId, setMaterializedId] = useState<string | null>(null);
  const effectiveThreadId = threadId ?? materializedId;
  const initPromiseRef = useRef<Promise<string | null> | null>(null);
  const initGenerationRef = useRef(0);
  useEffect(() => {
    if (threadId) {
      setMaterializedId(null);
      initGenerationRef.current += 1;
      initPromiseRef.current = null;
    }
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

  // The project's shared sources, listed alongside this chat's own so a file added
  // from another chat is visible rather than silently in effect. Retrieval already
  // combines both scopes (core/rag/tool.py).
  const projectLister = useCallback(
    () => (projectId ? listProjectDocuments(projectId) : Promise.resolve([])),
    [projectId],
  );
  const {
    documents: projectDocuments,
    uploading: projectUploading,
    upload: uploadToProject,
    remove: removeFromProject,
  } = useRagDocuments(
    projectId && ragEnabled ? { type: "project", projectId } : null,
    projectLister,
  );

  // Tell the composer whether any doc is still indexing, so it can hold a queued
  // send until retrieval covers them (Composer.enqueueSend). For KB / RAG-off scope
  // is null, so both lists are empty and this reads false.
  const hasIndexing = [...documents, ...projectDocuments].some(
    (d) => d.status === "pending" || d.status === "running",
  );
  useEffect(() => {
    onIndexingChange?.(hasIndexing);
  }, [hasIndexing, onIndexingChange]);
  useEffect(() => () => onIndexingChange?.(false), [onIndexingChange]);

  // Materialize the thread id on first use; ref-deduped so a double-click can't
  // start two threads. A thread switch gets separate work even if the prior request is pending.
  const ensureThreadId = useCallback((): Promise<string | null> => {
    if (effectiveThreadId) {
      return requireStoredThread(effectiveThreadId).then(
        () => effectiveThreadId,
        () => {
          toast.error("Couldn't start a chat for these documents");
          return null;
        },
      );
    }
    const current = initPromiseRef.current;
    if (current) {
      return current;
    }
    const clearGeneration = chatHistoryClearBoundary.capture();
    const generation = ++initGenerationRef.current;
    const pending = aui
      .threadListItem()
      .initialize()
      .then(async ({ remoteId }) => {
        await requireStoredThread(remoteId);
        // a clear that landed while the row write was in flight is deleting this thread
        if (chatHistoryClearBoundary.capture() !== clearGeneration) {
          throw new Error("Chat history was cleared");
        }
        // an older request can still finish after the component moved to another thread
        if (initGenerationRef.current === generation) {
          setMaterializedId(remoteId);
        }
        return remoteId;
      })
      .catch(() => {
        toast.error("Couldn't start a chat for these documents");
        return null;
      });
    initPromiseRef.current = pending;
    const clear = () => {
      if (initPromiseRef.current === pending) {
        initPromiseRef.current = null;
      }
    };
    pending.then(clear, clear);
    return pending;
  }, [aui, effectiveThreadId]);

  // One entry point for the picker and desktop drops: project files go straight to
  // the project, per-chat files materialize the thread first. The sources probe
  // caches for 30s, so invalidate both sides of a project upload or a message sent
  // mid-index keeps answering from a cached "no sources".
  const attach = useCallback(
    (items: Parameters<typeof upload>[0]) => {
      if (sharesWithProject && projectId) {
        invalidateProjectSources(projectId);
        void uploadToProject(items).finally(() =>
          invalidateProjectSources(projectId),
        );
        return;
      }
      // Pass the id as a promise so upload() flips its in-flight guard before
      // materialization re-renders us; on the first click `scope` is still null.
      const threadScope = ensureThreadId().then((id) =>
        id ? ({ type: "thread", threadId: id } as const) : null,
      );
      void upload(items, threadScope);
    },
    [ensureThreadId, projectId, sharesWithProject, upload, uploadToProject],
  );

  // Desktop drops land in the native-intent store because the drop listener lives on
  // the chat page; only the chat that received the OS drop may drain its batch.
  const nativeAttachmentTargetKey = useNativeAttachmentTargetKey();
  const hasPendingAttachments = useNativeIntentStore((s) =>
    Boolean(
      nativeAttachmentTargetKey &&
        (s.pendingAttachments[nativeAttachmentTargetKey]?.length ?? 0) > 0,
    ),
  );
  useEffect(() => {
    if (!hasPendingAttachments || !nativeAttachmentTargetKey) {
      return;
    }
    // A KB-scoped chat uploads through the KB dialog, so a thread upload here would
    // index into something this bar never shows.
    if (ragEnabled && ragSource.type === "kb") {
      useNativeIntentStore.getState().takeAttachments(nativeAttachmentTargetKey);
      toast.error("This chat retrieves from a knowledge base", {
        description: "Add these files to the knowledge base instead.",
      });
      return;
    }
    const intents = useNativeIntentStore
      .getState()
      .takeAttachments(nativeAttachmentTargetKey);
    if (intents.length === 0) return;
    // A stale KB preference is inactive while RAG is off; use thread retrieval.
    if (!ragEnabled) {
      setRagSource({ type: "thread" });
      setRagEnabled(true);
    }
    attach(
      intents.map((intent) => ({
        kind: "native" as const,
        token: intent.path.token,
        name: intent.displayLabel,
        sizeBytes: intent.path.sizeBytes,
        modifiedMs: intent.path.modifiedMs,
      })),
    );
  }, [
    hasPendingAttachments,
    nativeAttachmentTargetKey,
    attach,
    ragEnabled,
    ragSource,
    setRagSource,
    setRagEnabled,
  ]);

  const chipScrollRef = useRef<HTMLDivElement>(null);
  const [chipsOverflow, setChipsOverflow] = useState(false);
  const updateChipFade = useCallback(() => {
    const el = chipScrollRef.current;
    if (!el) return;
    setChipsOverflow(el.scrollHeight - el.scrollTop - el.clientHeight > 1);
  }, []);
  useEffect(() => {
    updateChipFade();
  }, [documents, updateChipFade]);

  // Open the picker synchronously to keep the click's user activation. Do NOT
  // materialize here: setActiveThreadId while the native dialog is open can remount
  // the composer and orphan this <input>. Materialize in onChange instead.
  const handleAddDocs = useCallback(() => {
    fileInputRef.current?.click();
  }, []);

  // Shown whenever the RAG pill is on: ingestion only needs the embedder, so
  // files can index before a chat model loads.
  if (!ragEnabled) return null;
  // A KB source uploads via the KB dialog, not here; show which KB is active.
  if (ragSource.type === "kb") {
    return <KnowledgeBaseSourceChip kbId={ragSource.kbId} />;
  }

  const busy = uploading || projectUploading;
  const chipCount = documents.length + projectDocuments.length;

  return (
    <div className="mb-2 flex w-full flex-row items-start gap-1.5 pl-0.5 pr-1.5 pt-0.5 pb-1">
      <AttachFilesButton
        disabled={busy}
        compact={chipCount > 0}
        sharesWithProject={sharesWithProject}
        onClick={handleAddDocs}
      />
      {/* Only a project chat has two scopes to choose between. */}
      {projectId ? (
        <AttachmentTargetMenu
          disabled={busy}
          sharesWithProject={sharesWithProject}
          onSelect={setProjectAttachmentTarget}
        />
      ) : null}
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
          attach(files);
        }}
      />
      {/* Cap height so a large set scrolls; fade the cut-off row. */}
      <div
        ref={chipScrollRef}
        onScroll={updateChipFade}
        className={cn(
          "flex max-h-24 flex-1 flex-row flex-wrap items-center gap-1.5 overflow-y-auto",
          chipsOverflow && "rag-docs-bottom-fade",
        )}
      >
        {/* Project sources first: inherited context, and it outlives this chat. */}
        {projectDocuments.map((doc) => (
          <DocumentStatusChip
            key={`project:${doc.id}`}
            filename={doc.filename}
            status={doc.status}
            progress={doc.progress}
            error={doc.error}
            shared={true}
            onRemove={
              doc.id.startsWith("pending_") || isLinkedFolderManaged(doc)
                ? undefined
                : () => void removeFromProject(doc.id)
            }
          />
        ))}
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
