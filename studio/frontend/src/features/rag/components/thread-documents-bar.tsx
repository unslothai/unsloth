


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
  PENDING_CHAT_ATTACHMENT_KEY,
  readPendingAttachmentTargetClaim,
  useChatRuntimeStore,
} from "@/features/chat/stores/chat-runtime-store";
import type { ProjectAttachmentTarget } from "@/features/chat/utils/project-attachment-target";
import {
  chatHistoryClearBoundary,
  ChatThreadDeletedError,
  ensureStoredChatThread,
  getStoredChatThread,
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
  announceProjectSourcesUpdated,
  invalidateProjectSources,
  listKnowledgeBases,
  listProjectDocuments,
  listThreadDocuments,
} from "../api/rag-api";
import { useRagAvailabilityStore } from "../api/rag-availability";
import {
  type DocumentStatus,
  RAG_UPLOAD_ACCEPT,
  type RagDocument,
  isLinkedFolderManaged,
} from "../types/rag";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
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

/** Read-only listing of the project's sources, shown when the Docs pill is off:
 * they still reach the model, so they must not be invisible. */
function InheritedProjectSources({
  documents,
}: {
  documents: { id: string; filename: string; status: DocumentStatus }[];
}) {
  return (
    <div className="mb-2 flex w-full flex-row items-center gap-1.5 pl-0.5 pr-1.5 pt-0.5 pb-1">
      <span
        className="composer-pill-btn shrink-0 cursor-default !text-foreground/60"
        title="This chat retrieves from its project's sources. Manage them in the project's Sources tab."
      >
        <HugeiconsIcon icon={Folder02Icon} strokeWidth={2} className="size-3.5" />
        <span>Project sources</span>
      </span>
      {/* Same cap as the editable list: a linked folder can carry hundreds of
          sources, and an uncapped row would swallow the chat viewport. */}
      <div className="flex max-h-24 flex-1 flex-row flex-wrap items-center gap-1.5 overflow-y-auto">
        {documents.map((doc) => (
          <DocumentStatusChip
            key={`inherited:${doc.id}`}
            filename={doc.filename}
            status={doc.status}
            shared={true}
          />
        ))}
      </div>
    </div>
  );
}

/** The project the displayed chat belongs to, read from the chat's own row: the
 * global activeProjectId still names the project being left during a navigation.
 * `undefined` while unresolved, which holds the attach controls rather than
 * reading as "not in a project". */
/** Reads of the chat's own row before the scope is left unresolved. */
const PROJECT_LOOKUP_RETRIES = 3;

function useThreadProjectId(
  threadId: string | null,
): string | null | undefined {
  const activeProjectId = useChatRuntimeStore((s) => s.activeProjectId);
  const [resolved, setResolved] = useState<{
    threadId: string;
    // The activeProjectId this answer was produced for. A change means the chat
    // may have moved, so the old answer stops counting until the re-read lands.
    trigger: string | null;
    projectId: string | null;
  } | null>(null);

  // activeProjectId is a trigger, not the answer: moving the open chat updates
  // its row and this value without changing the thread id.
  useEffect(() => {
    if (!threadId || isThreadIncognito(threadId)) {
      return;
    }
    let cancelled = false;
    void (async () => {
      // A failed read is not proof of no project, and recording one would file
      // the next attachment into the chat. Retry, then leave it unresolved:
      // nothing re-runs this until the chat or the open project changes.
      for (let attempt = 0; attempt < PROJECT_LOOKUP_RETRIES; attempt += 1) {
        try {
          const thread = await getStoredChatThread(threadId);
          if (cancelled) return;
          // No row yet: initialize() does not await the write, so the composer's
          // project is the answer that row is about to record.
          const projectId = thread ? (thread.projectId ?? null) : activeProjectId;
          setResolved({ threadId, trigger: activeProjectId, projectId });
          return;
        } catch {
          if (cancelled) return;
          await new Promise((resolve) => setTimeout(resolve, 500 * (attempt + 1)));
          if (cancelled) return;
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [threadId, activeProjectId]);

  // A chat with no id yet is the one being composed, so it belongs to whatever
  // project the composer is in.
  if (!threadId) {
    return activeProjectId;
  }
  if (isThreadIncognito(threadId)) {
    return null;
  }
  return resolved?.threadId === threadId && resolved.trigger === activeProjectId
    ? resolved.projectId
    : undefined;
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
  const projectAttachmentDefault = useChatRuntimeStore(
    (s) => s.projectAttachmentTarget,
  );
  const projectAttachmentTargetByThread = useChatRuntimeStore(
    (s) => s.projectAttachmentTargetByThread,
  );
  const setThreadProjectAttachmentTarget = useChatRuntimeStore(
    (s) => s.setThreadProjectAttachmentTarget,
  );
  const aui = useAui();
  const fileInputRef = useRef<HTMLInputElement>(null);

  // A fresh chat has no thread id until the first message; materialize one on demand
  // so docs can attach (append() in runtime-provider reuses it). Track it locally:
  // pushing to global activeThreadId would, in a project, remount this bar mid-upload
  // (ProjectLanding's pendingNewThreadId branch) and drop the just-attached chips.
  const [materializedId, setMaterializedId] = useState<string | null>(null);
  const effectiveThreadId = threadId ?? materializedId;
  const initPromiseRef = useRef<Promise<string | null> | null>(null);
  const initGenerationRef = useRef(0);
  const hadThreadIdRef = useRef(threadId !== null);
  useEffect(() => {
    const hadThreadId = hadThreadIdRef.current;
    hadThreadIdRef.current = threadId !== null;
    if (!threadId) {
      return;
    }
    // A plain send creates the chat too, without ensureThreadId. Hand the
    // earlier choice to the chat that just got an id, or the next one inherits it.
    if (!hadThreadId) {
      useChatRuntimeStore.getState().adoptPendingProjectAttachmentTarget(threadId);
    }
    setMaterializedId(null);
    initGenerationRef.current += 1;
    initPromiseRef.current = null;
  }, [threadId]);

  // An abandoned composer leaves its choice under the pending key, where the
  // next new chat would claim it. Adoption removes the key, so this drops only
  // what nobody claimed.
  useEffect(
    () => () =>
      useChatRuntimeStore.getState().clearPendingProjectAttachmentTarget(),
    [],
  );

  // Mirrors chat-adapter's rag_scope: an active KB replaces the project scope,
  // but a KB preference left over while the pill is off does not.
  const threadProjectId = useThreadProjectId(effectiveThreadId);
  // Attaching before the row has been read would file the file by guess.
  const projectUnresolved = threadProjectId === undefined;
  // A host where the vector extension cannot load answers 503 to every project
  // source request, so do not open a scope it can only fail.
  const ragUnavailable = useRagAvailabilityStore((s) => s.isUnavailable());
  const projectId =
    (ragEnabled && ragSource.type === "kb") || ragUnavailable
      ? null
      : (threadProjectId ?? null);
  // This chat's own choice if it made one, otherwise the saved default. Keeps a
  // pick in one chat from redirecting every other chat in the project.
  const projectAttachmentTarget =
    projectAttachmentTargetByThread[
      effectiveThreadId ?? PENDING_CHAT_ATTACHMENT_KEY
    ] ?? projectAttachmentDefault;
  const sharesWithProject =
    projectId !== null && projectAttachmentTarget === "project";

  const lister = useCallback(
    () =>
      effectiveThreadId
        ? listThreadDocuments(effectiveThreadId)
        : Promise.resolve([]),
    [effectiveThreadId],
  );
  const {
    documents,
    uploading,
    hasIndexing: threadIndexing,
    loading: threadListLoading,
    upload,
    remove,
  } = useRagDocuments(
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
    hasIndexing: projectIndexing,
    loading: projectListLoading,
    upload: uploadToProject,
    remove: removeFromProject,
  } = useRagDocuments(
    projectId ? { type: "project", projectId } : null,
    projectLister,
  );

  // Tell the composer whether any doc is still indexing, so it can hold a queued
  // send until retrieval covers them (Composer.enqueueSend). For KB / RAG-off scope
  // is null, so both lists are empty and this reads false.
  // From the hooks, not the rows: work started in the Sources panel is in flight
  // before either instance has a row for it, and a job already running on a
  // reopened project arrives with the first list, so hold until that lands.
  // Both scopes hold on their first list, for the same reason: reopening a chat
  // whose own attachment was still indexing lists nothing until it lands either.
  const hasIndexing =
    threadIndexing || threadListLoading || projectIndexing || projectListLoading;
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
    // Taken before the await: this composer can be abandoned while it runs, and
    // the choice under the shared key would then be the next composer's.
    const claim = readPendingAttachmentTargetClaim();
    const pending = aui
      .threadListItem()
      .initialize()
      .then(async ({ remoteId }) => {
        await requireStoredThread(remoteId);
        useChatRuntimeStore
          .getState()
          .adoptPendingProjectAttachmentTarget(remoteId, claim);
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

  // One entry point for the picker and desktop drops: project files go straight
  // there, per-chat files materialize the thread first. The probe caches for 30s,
  // so invalidate both sides or a send mid-index reads a stale "no sources".
  const attach = useCallback(
    (items: Parameters<typeof upload>[0]) => {
      if (sharesWithProject && projectId) {
        invalidateProjectSources(projectId);
        // Explicit scope: a desktop drop enables RAG and attaches in the same
        // tick, so the hook's own scope is still null on this render.
        void uploadToProject(items, { type: "project", projectId }).finally(() =>
          announceProjectSourcesUpdated(projectId),
        );
        return;
      }
      // The id as a promise, so upload() flips its in-flight guard before
      // materialization re-renders us: on the first click `scope` is null.
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
    // Hold the batch rather than draining it into the wrong scope. The intents
    // stay in the store, so this runs again once the row has been read.
    if (projectUnresolved) {
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
    projectUnresolved,
    nativeAttachmentTargetKey,
    attach,
    ragEnabled,
    ragSource,
    setRagSource,
    setRagEnabled,
  ]);

  const chipScrollRef = useRef<HTMLDivElement>(null);
  const [chipsOverflow, setChipsOverflow] = useState(false);
  // Removing a project source here deletes it for every chat, beside a chat chip
  // whose X is undoable. Confirm, as the Sources tab and Settings do.
  const [removingShared, setRemovingShared] = useState<RagDocument | null>(null);
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

  // A KB source uploads via the KB dialog, not here; show which KB is active.
  if (ragEnabled && ragSource.type === "kb") {
    return <KnowledgeBaseSourceChip kbId={ragSource.kbId} />;
  }
  // Project sources retrieve whether the Docs pill is on or not (chat-adapter's
  // projectRagEnabled), so list them either way rather than letting the model
  // answer from files the user cannot see. The attach controls stay behind the
  // pill: with it off, thread scope is inert.
  if (!ragEnabled) {
    return projectDocuments.length > 0 ? (
      <InheritedProjectSources documents={projectDocuments} />
    ) : null;
  }

  // Attaching before the chat's project is known would file the file by guess.
  const busy = uploading || projectUploading || projectUnresolved;
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
          onSelect={(target) =>
            setThreadProjectAttachmentTarget(effectiveThreadId, target)
          }
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
                : () => setRemovingShared(doc)
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
      <AlertDialog
        open={removingShared !== null}
        onOpenChange={(open) => {
          if (!open) setRemovingShared(null);
        }}
      >
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Remove from project sources</AlertDialogTitle>
            <AlertDialogDescription>
              Remove "{removingShared?.filename}"? Every chat in this project
              loses it, and the file and its indexed content are deleted. This
              cannot be undone.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction
              onClick={() => {
                const doc = removingShared;
                setRemovingShared(null);
                if (doc) void removeFromProject(doc.id);
              }}
            >
              Remove
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
}
