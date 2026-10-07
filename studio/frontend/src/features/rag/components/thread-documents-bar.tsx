// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  type RefObject,
  useCallback,
  useEffect,
  useRef,
  useState,
} from "react";
import { HugeiconsIcon } from "@hugeicons/react";
import {
  AttachmentIcon,
  FileDatabaseIcon,
  FolderAttachmentIcon,
  Folder02Icon,
} from "@hugeicons/core-free-icons";
import { Tick02Icon } from "@/lib/tick-icon";
import { ChevronDownStandardIcon } from "@/lib/chevron-icons";
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
  subscribeKnowledgeBasesChanged,
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
import {
  type KnowledgeBaseFocus,
  KnowledgeBaseDialog,
} from "./knowledge-base-dialog";
import { EXPIRY_GRACE_MS } from "./staged-source";
import { uploadItemFromIntent, useRagDocuments } from "./use-rag-documents";

function useKnowledgeBaseName(kbId: string | null): string | null {
  // Keyed by id: a drop during a switch must not name one KB and add to another.
  const [known, setKnown] = useState<{ kbId: string; name: string | null } | null>(
    null,
  );
  useEffect(() => {
    if (!kbId) return;
    let cancelled = false;
    let latest = 0;
    const load = () => {
      const request = ++latest;
      listKnowledgeBases()
        .then((rows) => {
          if (cancelled || request !== latest) return;
          setKnown({ kbId, name: rows.find((kb) => kb.id === kbId)?.name ?? null });
        })
        .catch(() => {
          // A failed refetch says nothing about the KB.
        });
    };
    load();
    const unsubscribe = subscribeKnowledgeBasesChanged(load);
    return () => {
      cancelled = true;
      unsubscribe();
    };
  }, [kbId]);
  return known !== null && known.kbId === kbId ? known.name : null;
}

function KnowledgeBaseSourceChip({
  name,
  onOpen,
  buttonRef,
}: {
  name: string | null;
  onOpen: () => void;
  buttonRef: RefObject<HTMLButtonElement | null>;
}) {
  return (
    <div className="mb-2 flex w-full flex-row items-center gap-1.5 pl-0.5 pr-1.5 pt-0.5 pb-1">
      <button
        ref={buttonRef}
        type="button"
        onClick={onOpen}
        className="composer-pill-btn min-w-0 max-w-full"
        title="Add or remove documents"
      >
        <HugeiconsIcon
          icon={FileDatabaseIcon}
          strokeWidth={2}
          className="size-3.5 shrink-0"
        />
        <span className="min-w-0 truncate">
          {name ? `Knowledge base: ${name}` : "Knowledge base"}
        </span>
      </button>
    </div>
  );
}

/**
 * The thread id can arrive before its row write lands, and upload_thread_document does not
 * check the thread; only a definitive miss blocks the upload.
 */
async function requireStoredThread(threadId: string): Promise<void> {
  if (isThreadIncognito(threadId)) return;
  let stored: Awaited<ReturnType<typeof ensureStoredChatThread>>;
  try {
    stored = await ensureStoredChatThread(threadId);
  } catch (error) {
    // A tombstone is an answer: indexing against it would orphan the documents.
    if (error instanceof ChatThreadDeletedError) {
      throw error;
    }
    return;
  }
  if (!stored) {
    throw new Error(`Thread ${threadId} was not persisted`);
  }
}

/** Shown when the Docs pill is off: project sources still reach the model. */
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
        <HugeiconsIcon icon={FolderAttachmentIcon} strokeWidth={2} className="size-3.5" />
        <span>Project sources</span>
      </span>
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

/** Reads of the chat's own row before the scope is left unresolved. */
const PROJECT_LOOKUP_RETRIES = 3;

/** Read from the chat's own row: activeProjectId still names the old project mid-navigation.
 * `undefined` while unresolved. */
function useThreadProjectId(
  threadId: string | null,
): string | null | undefined {
  const activeProjectId = useChatRuntimeStore((s) => s.activeProjectId);
  const [resolved, setResolved] = useState<{
    threadId: string;
    trigger: string | null;
    projectId: string | null;
  } | null>(null);

  useEffect(() => {
    if (!threadId || isThreadIncognito(threadId)) {
      return;
    }
    let cancelled = false;
    void (async () => {
      // A failed read is not proof of no project; retry, then leave it unresolved.
      for (let attempt = 0; attempt < PROJECT_LOOKUP_RETRIES; attempt += 1) {
        try {
          const thread = await getStoredChatThread(threadId);
          if (cancelled) return;
          // No row yet: initialize() does not await the write, so use the composer's project.
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

function AttachFilesButton({
  disabled,
  compact,
  sharesWithProject,
  onClick,
}: {
  disabled: boolean;
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

const ATTACHMENT_TARGETS: {
  value: ProjectAttachmentTarget;
  icon: typeof Folder02Icon;
  title: string;
  description: string;
}[] = [
  {
    value: "project",
    icon: Folder02Icon,
    title: "The project",
    description: "Every chat in this project can use them",
  },
  {
    value: "thread",
    icon: AttachmentIcon,
    title: "This chat only",
    description: "Other chats in the project won't see them",
  },
];

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
  const current = sharesWithProject ? "project" : "thread";
  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild={true}>
        {/* Caret and resting fill so it reads as a picker. */}
        <button
          type="button"
          disabled={disabled}
          aria-label="Choose where attached files go"
          title="Choose where attached files go"
          className="composer-pill-btn attachment-target-pill shrink-0 -translate-y-px gap-1 !text-foreground/70 pl-3 pr-1.5"
        >
          <span className="text-ui-12">
            {sharesWithProject ? "Project" : "This chat"}
          </span>
          <HugeiconsIcon
            icon={ChevronDownStandardIcon}
            strokeWidth={1.5}
            className="composer-pill-caret size-[calc(14px*var(--ui-space-scale,1))]"
          />
        </button>
      </DropdownMenuTrigger>
      <DropdownMenuContent
        align="start"
        className="unsloth-plus-menu w-[calc(328px*var(--ui-space-scale,1))]"
      >
        <DropdownMenuLabel>New files go to</DropdownMenuLabel>
        {ATTACHMENT_TARGETS.map((target) => (
          <DropdownMenuItem
            key={target.value}
            onSelect={() => onSelect(target.value)}
            className="items-start"
          >
            {/* Icon aligns with the title line. */}
            <span className="flex h-[1lh] shrink-0 items-center">
              <HugeiconsIcon
                icon={target.icon}
                strokeWidth={1.75}
                className="size-icon"
              />
            </span>
            <span className="flex min-w-0 flex-1 flex-col gap-0.5">
              <span>{target.title}</span>
              <span className="text-ui-12 leading-snug text-muted-foreground">
                {target.description}
              </span>
            </span>
            {/* Tick centred on the row. */}
            <HugeiconsIcon
              icon={Tick02Icon}
              strokeWidth={2}
              className={cn(
                "unsloth-tick shrink-0 self-center",
                current !== target.value && "opacity-0",
              )}
            />
          </DropdownMenuItem>
        ))}
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

  // Track the materialized id locally: setting activeThreadId in a project remounts this bar
  // mid-upload (ProjectLanding's pendingNewThreadId branch) and drops the new chips.
  const [materializedId, setMaterializedId] = useState<string | null>(null);
  const effectiveThreadId = threadId ?? materializedId;
  const initPromiseRef = useRef<Promise<string> | null>(null);
  const initGenerationRef = useRef(0);
  const hadThreadIdRef = useRef(threadId !== null);
  useEffect(() => {
    const hadThreadId = hadThreadIdRef.current;
    hadThreadIdRef.current = threadId !== null;
    if (!threadId) {
      return;
    }
    // A plain send creates the chat too; hand it the earlier choice, or the next chat inherits it.
    if (!hadThreadId) {
      useChatRuntimeStore.getState().adoptPendingProjectAttachmentTarget(threadId);
    }
    setMaterializedId(null);
    initGenerationRef.current += 1;
    initPromiseRef.current = null;
  }, [threadId]);

  // Adoption removes the pending key, so this drops only a choice nobody claimed.
  useEffect(
    () => () =>
      useChatRuntimeStore.getState().clearPendingProjectAttachmentTarget(),
    [],
  );

  // Mirrors chat-adapter's rag_scope: an active KB replaces the project scope.
  const threadProjectId = useThreadProjectId(effectiveThreadId);
  const projectUnresolved = threadProjectId === undefined;
  const ragUnavailable = useRagAvailabilityStore((s) => s.isUnavailable());
  const projectId =
    (ragEnabled && ragSource.type === "kb") || ragUnavailable
      ? null
      : (threadProjectId ?? null);
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

  // Retrieval already combines both scopes (core/rag/tool.py).
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

  // Lets the composer hold a queued send until retrieval covers indexing docs
  // (Composer.enqueueSend); both scopes hold until their first list lands.
  const hasIndexing =
    threadIndexing || threadListLoading || projectIndexing || projectListLoading;
  useEffect(() => {
    onIndexingChange?.(hasIndexing);
  }, [hasIndexing, onIndexingChange]);
  useEffect(() => () => onIndexingChange?.(false), [onIndexingChange]);

  // Ref-deduped so a double-click cannot start two threads.
  const ensureThreadId = useCallback((): Promise<string> => {
    // Only initialize when the id belongs to the current uninitialized item; mid-navigation that
    // would create and attach to the wrong chat.
    const currentItem = aui.threadListItem().getState();
    if (
      effectiveThreadId &&
      (currentItem.remoteId || currentItem.id !== effectiveThreadId)
    ) {
      return requireStoredThread(effectiveThreadId).then(
        () => effectiveThreadId,
      );
    }
    const current = initPromiseRef.current;
    if (current) {
      return current;
    }
    const clearGeneration = chatHistoryClearBoundary.capture();
    const generation = ++initGenerationRef.current;
    // Taken before the await: this composer can be abandoned while it runs.
    const claim = readPendingAttachmentTargetClaim();
    const pending = aui
      .threadListItem()
      .initialize()
      .then(async ({ remoteId }) => {
        await requireStoredThread(remoteId);
        useChatRuntimeStore
          .getState()
          .adoptPendingProjectAttachmentTarget(remoteId, claim);
        // A clear that landed while the row write was in flight is deleting this thread.
        if (chatHistoryClearBoundary.capture() !== clearGeneration) {
          throw new Error("Chat history was cleared");
        }
        // An older request can still finish after the component moved to another thread.
        if (initGenerationRef.current === generation) {
          setMaterializedId(remoteId);
        }
        return remoteId;
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

  // The sources probe caches for 30s, so invalidate both sides.
  const attach = useCallback(
    (items: Parameters<typeof upload>[0]) => {
      if (sharesWithProject && projectId) {
        invalidateProjectSources(projectId);
        // Explicit scope: a desktop drop enables RAG and attaches in the same tick.
        void uploadToProject(items, { type: "project", projectId }).finally(() =>
          announceProjectSourcesUpdated(projectId),
        );
        return;
      }
      void upload(items, async () => ({
        type: "thread",
        threadId: await ensureThreadId(),
      }));
    },
    [ensureThreadId, projectId, sharesWithProject, upload, uploadToProject],
  );

  // Only the chat that received the OS drop may drain its batch.
  const nativeAttachmentTargetKey = useNativeAttachmentTargetKey();
  const [kbDialogFocus, setKbDialogFocus] = useState<KnowledgeBaseFocus | null>(
    null,
  );
  const kbChipRef = useRef<HTMLButtonElement>(null);
  // The toaster outlives this bar; an "Add" after unmount would open nothing.
  const kbDropOffersRef = useRef(new Set<string | number>());
  useEffect(() => {
    const offers = kbDropOffersRef.current;
    return () => {
      for (const id of offers) toast.dismiss(id);
      offers.clear();
    };
  }, []);
  const activeKbName = useKnowledgeBaseName(
    ragEnabled && ragSource.type === "kb" ? ragSource.kbId : null,
  );
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
    if (projectUnresolved) {
      return;
    }
    const store = useNativeIntentStore.getState();
    const intents = store.takeAttachments(nativeAttachmentTargetKey);
    if (intents.length === 0) {
      return;
    }
    if (ragEnabled && ragSource.type === "kb") {
      const kbId = ragSource.kbId;
      const files =
        intents.length === 1
          ? `"${intents[0].displayLabel}"`
          : `${intents.length} files`;
      const target = activeKbName
        ? `"${activeKbName}"`
        : "this chat's knowledge base";
      const expiresAt = Math.min(...intents.map((intent) => intent.path.expiresAtMs));
      const offer = toast(`Add ${files} to ${target}?`, {
        description:
          "This chat retrieves from that knowledge base, not from files dropped in the chat.",
        duration: Number.isFinite(expiresAt)
          ? Math.max(8_000, expiresAt - EXPIRY_GRACE_MS - Date.now())
          : Infinity,
        action: {
          label: "Add",
          onClick: () =>
            setKbDialogFocus({
              kbId,
              uploads: intents.map(uploadItemFromIntent),
            }),
        },
      });
      kbDropOffersRef.current.add(offer);
      return;
    }
    if (!ragEnabled) {
      setRagSource({ type: "thread" });
      setRagEnabled(true);
    }
    attach(intents.map(uploadItemFromIntent));
  }, [
    hasPendingAttachments,
    projectUnresolved,
    nativeAttachmentTargetKey,
    attach,
    ragEnabled,
    ragSource,
    activeKbName,
    setRagSource,
    setRagEnabled,
  ]);

  const chipScrollRef = useRef<HTMLDivElement>(null);
  const [chipsOverflow, setChipsOverflow] = useState(false);
  const [removingShared, setRemovingShared] = useState<RagDocument | null>(null);
  const updateChipFade = useCallback(() => {
    const el = chipScrollRef.current;
    if (!el) return;
    setChipsOverflow(el.scrollHeight - el.scrollTop - el.clientHeight > 1);
  }, []);
  useEffect(() => {
    updateChipFade();
  }, [documents, updateChipFade]);

  // Open the picker synchronously to keep user activation; materializing here can remount the
  // composer and orphan this <input>.
  const handleAddDocs = useCallback(() => {
    fileInputRef.current?.click();
  }, []);

  // Always the first fragment child so it is never remounted across branches.
  const kbDialog = (
    <KnowledgeBaseDialog
      open={kbDialogFocus !== null}
      onOpenChange={(next) => {
        if (!next) setKbDialogFocus(null);
      }}
      focus={kbDialogFocus}
      onCloseAutoFocus={(event) => {
        const chip = kbChipRef.current;
        if (chip?.isConnected) {
          event.preventDefault();
          chip.focus({ preventScroll: true });
        }
      }}
    />
  );

  if (ragEnabled && ragSource.type === "kb") {
    const kbId = ragSource.kbId;
    return (
      <>
        {kbDialog}
        <KnowledgeBaseSourceChip
          name={activeKbName}
          onOpen={() => setKbDialogFocus({ kbId })}
          buttonRef={kbChipRef}
        />
      </>
    );
  }
  // Project sources retrieve whether or not the Docs pill is on (chat-adapter's projectRagEnabled).
  if (!ragEnabled) {
    return (
      <>
        {kbDialog}
        {projectDocuments.length > 0 ? (
          <InheritedProjectSources documents={projectDocuments} />
        ) : null}
      </>
    );
  }

  const busy = uploading || projectUploading || projectUnresolved;
  const chipCount = documents.length + projectDocuments.length;

  return (
    <>
      {kbDialog}
      <div className="mb-2 flex w-full flex-row items-start gap-1.5 pl-0.5 pr-1.5 pt-0.5 pb-1">
        {/* Centred together; the row stays top-aligned for chips. */}
        <div className="flex shrink-0 items-center gap-1.5">
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
        </div>
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
        <div
          ref={chipScrollRef}
          onScroll={updateChipFade}
          className={cn(
            "flex max-h-24 flex-1 flex-row flex-wrap items-center gap-1.5 overflow-y-auto",
            chipsOverflow && "rag-docs-bottom-fade",
          )}
        >
          {projectDocuments.map((doc) => (
            <DocumentStatusChip
              key={`project:${doc.id}`}
              filename={doc.filename}
              status={doc.status}
              progress={doc.progress}
              stage={doc.stage}
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
              stage={doc.stage}
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
    </>
  );
}
