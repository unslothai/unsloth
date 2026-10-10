// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  type ReactNode,
  useCallback,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { ChevronLeftIcon, ChevronRightIcon, PlusIcon, XIcon } from "lucide-react";
import { HugeiconsIcon } from "@hugeicons/react";
import {
  AttachmentIcon,
  FileDatabaseIcon,
  FolderAttachmentIcon,
  Folder02Icon,
} from "@hugeicons/core-free-icons";
import {
  AttachmentKindIcon,
  CARD_EDGE,
  CARD_SIZE,
  CARD_SLOT,
  CARD_SURFACE,
  FileCardBody,
} from "@/components/assistant-ui/attachment";
import { Spinner } from "@/components/ui/spinner";
import { Tick02Icon } from "@/lib/tick-icon";
import { ChevronDownStandardIcon } from "@/lib/chevron-icons";
import { useAui } from "@assistant-ui/react";
import { cn } from "@/lib/utils";
import {
  PENDING_CHAT_ATTACHMENT_KEY,
  readPendingAttachmentTargetClaim,
  useChatRuntimeStore,
} from "@/features/chat/stores/chat-runtime-store";
import { useRagToolDisabled } from "@/features/chat/hooks/use-rag-tool-disabled";
import type { ProjectAttachmentTarget } from "@/features/chat/utils/project-attachment-target";
import {
  chatHistoryClearBoundary,
  ChatThreadDeletedError,
  ensureStoredChatThread,
  getStoredChatThread,
  annotationsOfFile,
  attachmentFileKind,
  isPastedTextFile,
  isThreadIncognito,
} from "@/features/chat";
import {
  useNativeAttachmentTargetKey,
  useNativeIntentStore,
} from "@/features/native-intents";
import { openFilePicker } from "@/lib/open-file-picker";
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
  listLinkedFolders,
  subscribeKnowledgeBasesChanged,
  listProjectDocuments,
  listThreadDocuments,
} from "../api/rag-api";
import { useRagAvailabilityStore } from "../api/rag-availability";
import { type RagDocument, isLinkedFolderManaged } from "../types/rag";
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
import { STAGE_LABELS } from "./document-status-chip";
import { useDocumentPreviewStore } from "./preview-store";
import {
  type KnowledgeBaseFocus,
  KnowledgeBaseDialog,
} from "./knowledge-base-dialog";
import {
  RAG_SOURCE_UPLOAD_ACCEPT,
  SUPPORTED_SOURCES_HINT,
  isSupportedSourceName,
} from "./source-drop-policy";
import { EXPIRY_GRACE_MS } from "./staged-source";
import {
  type RagUploadItem,
  type TrackedDocument,
  uploadItemFromIntent,
  useRagDocuments,
} from "./use-rag-documents";
import { useSourceDrop } from "./use-source-drop";

// Refetched after any KB mutation so a rename shows at once.
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
    // tombstones block indexing; other errors do not prove the thread is gone.
    if (error instanceof ChatThreadDeletedError) {
      throw error;
    }
    return;
  }
  if (!stored) {
    throw new Error(`Thread ${threadId} was not persisted`);
  }
}

/** row data avoids stale activeProjectId; undefined blocks attachments. */
/** retries chat row reads before leaving the project unresolved. */
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

function isRagIndexable(file: File): boolean {
  // Pasted text and annotations are not documents.
  if (isPastedTextFile(file) || annotationsOfFile(file)) return false;
  return isSupportedSourceName(file.name);
}

/** Display names of a project's linked folders, read once per set of folder ids. */
function useLinkedFolderNames(
  projectId: string | null,
  folderIds: string,
): ReadonlyMap<string, string> {
  const [names, setNames] = useState<ReadonlyMap<string, string>>(new Map());
  useEffect(() => {
    if (!projectId || !folderIds) return;
    let cancelled = false;
    listLinkedFolders({ type: "project", id: projectId })
      .then((folders) => {
        if (cancelled) return;
        setNames(new Map(folders.map((f) => [f.id, f.displayName])));
      })
      .catch(() => {
        // The card falls back to a generic label.
      });
    return () => {
      cancelled = true;
    };
  }, [projectId, folderIds]);
  return names;
}

/** Splits project documents into loose files and per-folder groups. A linked folder can
 * hold thousands of files, so it is drawn as one card rather than one card per file. */
function groupByLinkedFolder(docs: TrackedDocument[]): {
  loose: TrackedDocument[];
  folders: [string, TrackedDocument[]][];
} {
  const loose: TrackedDocument[] = [];
  const folders = new Map<string, TrackedDocument[]>();
  for (const doc of docs) {
    const folderId = doc.linkedFolderId;
    if (!folderId) {
      loose.push(doc);
      continue;
    }
    const group = folders.get(folderId);
    if (group) group.push(doc);
    else folders.set(folderId, [doc]);
  }
  return { loose, folders: [...folders] };
}

function isProcessing(doc: TrackedDocument): boolean {
  return doc.status === "pending" || doc.status === "running";
}

function ProjectBadge() {
  return (
    <span
      className="pointer-events-none absolute top-1.5 left-1.5 inline-flex items-center gap-1 rounded-full bg-muted px-1.5 py-0.5 text-ui-10 text-muted-foreground"
      aria-hidden={true}
    >
      <HugeiconsIcon icon={Folder02Icon} strokeWidth={2} className="size-3" />
      Project
    </span>
  );
}

function LinkedFolderCard({
  name,
  docs,
}: {
  name: string | undefined;
  docs: TrackedDocument[];
}) {
  const indexing = docs.filter(isProcessing).length;
  const label = name ?? "Linked folder";
  const count = `${docs.length} ${docs.length === 1 ? "file" : "files"}`;
  return (
    <div className={cn("relative", CARD_SLOT)}>
      <ProjectBadge />
      <div
        title={`${label}: ${count}, kept in sync and shared with every chat in this project. Manage it in the project's Sources tab.`}
        className={cn("flex overflow-hidden rounded-[15px]", CARD_SIZE, CARD_EDGE)}
      >
        <FileCardBody
          name={label}
          kind="document"
          icon={
            <HugeiconsIcon
              icon={Folder02Icon}
              strokeWidth={1.75}
              className="size-3.25 shrink-0 text-muted-foreground"
            />
          }
          center={
            <span className="flex flex-col items-center gap-1.5 text-center text-ui-11">
              {indexing > 0 ? (
                <Spinner className="size-4" />
              ) : (
                <HugeiconsIcon icon={Folder02Icon} strokeWidth={1.5} className="size-6" />
              )}
              <span>{indexing > 0 ? `Indexing ${indexing} of ${docs.length}` : count}</span>
              <span className="sr-only">linked folder, shared with the project</span>
            </span>
          }
        />
      </div>
    </div>
  );
}

/** An indexed document, drawn as a composer attachment card. */
function DocumentCard({
  doc,
  shared,
  onRemove,
}: {
  doc: TrackedDocument;
  shared: boolean;
  onRemove?: () => void;
}) {
  const openPreview = useDocumentPreviewStore((s) => s.openPreview);
  const kind = attachmentFileKind(doc.filename, undefined);
  const processing = isProcessing(doc);
  const failed = doc.status === "failed";
  const pending = doc.id.startsWith("pending_");
  const percent =
    doc.progress != null
      ? Math.round(doc.progress <= 1 ? doc.progress * 100 : doc.progress)
      : null;
  const stageLabel = (doc.stage && STAGE_LABELS[doc.stage]) || "Indexing";
  const center = processing ? (
    <span className="flex flex-col items-center gap-1.5 text-center text-ui-11">
      <Spinner className="size-4" />
      <span className="line-clamp-2">
        {stageLabel}
        {percent != null ? ` ${percent}%` : null}
      </span>
    </span>
  ) : failed ? (
    <span className="line-clamp-3 text-center text-ui-11 text-destructive">
      Couldn't index
    </span>
  ) : (
    <AttachmentKindIcon kind={kind} className="size-6" />
  );
  const canOpen = doc.status === "completed" && !pending;
  return (
    <div
      className={cn(
        "group/attachment-card relative",
        CARD_SLOT,
      )}
    >
      <button
        type="button"
        disabled={!canOpen}
        onClick={() =>
          openPreview({ documentId: doc.id, filename: doc.filename })
        }
        title={
          doc.error ??
          (shared
            ? `${doc.filename}: shared with every chat in this project`
            : doc.filename)
        }
        aria-label={[
          canOpen ? `Preview ${doc.filename}` : doc.filename,
          processing ? `${stageLabel}${percent != null ? ` ${percent}%` : ""}` : null,
          failed ? "couldn't index" : null,
          shared ? "shared with the project" : null,
        ]
          .filter(Boolean)
          .join(", ")}
        className={cn(
          "flex overflow-hidden rounded-[15px] text-left transition-colors disabled:cursor-default",
          CARD_SIZE,
          CARD_EDGE,
          canOpen && CARD_SURFACE,
          canOpen && "cursor-pointer",
          failed && "border-destructive/40",
        )}
      >
        <FileCardBody name={doc.filename} kind={kind} center={center} />
      </button>
      {shared ? <ProjectBadge /> : null}
      {onRemove && !processing ? (
        <button
          type="button"
          onClick={onRemove}
          aria-label={`Remove ${doc.filename}`}
          className="absolute top-1.5 right-1.5 flex size-5 items-center justify-center rounded-full bg-foreground text-background opacity-0 shadow-sm transition-opacity focus-visible:opacity-100 group-hover/attachment-card:opacity-100 group-focus-within/attachment-card:opacity-100 [@media(pointer:coarse)]:opacity-100"
        >
          <XIcon className="size-3 stroke-[2.5px]" />
        </button>
      ) : null}
    </div>
  );
}

/** Trailing "Add files" card. */
function AddFilesCard({
  disabled,
  onClick,
}: {
  disabled: boolean;
  onClick: () => void;
}) {
  return (
    <div className={CARD_SLOT}>
      <button
        type="button"
        disabled={disabled}
        onClick={onClick}
        className={cn(
          "unsloth-files-add-card flex flex-col items-center justify-center gap-1.5 rounded-[15px] border border-dashed border-[color-mix(in_oklab,var(--foreground)_calc(20%*var(--contrast-edge-gain,1)),transparent)] text-ui-12 text-muted-foreground transition-colors hover:border-[color-mix(in_oklab,var(--foreground)_calc(35%*var(--contrast-edge-gain,1)),transparent)] hover:text-foreground",
          CARD_SIZE,
        )}
      >
        <PlusIcon className="size-5 stroke-[1.5px]" />
        Add files
      </button>
    </div>
  );
}

/** Card above the composer: a header with controls over a strip of the chat's files. */
function ChatFilesPanel({
  icon,
  title,
  titleSuffix,
  note,
  headerControls,
  onClose,
  closeLabel,
  onDropItems,
  dropDisabledReason,
  children,
}: {
  icon: typeof Folder02Icon;
  title: string;
  /** Muted text after the title. */
  titleSuffix?: string;
  note?: ReactNode;
  headerControls?: ReactNode;
  onClose?: () => void;
  closeLabel?: string;
  /** Handles browser and desktop drops on the panel. Unset, drops fall through to the composer. */
  onDropItems?: (items: RagUploadItem[]) => void;
  /** Set while the panel can't take files: a drop is still claimed and refused with this. */
  dropDisabledReason?: string;
  children?: ReactNode;
}) {
  const stripRef = useRef<HTMLDivElement>(null);
  const [scroll, setScroll] = useState({ back: false, forward: false });
  const measure = useCallback(() => {
    const el = stripRef.current;
    if (!el) return;
    const back = el.scrollLeft > 1;
    const forward = el.scrollWidth - el.clientWidth - el.scrollLeft > 1;
    setScroll((prev) =>
      prev.back === back && prev.forward === forward ? prev : { back, forward },
    );
  }, []);
  useLayoutEffect(() => {
    const el = stripRef.current;
    if (!el) return;
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(el);
    // Adding a card does not resize the strip.
    const mutations = new MutationObserver(measure);
    mutations.observe(el, { childList: true });
    return () => {
      observer.disconnect();
      mutations.disconnect();
    };
  }, [measure]);
  const page = (direction: 1 | -1) => {
    const el = stripRef.current;
    if (!el) return;
    el.scrollBy({ left: direction * el.clientWidth * 0.8, behavior: "smooth" });
  };
  const overflowing = scroll.back || scroll.forward;
  // Drops here are indexed; elsewhere they attach to the message. The page handler skips
  // prevented drops, and the desktop's window-wide handler skips registered targets.
  const ignoreDrop = useCallback(() => {}, []);
  const { dragging, dropProps, nativeDropTarget } = useSourceDrop({
    onItems: onDropItems ?? ignoreDrop,
    disabledReason: dropDisabledReason,
  });
  const acceptsDrop = Boolean(onDropItems);
  return (
    <section
      ref={acceptsDrop ? nativeDropTarget : undefined}
      aria-label={title}
      className="unsloth-files-panel"
      data-dragging={acceptsDrop && dragging ? "true" : undefined}
      {...(acceptsDrop ? dropProps : {})}
    >
      {acceptsDrop ? (
        <div className="unsloth-files-panel-drop-overlay" aria-hidden={true}>
          <PlusIcon className="size-5" />
          Drop to chat with these files
        </div>
      ) : null}
      <div className="flex min-h-8 items-center gap-2 pl-1">
        <HugeiconsIcon
          icon={icon}
          strokeWidth={1.75}
          className="size-4 shrink-0 text-muted-foreground"
        />
        <h2 className="min-w-0 truncate text-ui-14 font-medium text-foreground">
          {title}
          {titleSuffix ? (
            <span className="font-normal text-muted-foreground"> {titleSuffix}</span>
          ) : null}
        </h2>
        {note}
        <div className="ml-auto flex shrink-0 items-center gap-1">
          {headerControls}
          {overflowing ? (
            <>
              <button
                type="button"
                className="unsloth-files-panel-icon-btn"
                onClick={() => page(-1)}
                disabled={!scroll.back}
                aria-label="Scroll files back"
              >
                <ChevronLeftIcon className="size-4" />
              </button>
              <button
                type="button"
                className="unsloth-files-panel-icon-btn"
                onClick={() => page(1)}
                disabled={!scroll.forward}
                aria-label="Scroll files forward"
              >
                <ChevronRightIcon className="size-4" />
              </button>
            </>
          ) : null}
          {onClose ? (
            <button
              type="button"
              className="unsloth-files-panel-icon-btn"
              onClick={onClose}
              aria-label={closeLabel ?? "Close"}
              title={closeLabel}
            >
              <XIcon className="size-4" />
            </button>
          ) : null}
        </div>
      </div>
      {children ? (
        <div
          ref={stripRef}
          onScroll={measure}
          className="unsloth-files-panel-strip"
        >
          {children}
        </div>
      ) : null}
    </section>
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
          className="composer-pill-btn attachment-target-pill shrink-0 gap-1 !text-foreground/70 pl-3 pr-1.5"
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
        align="end"
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
  const ragToolDisabled = useRagToolDisabled();
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

  // materialize locally; append() reuses the id without a mid-upload remount.
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
    // A plain send creates the chat too, without ensureThreadId. Hand the
    // earlier choice to the chat that just got an id, or the next one inherits it.
    if (!hadThreadId) {
      useChatRuntimeStore.getState().adoptPendingProjectAttachmentTarget(threadId);
    }
    setMaterializedId(null);
    initGenerationRef.current += 1;
    initPromiseRef.current = null;
  }, [threadId]);

  // An abandoned composer leaves its choice under the pending key, where the next new chat would
  // claim it. Adoption removes the key, so this drops only what nobody claimed.
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

  // Tell the composer whether any doc is still indexing, so it can hold a queued send until
  // retrieval covers them (Composer.enqueueSend). For KB / RAG-off scope is null, so both lists are
  // empty and this reads false. From the hooks, not the rows: work started in the Sources panel is
  // in flight before either instance has a row for it, and a job already running on a reopened
  // project arrives with the first list, so hold until that lands. Both scopes hold on their first
  // list, for the same reason: reopening a chat whose own attachment was still indexing lists
  // nothing until it lands either.
  const hasIndexing =
    threadIndexing || threadListLoading || projectIndexing || projectListLoading;
  useEffect(() => {
    onIndexingChange?.(hasIndexing);
  }, [hasIndexing, onIndexingChange]);
  useEffect(() => () => onIndexingChange?.(false), [onIndexingChange]);

  // Materialize the thread id on first use; ref-deduped so a double-click can't
  // start two threads. A thread switch gets separate work even if the prior request is pending.
  const ensureThreadId = useCallback((): Promise<string> => {
    // A new chat already has a local id before initialize() creates its stored row.
    // Only initialize when that id belongs to the current uninitialized item. During
    // navigation the saved target reaches this bar before switchToThread replaces the
    // outgoing item; initializing then would create and attach to the wrong chat.
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
      // Filter duplicates before initializing the chat.
      void upload(items, async () => ({
        type: "thread",
        threadId: await ensureThreadId(),
      }));
    },
    [ensureThreadId, projectId, sharesWithProject, upload, uploadToProject],
  );

  // Only files added or dropped here are indexed; composer attachments stay with the message.
  const attachIndexable = useCallback(
    (files: File[]) => {
      const indexable = files.filter(isRagIndexable);
      const skipped = files.length - indexable.length;
      if (skipped > 0) {
        toast.error(
          skipped === 1
            ? `"${files.find((file) => !isRagIndexable(file))?.name}" can't be searched`
            : `${skipped} files can't be searched`,
          { description: SUPPORTED_SOURCES_HINT },
        );
      }
      if (indexable.length > 0) attach(indexable);
    },
    [attach],
  );
  const pickFiles = useCallback(() => {
    openFilePicker(RAG_SOURCE_UPLOAD_ACCEPT, attachIndexable);
  }, [attachIndexable]);

  // Desktop drops land in the native-intent store because the drop listener lives on
  // the chat page; only the chat that received the OS drop may drain its batch.
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
    // Hold the batch rather than draining it before the chat's project scope is known.
    if (projectUnresolved) {
      return;
    }
    const store = useNativeIntentStore.getState();
    const intents = store.takeAttachments(nativeAttachmentTargetKey);
    if (intents.length === 0) {
      return;
    }
    // A KB-scoped chat uploads through the KB dialog, so a thread upload here would
    // index into something this bar never shows.
    if (ragEnabled && ragSource.type === "kb") {
      const kbId = ragSource.kbId;
      const files =
        intents.length === 1
          ? `"${intents[0].displayLabel}"`
          : `${intents.length} files`;
      const target = activeKbName
        ? `"${activeKbName}"`
        : "this chat's knowledge base";
      // Nothing else holds these files: keep the offer while their path tokens are readable.
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
    // A stale KB preference is inactive while RAG is off; use thread retrieval.
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

  // Removing a project source here deletes it for every chat, beside a chat card
  // whose X is undoable. Confirm, as the Sources tab and Settings do.
  const [removingShared, setRemovingShared] = useState<RagDocument | null>(null);

  const projectGroups = useMemo(
    () => groupByLinkedFolder(projectDocuments),
    [projectDocuments],
  );
  const folderNames = useLinkedFolderNames(
    projectId,
    projectGroups.folders.map(([id]) => id).join(","),
  );
  const folderCards = projectGroups.folders.map(([folderId, docs]) => (
    <div
      key={`folder:${folderId}`}
      className={cn("contents", ragToolDisabled && "[&>*]:opacity-50")}
    >
      <LinkedFolderCard name={folderNames.get(folderId)} docs={docs} />
    </div>
  ));
  const fileCount = documents.length + projectDocuments.length;
  const countSuffix = fileCount === 1 ? "1 file" : `${fileCount} files`;
  // Every branch, always the first fragment child: an "Add" must open after the source
  // moves, and deleting the active KB in it must not remount it.
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

  // A KB source uploads via the KB dialog, not here; show which KB is active.
  if (ragEnabled && ragSource.type === "kb") {
    const kbId = ragSource.kbId;
    return (
      <>
        {kbDialog}
        {/* Keyed per mode: the strip's observers attach on mount, and this one has no strip. */}
        <ChatFilesPanel
          key="kb"
          icon={FileDatabaseIcon}
          title={activeKbName ? `Knowledge base: ${activeKbName}` : "Knowledge base"}
          onClose={() => setRagEnabled(false)}
          closeLabel="Stop chatting with files"
          headerControls={
            <button
              ref={kbChipRef}
              type="button"
              onClick={() => setKbDialogFocus({ kbId })}
              className="unsloth-files-panel-action"
            >
              Manage files
            </button>
          }
        />
      </>
    );
  }
  // project sources stay visible with Docs off because retrieval uses them; thread scope is inert.
  if (!ragEnabled) {
    return (
      <>
        {kbDialog}
        {projectDocuments.length > 0 ? (
          <ChatFilesPanel
            key="project"
            icon={FolderAttachmentIcon}
            title={ragToolDisabled ? "Project sources not used" : "Project sources"}
            titleSuffix={countSuffix}
            note={
              <span
                className="hidden truncate text-ui-12 text-muted-foreground @[32rem]/files-panel:inline"
                title={
                  ragToolDisabled
                    ? "The selected model can't search documents, so this chat doesn't use its project's sources. Pick a model with tool support to use them."
                    : "This chat retrieves from its project's sources. Manage them in the project's Sources tab."
                }
              >
                {ragToolDisabled
                  ? "The selected model can't search documents"
                  : "This chat retrieves from its project's sources"}
              </span>
            }
          >
            {folderCards}
            {projectGroups.loose.map((doc) => (
              <div
                key={`inherited:${doc.id}`}
                className={cn("contents", ragToolDisabled && "[&>*]:opacity-50")}
              >
                <DocumentCard doc={doc} shared={true} />
              </div>
            ))}
          </ChatFilesPanel>
        ) : null}
      </>
    );
  }

  // block attachments until the chat's project is known to avoid guessing their scope.
  const busy = uploading || projectUploading || projectUnresolved;
  // Still claim a drop while busy, or it falls through and attaches to the message instead.
  const busyReason = !busy
    ? undefined
    : projectUnresolved
      ? "Still loading this chat's project. Try again in a moment."
      : "Still uploading. Drop the files again once it finishes.";

  return (
    <>
      {kbDialog}
      <ChatFilesPanel
        key="files"
        icon={FileDatabaseIcon}
        title="Chat with files"
        titleSuffix={fileCount > 0 ? `(RAG) · ${countSuffix}` : "(RAG)"}
        onDropItems={attach}
        dropDisabledReason={busyReason}
        onClose={() => setRagEnabled(false)}
        closeLabel="Stop chatting with files"
        note={
          ragToolDisabled && fileCount > 0 ? (
            <span
              className="shrink-0 rounded-full bg-muted px-2 py-0.5 text-ui-11 text-muted-foreground"
              title="The selected model can't search documents, so these files aren't used. Pick a model with tool support to use them."
            >
              Not used
            </span>
          ) : null
        }
        headerControls={
          <>
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
            <button
              type="button"
              disabled={busy}
              onClick={pickFiles}
              className="unsloth-files-panel-action"
              title={
                busyReason ??
                (sharesWithProject
                  ? "Add files for retrieval, shared with every chat in this project"
                  : "Add files for retrieval in this chat")
              }
            >
              Add files
            </button>
          </>
        }
      >
        {/* project sources precede thread sources because they outlive the chat. */}
        {folderCards}
        {projectGroups.loose.map((doc) => (
          <div
            key={`project:${doc.id}`}
            className={cn("contents", ragToolDisabled && "[&>*]:opacity-50")}
          >
            <DocumentCard
              doc={doc}
              shared={true}
              onRemove={
                doc.id.startsWith("pending_") || isLinkedFolderManaged(doc)
                  ? undefined
                  : () => setRemovingShared(doc)
              }
            />
          </div>
        ))}
        {documents.map((doc) => (
          <div
            key={doc.id}
            className={cn("contents", ragToolDisabled && "[&>*]:opacity-50")}
          >
            <DocumentCard
              doc={doc}
              shared={false}
              onRemove={
                doc.id.startsWith("pending_")
                  ? undefined
                  : () => void remove(doc.id)
              }
            />
          </div>
        ))}
        <AddFilesCard disabled={busy} onClick={pickFiles} />
        {fileCount === 0 ? (
          <p className="flex max-w-[calc(18rem*var(--ui-space-scale,1))] shrink-0 items-center text-ui-12 leading-snug text-muted-foreground">
            Add or drop documents, spreadsheets, slides or code here. The model
            searches them and cites what it uses.
          </p>
        ) : null}
      </ChatFilesPanel>
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
    </>
  );
}
