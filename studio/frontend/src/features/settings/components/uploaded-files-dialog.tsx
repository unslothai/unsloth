// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

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
import { Spinner } from "@/components/ui/spinner";
import {
  type ChatAttachmentRecord,
  deleteChatAttachment,
  emitChatAttachmentDeleted,
  fetchChatAttachmentBlob,
  listChatAttachments,
} from "@/features/chat";
import {
  deleteDocument,
  getDocumentFileUrl,
  listAllDocuments,
  type UploadedDocument,
} from "@/features/rag";
import { toast } from "@/lib/toast";
import {
  ArrowUpRight01Icon,
  Delete02Icon,
  File02Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useNavigate } from "@tanstack/react-router";
import { type ReactNode, useEffect, useRef, useState } from "react";
import { useSettingsDialogStore } from "../stores/settings-dialog-store";

function formatUploadedAt(value: string | number | null | undefined): string {
  if (value === null || value === undefined || value === "") return "-";
  // Chat attachments carry ms epoch numbers; RAG documents carry SQLite
  // ISO-ish strings (no timezone). Unparseable strings fall through raw.
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return String(value);
  return parsed.toLocaleDateString(undefined, {
    year: "numeric",
    month: "long",
    day: "numeric",
  });
}

function formatSize(bytes: number | null | undefined): string {
  if (bytes === null || bytes === undefined) return "-";
  if (bytes < 1024) return `${bytes} B`;
  const units = ["KB", "MB", "GB"];
  let value = bytes;
  let unit = "B";
  for (const next of units) {
    if (value < 1024) break;
    value /= 1024;
    unit = next;
  }
  return `${value >= 10 ? Math.round(value) : value.toFixed(1)} ${unit}`;
}

function ragLocationLabel(doc: UploadedDocument): string {
  if (doc.kbId) return doc.kbName ? `KB · ${doc.kbName}` : "Knowledge base";
  if (doc.projectId) {
    return doc.projectName ? `Project · ${doc.projectName}` : "Project";
  }
  if (doc.threadId) return "Chat files (RAG)";
  return "-";
}

/** Short uppercase file-type label from the filename extension, falling back
 *  to the content-type subtype (e.g. "image/webp" gives WEBP). */
function fileTypeLabel(
  name: string,
  contentType?: string | null,
): string | null {
  const dot = name.lastIndexOf(".");
  const ext = dot > 0 ? name.slice(dot + 1).trim() : "";
  if (ext && ext.length <= 5) return ext.toUpperCase();
  const subtype = contentType?.split("/")[1]?.split("+")[0]?.trim();
  return subtype && subtype.length <= 10 ? subtype.toUpperCase() : null;
}

/** Lazy image thumbnail for a chat attachment; a file icon until it loads.
 *  The stored blob only downloads once the row scrolls into view, so a long
 *  history of screenshots does not fetch every image on open. */
function ChatImageThumb({
  messageId,
  attachmentId,
}: {
  messageId: string;
  attachmentId: string;
}) {
  const [src, setSrc] = useState<string | null>(null);
  const [visible, setVisible] = useState(false);
  const holderRef = useRef<HTMLSpanElement>(null);

  useEffect(() => {
    const el = holderRef.current;
    if (!el) return;
    if (typeof IntersectionObserver === "undefined") {
      return;
    }
    const observer = new IntersectionObserver((entries) => {
      if (entries.some((entry) => entry.isIntersecting)) {
        setVisible(true);
        observer.disconnect();
      }
    });
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    if (!visible) return;
    let cancelled = false;
    let url: string | null = null;
    fetchChatAttachmentBlob(messageId, attachmentId)
      .then((blob) => {
        if (cancelled) return;
        url = URL.createObjectURL(blob);
        setSrc(url);
      })
      .catch(() => {
        // Keep the file icon on failure.
      });
    return () => {
      cancelled = true;
      if (url) URL.revokeObjectURL(url);
    };
  }, [visible, messageId, attachmentId]);

  if (!src) {
    return (
      <span
        ref={holderRef}
        className="flex h-full w-full items-center justify-center"
      >
        <FileIconThumb />
      </span>
    );
  }
  return <img src={src} alt="" className="h-full w-full object-cover" />;
}

function FileIconThumb() {
  return (
    <HugeiconsIcon
      icon={File02Icon}
      strokeWidth={1.75}
      className="size-4 text-muted-foreground"
    />
  );
}

/** One display row: a RAG document or a chat message attachment. */
interface UploadedFileRow {
  key: string;
  source: "rag" | "chat";
  name: string;
  location: string;
  sizeBytes?: number | null;
  createdAt?: string | number | null;
  failed?: boolean;
  /** Epoch ms for sorting; rows with unknown dates sort last. */
  sortTime: number;
  typeLabel: string | null;
  /** Image rows render a thumbnail; others show a file icon. */
  thumb: ReactNode;
  /** Chat rows link back to their thread. */
  threadId?: string | null;
  /** Compare-chat rows navigate by pair id instead of opening one pane alone. */
  pairId?: string | null;
  open: () => Promise<void>;
  remove: () => Promise<void>;
  deleteDescription: string;
}

function toSortTime(value: string | number | null | undefined): number {
  if (value === null || value === undefined || value === "") return 0;
  const parsed = new Date(value).getTime();
  return Number.isNaN(parsed) ? 0 : parsed;
}

// Safari and Firefox block window.open after an await (the user gesture is
// gone), so open a blank tab synchronously and point it at the URL once
// resolved. A blocked synchronous open is surfaced instead of silently losing
// the file after the asynchronous URL lookup.
async function openResolvedUrl(resolve: () => Promise<string>): Promise<void> {
  const win = window.open("", "_blank");
  if (!win) {
    throw new Error(
      "Your browser blocked the new tab. Allow popups and retry.",
    );
  }
  win.opener = null;
  let url: string;
  try {
    url = await resolve();
  } catch (err) {
    win.close();
    throw err;
  }
  win.location.replace(url);
}

function ragRow(doc: UploadedDocument): UploadedFileRow {
  return {
    key: `rag-${doc.id}`,
    source: "rag",
    name: doc.filename,
    location: ragLocationLabel(doc),
    sizeBytes: doc.sizeBytes,
    createdAt: doc.createdAt,
    failed: doc.status === "failed",
    sortTime: toSortTime(doc.createdAt),
    typeLabel: fileTypeLabel(doc.filename),
    // RAG uploads are documents (pdf, txt, md, docx, html), not images.
    thumb: <FileIconThumb />,
    open: () => openResolvedUrl(() => getDocumentFileUrl(doc.id)),
    remove: async () => {
      await deleteDocument(doc.id, doc.projectId);
    },
    deleteDescription:
      "The file and its indexed content are removed. This cannot be undone.",
  };
}

function chatAttachmentRow(att: ChatAttachmentRecord): UploadedFileRow {
  const isImage =
    att.type === "image" || Boolean(att.contentType?.startsWith("image/"));
  return {
    key: `chat-${att.messageId}-${att.id}`,
    source: "chat",
    name: att.name,
    location: att.threadTitle ? `Chat · ${att.threadTitle}` : "Chat",
    sizeBytes: att.sizeBytes,
    createdAt: att.createdAt,
    sortTime: toSortTime(att.createdAt),
    typeLabel: fileTypeLabel(att.name, att.contentType),
    threadId: att.threadId,
    pairId: att.pairId,
    thumb: isImage ? (
      <ChatImageThumb messageId={att.messageId} attachmentId={att.id} />
    ) : (
      <FileIconThumb />
    ),
    open: () =>
      openResolvedUrl(async () => {
        const blob = await fetchChatAttachmentBlob(att.messageId, att.id);
        const url = URL.createObjectURL(blob);
        // Give the new tab time to load the blob before revoking.
        setTimeout(() => URL.revokeObjectURL(url), 60_000);
        return url;
      }),
    remove: async () => {
      await deleteChatAttachment(att.messageId, att.id);
      // Patch any loaded runtime copy so a later repo sync cannot write the
      // deleted attachment back to storage.
      emitChatAttachmentDeleted({
        messageId: att.messageId,
        attachmentId: att.id,
      });
    },
    deleteDescription:
      "The attachment is removed from its chat message; the message text is kept. This cannot be undone.",
  };
}

type SourceLoad<T> = {
  status: "loading" | "ready" | "error";
  data: T;
  error: string | null;
};

function errorMessage(error: unknown, fallback: string): string {
  return error instanceof Error ? error.message : fallback;
}

/** Inline settings page listing uploaded files from each available source. */
export function UploadedFilesView() {
  const [ragFiles, setRagFiles] = useState<SourceLoad<UploadedDocument[]>>({
    status: "loading",
    data: [],
    error: null,
  });
  const [chatFiles, setChatFiles] = useState<
    SourceLoad<ChatAttachmentRecord[]>
  >({ status: "loading", data: [], error: null });
  const [chatNextOffset, setChatNextOffset] = useState<number | null>(null);
  const [loadingMore, setLoadingMore] = useState(false);
  const [confirmingDelete, setConfirmingDelete] =
    useState<UploadedFileRow | null>(null);
  const navigate = useNavigate();

  const rows = [
    ...ragFiles.data.map(ragRow),
    ...chatFiles.data.map(chatAttachmentRow),
  ].sort((a, b) => b.sortTime - a.sortTime);

  // Jump to the chat thread the attachment lives in, closing the settings
  // dialog so the thread is actually visible.
  function goToChat(row: UploadedFileRow) {
    if (!row.threadId) return;
    useSettingsDialogStore.getState().closeDialog();
    if (row.pairId) {
      void navigate({ to: "/chat", search: { compare: row.pairId } });
    } else {
      void navigate({ to: "/chat", search: { thread: row.threadId } });
    }
  }

  useEffect(() => {
    let cancelled = false;
    void listAllDocuments().then(
      (data) => {
        if (!cancelled) setRagFiles({ status: "ready", data, error: null });
      },
      (error: unknown) => {
        if (!cancelled) {
          setRagFiles({
            status: "error",
            data: [],
            error: errorMessage(error, "Failed to load RAG documents"),
          });
        }
      },
    );
    void listChatAttachments().then(
      (page) => {
        if (!cancelled) {
          setChatFiles({
            status: "ready",
            data: page.attachments,
            error: null,
          });
          setChatNextOffset(page.nextOffset);
        }
      },
      (error: unknown) => {
        if (!cancelled) {
          setChatFiles({
            status: "error",
            data: [],
            error: errorMessage(error, "Failed to load chat attachments"),
          });
        }
      },
    );
    return () => {
      cancelled = true;
    };
  }, []);

  function retryRagFiles() {
    setRagFiles((current) => ({ ...current, status: "loading", error: null }));
    void listAllDocuments().then(
      (data) => setRagFiles({ status: "ready", data, error: null }),
      (error: unknown) =>
        setRagFiles((current) => ({
          ...current,
          status: "error",
          error: errorMessage(error, "Failed to load RAG documents"),
        })),
    );
  }

  async function loadChatPage(offset: number, append: boolean) {
    setLoadingMore(true);
    setChatFiles((current) => ({ ...current, status: "loading", error: null }));
    try {
      const page = await listChatAttachments(offset);
      setChatFiles((current) => ({
        status: "ready",
        data: append
          ? [
              ...current.data,
              ...page.attachments.filter(
                (incoming) =>
                  !current.data.some(
                    (existing) =>
                      existing.id === incoming.id &&
                      existing.messageId === incoming.messageId,
                  ),
              ),
            ]
          : page.attachments,
        error: null,
      }));
      setChatNextOffset(page.nextOffset);
    } catch (error) {
      setChatFiles((current) => ({
        ...current,
        status: "error",
        error: errorMessage(error, "Failed to load chat attachments"),
      }));
    } finally {
      setLoadingMore(false);
    }
  }

  function retryChatFiles() {
    const append = chatFiles.data.length > 0 && chatNextOffset !== null;
    void loadChatPage(append ? chatNextOffset : 0, append);
  }

  async function handleOpen(row: UploadedFileRow) {
    try {
      await row.open();
    } catch (err) {
      toast.error("Failed to open file", {
        description: err instanceof Error ? err.message : undefined,
      });
    }
  }

  async function handleDelete(row: UploadedFileRow) {
    // Offset pages and destructive mutations must not race: a deletion shifts
    // the boundary used by an in-flight page request.
    if (loadingMore) return;
    try {
      await row.remove();
      if (row.source === "rag") {
        setRagFiles((current) => ({
          ...current,
          data: current.data.filter((doc) => `rag-${doc.id}` !== row.key),
        }));
      } else {
        setChatFiles((current) => ({
          ...current,
          data: current.data.filter(
            (attachment) =>
              `chat-${attachment.messageId}-${attachment.id}` !== row.key,
          ),
        }));
        // Offset pagination is relative to the current server inventory. A
        // deletion before the next page shifts every later row back by one.
        setChatNextOffset((current) =>
          current === null ? null : Math.max(0, current - 1),
        );
      }
      toast.success("File deleted");
    } catch (err) {
      toast.error("Failed to delete file", {
        description: err instanceof Error ? err.message : undefined,
      });
    }
  }

  return (
    <div className="flex flex-col gap-4">
      {ragFiles.status === "error" ? (
        <div className="flex flex-wrap items-center justify-between gap-2 rounded-md border border-destructive/30 bg-destructive/5 px-3 py-2 text-sm">
          <span>RAG documents unavailable: {ragFiles.error}</span>
          <button
            type="button"
            onClick={retryRagFiles}
            className="font-medium underline underline-offset-2"
          >
            Retry
          </button>
        </div>
      ) : null}
      {chatFiles.status === "error" ? (
        <div className="flex flex-wrap items-center justify-between gap-2 rounded-md border border-destructive/30 bg-destructive/5 px-3 py-2 text-sm">
          <span>Chat attachments unavailable: {chatFiles.error}</span>
          <button
            type="button"
            onClick={retryChatFiles}
            className="font-medium underline underline-offset-2"
          >
            Retry
          </button>
        </div>
      ) : null}

      {rows.length === 0 &&
      (ragFiles.status === "loading" || chatFiles.status === "loading") ? (
        <div className="flex justify-center py-8">
          <Spinner className="size-5 text-muted-foreground" />
        </div>
      ) : rows.length === 0 &&
        ragFiles.status !== "error" &&
        chatFiles.status !== "error" ? (
        <p className="py-8 text-center text-sm text-muted-foreground">
          No uploaded files.
        </p>
      ) : rows.length > 0 ? (
        <div>
          <div className="hidden items-center gap-3 border-b border-border/60 px-1 pb-2 text-xs font-semibold text-foreground sm:flex">
            <span className="flex-1">Name</span>
            <span className="w-36 shrink-0">Location</span>
            <span className="w-24 shrink-0">Uploaded</span>
            <span className="w-16 shrink-0" />
          </div>
          {rows.map((row) => (
            <div
              key={row.key}
              className="group flex flex-wrap items-center gap-x-3 gap-y-1 border-b border-border/40 px-1 py-2.5 text-sm last:border-0 sm:flex-nowrap"
            >
              {/* Clicking the file jumps to its chat; files without one
                open directly. The theme scales rounded-md up to a near
                circle at this size, so the thumb pins a small radius. */}
              <button
                type="button"
                onClick={() =>
                  row.threadId ? goToChat(row) : void handleOpen(row)
                }
                title={
                  row.threadId ? `Go to ${row.location}` : `Open ${row.name}`
                }
                className="group/name flex min-w-0 flex-1 basis-[calc(100%-80px)] items-center gap-2.5 overflow-hidden text-left sm:basis-auto"
              >
                <span className="flex size-8 shrink-0 items-center justify-center overflow-hidden rounded-[7px] border border-border/50 bg-muted/40">
                  {row.thumb}
                </span>
                <span className="flex min-w-0 flex-1 flex-col">
                  <span className="flex min-w-0 items-center gap-2">
                    {/* Floor keeps the name visible when the chip and fixed
                      columns squeeze the cell at narrow widths. */}
                    <span className="min-w-[56px] truncate underline-offset-2 group-hover/name:underline">
                      {row.name}
                    </span>
                    {row.typeLabel ? (
                      <span className="shrink-0 rounded-md bg-black/[0.06] px-1.5 py-px text-[0.5625rem] font-medium uppercase tracking-wide text-muted-foreground dark:bg-white/[0.1]">
                        {row.typeLabel}
                      </span>
                    ) : null}
                    {row.failed ? (
                      <span className="shrink-0 text-xs text-destructive">
                        failed
                      </span>
                    ) : null}
                  </span>
                  <span className="text-xs text-muted-foreground tabular-nums">
                    {formatSize(row.sizeBytes)}
                  </span>
                </span>
              </button>
              {row.threadId ? (
                <button
                  type="button"
                  onClick={() => goToChat(row)}
                  title={`Go to ${row.location}`}
                  className="order-3 w-full truncate pl-10 text-left text-muted-foreground underline-offset-2 transition-colors hover:text-foreground hover:underline sm:order-none sm:w-36 sm:pl-0"
                >
                  {row.location}
                </button>
              ) : (
                <span
                  className="order-3 w-full truncate pl-10 text-muted-foreground sm:order-none sm:w-36 sm:pl-0"
                  title={row.location}
                >
                  {row.location}
                </span>
              )}
              <span className="order-4 w-full pl-10 text-muted-foreground tabular-nums sm:order-none sm:w-24 sm:pl-0">
                {formatUploadedAt(row.createdAt)}
              </span>
              <span className="flex w-16 shrink-0 items-center justify-end gap-1">
                <button
                  type="button"
                  onClick={() => void handleOpen(row)}
                  aria-label={`Open ${row.name}`}
                  title="Open"
                  className="inline-flex size-7 items-center justify-center rounded-full text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
                >
                  <HugeiconsIcon
                    icon={ArrowUpRight01Icon}
                    strokeWidth={1.75}
                    className="size-4"
                  />
                </button>
                <button
                  type="button"
                  disabled={loadingMore}
                  onClick={() => setConfirmingDelete(row)}
                  aria-label={`Delete ${row.name}`}
                  title="Delete"
                  className="inline-flex size-7 items-center justify-center rounded-full text-muted-foreground transition-colors hover:bg-destructive/10 hover:text-destructive disabled:cursor-wait disabled:opacity-50"
                >
                  <HugeiconsIcon
                    icon={Delete02Icon}
                    strokeWidth={1.75}
                    className="size-4"
                  />
                </button>
              </span>
            </div>
          ))}
          {chatNextOffset !== null ? (
            <div className="flex justify-center pt-3">
              <button
                type="button"
                disabled={loadingMore}
                onClick={() => void loadChatPage(chatNextOffset, true)}
                className="rounded-md border border-border px-3 py-1.5 text-sm font-medium hover:bg-muted disabled:cursor-wait disabled:opacity-60"
              >
                {loadingMore ? "Loading..." : "Load more chat attachments"}
              </button>
            </div>
          ) : null}
        </div>
      ) : null}

      <AlertDialog
        open={confirmingDelete !== null}
        onOpenChange={(o) => {
          if (!o) setConfirmingDelete(null);
        }}
      >
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete file</AlertDialogTitle>
            <AlertDialogDescription>
              Delete{" "}
              <span className="font-medium text-foreground">
                &quot;{confirmingDelete?.name}&quot;
              </span>
              ? {confirmingDelete?.deleteDescription}
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction
              variant="destructive"
              onClick={() => {
                const row = confirmingDelete;
                setConfirmingDelete(null);
                if (row) void handleDelete(row);
              }}
            >
              Delete
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
}
