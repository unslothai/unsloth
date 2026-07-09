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
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Spinner } from "@/components/ui/spinner";
import {
  type ChatAttachmentRecord,
  deleteChatAttachment,
  fetchChatAttachmentBlob,
  listChatAttachments,
} from "@/features/chat/api/chat-api";
import {
  deleteDocument,
  getDocumentFileUrl,
  listAllDocuments,
} from "@/features/rag/api/rag-api";
import type { UploadedDocument } from "@/features/rag/types/rag";
import { toast } from "@/lib/toast";
import { ArrowUpRight01Icon, Delete02Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useEffect, useState } from "react";

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

/** One display row: a RAG document or a chat message attachment. */
interface UploadedFileRow {
  key: string;
  name: string;
  location: string;
  sizeBytes?: number | null;
  createdAt?: string | number | null;
  failed?: boolean;
  /** Epoch ms for sorting; rows with unknown dates sort last. */
  sortTime: number;
  open: () => Promise<void>;
  remove: () => Promise<void>;
  deleteDescription: string;
}

function toSortTime(value: string | number | null | undefined): number {
  if (value === null || value === undefined || value === "") return 0;
  const parsed = new Date(value).getTime();
  return Number.isNaN(parsed) ? 0 : parsed;
}

function ragRow(doc: UploadedDocument): UploadedFileRow {
  return {
    key: `rag-${doc.id}`,
    name: doc.filename,
    location: ragLocationLabel(doc),
    sizeBytes: doc.sizeBytes,
    createdAt: doc.createdAt,
    failed: doc.status === "failed",
    sortTime: toSortTime(doc.createdAt),
    open: async () => {
      const url = await getDocumentFileUrl(doc.id);
      window.open(url, "_blank", "noopener");
    },
    remove: async () => {
      await deleteDocument(doc.id);
    },
    deleteDescription:
      "The file and its indexed content are removed. This cannot be undone.",
  };
}

function chatAttachmentRow(att: ChatAttachmentRecord): UploadedFileRow {
  return {
    key: `chat-${att.messageId}-${att.id}`,
    name: att.name,
    location: att.threadTitle ? `Chat · ${att.threadTitle}` : "Chat",
    sizeBytes: att.sizeBytes,
    createdAt: att.createdAt,
    sortTime: toSortTime(att.createdAt),
    open: async () => {
      const blob = await fetchChatAttachmentBlob(att.messageId, att.id);
      const url = URL.createObjectURL(blob);
      window.open(url, "_blank", "noopener");
      // Give the new tab time to load the blob before revoking.
      setTimeout(() => URL.revokeObjectURL(url), 60_000);
    },
    remove: async () => {
      await deleteChatAttachment(att.messageId, att.id);
    },
    deleteDescription:
      "The attachment is removed from its chat message; the message text is kept. This cannot be undone.",
  };
}

export function UploadedFilesDialog({
  open,
  onOpenChange,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}) {
  const [rows, setRows] = useState<UploadedFileRow[] | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [confirmingDelete, setConfirmingDelete] =
    useState<UploadedFileRow | null>(null);

  useEffect(() => {
    if (!open) return;
    let cancelled = false;
    setLoadError(null);
    // Load both sources independently: RAG being unavailable (no sqlite-vec)
    // must not hide chat attachments, and vice versa.
    void Promise.allSettled([listAllDocuments(), listChatAttachments()]).then(
      ([ragResult, chatResult]) => {
        if (cancelled) return;
        const next: UploadedFileRow[] = [];
        if (ragResult.status === "fulfilled") {
          next.push(...ragResult.value.map(ragRow));
        }
        if (chatResult.status === "fulfilled") {
          next.push(...chatResult.value.map(chatAttachmentRow));
        }
        next.sort((a, b) => b.sortTime - a.sortTime);
        setRows(next);
        if (
          ragResult.status === "rejected" &&
          chatResult.status === "rejected"
        ) {
          const reason = ragResult.reason;
          setLoadError(
            reason instanceof Error
              ? reason.message
              : "Failed to load uploaded files",
          );
        }
      },
    );
    return () => {
      cancelled = true;
    };
  }, [open]);

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
    try {
      await row.remove();
      setRows((prev) => prev?.filter((r) => r.key !== row.key) ?? prev ?? null);
      toast.success("File deleted");
    } catch (err) {
      toast.error("Failed to delete file", {
        description: err instanceof Error ? err.message : undefined,
      });
    }
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-2xl">
        <DialogHeader>
          <DialogTitle>Uploaded files</DialogTitle>
        </DialogHeader>

        {rows === null ? (
          <div className="flex justify-center py-8">
            <Spinner className="size-5 text-muted-foreground" />
          </div>
        ) : loadError ? (
          <p className="py-8 text-center text-sm text-muted-foreground">
            {loadError}
          </p>
        ) : rows.length === 0 ? (
          <p className="py-8 text-center text-sm text-muted-foreground">
            No uploaded files.
          </p>
        ) : (
          <div className="max-h-[60vh] overflow-y-auto">
            <div className="flex items-center gap-4 border-b border-border/60 px-1 pb-2 text-xs font-semibold text-foreground">
              <span className="flex-1">Name</span>
              <span className="w-32 shrink-0">Location</span>
              <span className="w-16 shrink-0 text-right">Size</span>
              <span className="w-28 shrink-0">Uploaded</span>
              <span className="w-16 shrink-0" />
            </div>
            {rows.map((row) => (
              <div
                key={row.key}
                className="group flex items-center gap-4 border-b border-border/40 px-1 py-2.5 text-sm last:border-0"
              >
                <span className="min-w-0 flex-1 truncate" title={row.name}>
                  {row.name}
                  {row.failed ? (
                    <span className="ml-2 text-xs text-destructive">
                      failed
                    </span>
                  ) : null}
                </span>
                <span
                  className="w-32 shrink-0 truncate text-muted-foreground"
                  title={row.location}
                >
                  {row.location}
                </span>
                <span className="w-16 shrink-0 text-right text-muted-foreground tabular-nums">
                  {formatSize(row.sizeBytes)}
                </span>
                <span className="w-28 shrink-0 text-muted-foreground tabular-nums">
                  {formatUploadedAt(row.createdAt)}
                </span>
                <span className="flex w-16 shrink-0 items-center justify-end gap-1">
                  <button
                    type="button"
                    onClick={() => void handleOpen(row)}
                    aria-label="Open file"
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
                    onClick={() => setConfirmingDelete(row)}
                    aria-label="Delete file"
                    title="Delete"
                    className="inline-flex size-7 items-center justify-center rounded-full text-muted-foreground transition-colors hover:bg-destructive/10 hover:text-destructive"
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
          </div>
        )}
      </DialogContent>

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
    </Dialog>
  );
}
