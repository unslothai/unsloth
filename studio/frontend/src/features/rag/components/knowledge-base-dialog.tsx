// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  Delete02Icon,
  Edit03Icon,
  PlusSignIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { ChevronLeftIcon, ChevronRightIcon, UploadIcon } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";

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
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Spinner } from "@/components/ui/spinner";
import { Textarea } from "@/components/ui/textarea";
import { toast } from "@/lib/toast";
import { cn } from "@/lib/utils";

import {
  createKnowledgeBase,
  deleteKnowledgeBase,
  listKnowledgeBaseDocuments,
  listKnowledgeBases,
  updateKnowledgeBase,
} from "../api/rag-api";
import { useRagAvailabilityStore } from "../api/rag-availability";
import { type KnowledgeBase, isLinkedFolderManaged } from "../types/rag";
import { DocumentStatusChip } from "./document-status-chip";
import { LinkedFoldersManager } from "./linked-folders-manager";
import { RAG_SOURCE_UPLOAD_ACCEPT } from "./source-drop-policy";
import { type RagUploadItem, useRagDocuments } from "./use-rag-documents";
import { useSourceDrop } from "./use-source-drop";

type View =
  | { kind: "list" }
  | { kind: "create" }
  | { kind: "edit"; kb: KnowledgeBase }
  | { kind: "documents"; kb: KnowledgeBase; uploads?: RagUploadItem[] };

export interface KnowledgeBaseFocus {
  kbId: string;
  uploads?: RagUploadItem[];
}

export interface KnowledgeBaseDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  focus?: KnowledgeBaseFocus | null;
  /** Without a Radix trigger, focus would land on the body on close. */
  onCloseAutoFocus?: (event: Event) => void;
}

export function KnowledgeBaseDialog({
  open,
  onOpenChange,
  focus = null,
  onCloseAutoFocus,
}: KnowledgeBaseDialogProps) {
  const [kbs, setKbs] = useState<KnowledgeBase[]>([]);
  const [loading, setLoading] = useState(false);
  const [view, setView] = useState<View>({ kind: "list" });
  // Handed-over files not yet uploading; nothing else holds them.
  const handoffRef = useRef<RagUploadItem[]>([]);
  const handedFocusRef = useRef<KnowledgeBaseFocus | null>(null);
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [saving, setSaving] = useState(false);
  const [confirmingDelete, setConfirmingDelete] = useState<KnowledgeBase | null>(
    null,
  );
  // Measured only: while the answer is unknown this stays false and the dialog renders
  // exactly as it always has. See api/rag-availability.
  const ragUnavailable = useRagAvailabilityStore((s) => s.isUnavailable());
  const ragUnavailableReason = useRagAvailabilityStore((s) =>
    s.unavailableReason(),
  );
  const ragUnavailableHint = ragUnavailable
    ? (ragUnavailableReason ?? undefined)
    : undefined;

  const refresh = useCallback(async (): Promise<KnowledgeBase[] | null> => {
    setLoading(true);
    try {
      const rows = await listKnowledgeBases();
      setKbs(rows);
      return rows;
    } catch (err) {
      toast.error("Failed to load knowledge bases", {
        description: err instanceof Error ? err.message : String(err),
      });
      return null;
    } finally {
      setLoading(false);
    }
  }, []);

  const wasOpenRef = useRef(false);
  useEffect(() => {
    if (!open) {
      handoffRef.current = [];
      handedFocusRef.current = null;
      wasOpenRef.current = false;
      return;
    }
    const wasOpen = wasOpenRef.current;
    wasOpenRef.current = true;
    // Once per handoff, though StrictMode runs this twice for the same one.
    if (focus !== handedFocusRef.current) {
      handedFocusRef.current = focus;
      handoffRef.current = [...handoffRef.current, ...(focus?.uploads ?? [])];
    }
    const uploads = handoffRef.current;
    let cancelled = false;
    // A handoff into the KB already on screen keeps its view mounted: unmounting it
    // abandons the rest of a batch still uploading there.
    setView((current) =>
      wasOpen &&
      focus &&
      current.kind === "documents" &&
      current.kb.id === focus.kbId
        ? {
            kind: "documents",
            kb: current.kb,
            uploads: uploads.length ? uploads : undefined,
          }
        : { kind: "list" },
    );
    void refresh().then((rows) => {
      if (cancelled || !focus) return;
      const kb = rows?.find((row) => row.id === focus.kbId);
      if (!kb) {
        // A failed load already said so.
        if (uploads.length && rows) {
          toast.error("Knowledge base not found", {
            description: "Open or create one below and the files go there.",
          });
        }
        return;
      }
      setView({
        kind: "documents",
        kb,
        uploads: uploads.length ? uploads : undefined,
      });
    });
    return () => {
      cancelled = true;
    };
  }, [open, focus, refresh]);

  function startCreate() {
    setName("");
    setDescription("");
    setView({ kind: "create" });
  }

  function startEdit(kb: KnowledgeBase) {
    setName(kb.name);
    setDescription(kb.description ?? "");
    setView({ kind: "edit", kb });
  }

  function backToList() {
    setView({ kind: "list" });
  }

  function openDocuments(kb: KnowledgeBase) {
    const uploads = handoffRef.current;
    setView({
      kind: "documents",
      kb,
      uploads: uploads.length ? uploads : undefined,
    });
  }

  // The view outlives a close, so the files leave it as soon as they start.
  const takeUploads = useCallback(() => {
    handoffRef.current = [];
    setView((current) =>
      current.kind === "documents" && current.uploads
        ? { kind: "documents", kb: current.kb }
        : current,
    );
  }, []);

  async function submitForm() {
    // The button is disabled for this, but the form is also reachable by keyboard and
    // the verdict can land while it is open. A 503 toast is not an explanation.
    if (ragUnavailable) {
      toast.error("Knowledge bases are unavailable", {
        description: ragUnavailableReason ?? undefined,
      });
      return;
    }
    const trimmed = name.trim();
    if (!trimmed) {
      toast.error("Name is required");
      return;
    }
    setSaving(true);
    try {
      let createdId: string | null = null;
      if (view.kind === "edit") {
        await updateKnowledgeBase(view.kb.id, {
          name: trimmed,
          description: description.trim(),
        });
        toast.success("Knowledge base updated");
      } else {
        createdId = (
          await createKnowledgeBase({
            name: trimmed,
            description: description.trim() || undefined,
          })
        ).id;
        toast.success("Knowledge base created");
      }
      const created = (await refresh())?.find((row) => row.id === createdId);
      if (created) openDocuments(created);
      else setView({ kind: "list" });
    } catch (err) {
      toast.error("Save failed", {
        description: err instanceof Error ? err.message : String(err),
      });
    } finally {
      setSaving(false);
    }
  }

  async function removeKb(kb: KnowledgeBase) {
    try {
      await deleteKnowledgeBase(kb.id);
      await refresh();
    } catch (err) {
      toast.error("Delete failed", {
        description: err instanceof Error ? err.message : String(err),
      });
    }
  }

  const showForm = view.kind === "create" || view.kind === "edit";

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-2xl" onCloseAutoFocus={onCloseAutoFocus}>
        <DialogHeader>
          <DialogTitle>
            {view.kind === "documents" ? view.kb.name : "Knowledge bases"}
          </DialogTitle>
          <DialogDescription>
            {view.kind === "documents"
              ? "Upload documents to index for retrieval in chat."
              : "Group documents into a reusable knowledge base for chat retrieval."}
          </DialogDescription>
        </DialogHeader>

        {view.kind === "documents" ? (
          <KnowledgeBaseDocuments
            kb={view.kb}
            uploads={view.uploads}
            onUploadsStarted={takeUploads}
            onBack={backToList}
          />
        ) : showForm ? (
          <div className="flex flex-col gap-4">
            <div className="grid gap-2">
              <Label htmlFor="kb-name">Name</Label>
              <Input
                id="kb-name"
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="e.g. Product docs"
              />
            </div>
            <div className="grid gap-2">
              <Label htmlFor="kb-description">Description</Label>
              <Textarea
                id="kb-description"
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                placeholder="Optional. What this knowledge base contains."
                rows={3}
              />
            </div>
            <div className="flex justify-end gap-2 pt-2">
              <Button variant="ghost" onClick={backToList} disabled={saving}>
                Cancel
              </Button>
              <Button onClick={submitForm} disabled={saving || ragUnavailable}>
                {saving ? <Spinner /> : null}
                {view.kind === "edit" ? "Save changes" : "Create"}
              </Button>
            </div>
          </div>
        ) : (
          <div className="flex min-w-0 flex-col gap-3">
            <div className="flex justify-end">
              <Button
                size="sm"
                onClick={startCreate}
                disabled={ragUnavailable}
                title={ragUnavailableHint}
              >
                <HugeiconsIcon icon={PlusSignIcon} className="size-3.5" />
                New knowledge base
              </Button>
            </div>
            {loading ? (
              <div className="flex justify-center py-6">
                <Spinner />
              </div>
            ) : ragUnavailable ? (
              // An empty list on this host is not an empty store, so say which one it is.
              <div className="rounded-md border border-dashed px-4 py-6 text-center text-sm text-muted-foreground">
                {ragUnavailableReason ?? "Knowledge bases are unavailable."}
              </div>
            ) : kbs.length === 0 ? (
              <div className="rounded-md border border-dashed py-6 text-center text-sm text-muted-foreground">
                No knowledge bases yet.
              </div>
            ) : (
              <ul className="flex max-h-[60dvh] flex-col divide-y overflow-y-auto scroll-rounded rounded-md border">
                {kbs.map((kb) => (
                  <li
                    key={kb.id}
                    className="flex items-center justify-between gap-3 px-3 py-2"
                  >
                    <button
                      type="button"
                      onClick={() => openDocuments(kb)}
                      title="Open to add or remove documents"
                      className="-my-1 -ml-2 flex min-w-0 flex-1 items-center gap-2 rounded-md px-2 py-1 text-left transition-colors hover:bg-muted/60 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
                    >
                      <span className="block min-w-0 flex-1">
                        <span className="block truncate font-medium">
                          {kb.name}
                        </span>
                        <span className="block truncate text-xs text-muted-foreground">
                          {kb.documentCount ?? 0} document
                          {(kb.documentCount ?? 0) === 1 ? "" : "s"}
                          {kb.description ? ` · ${kb.description}` : ""}
                        </span>
                      </span>
                      <ChevronRightIcon
                        strokeWidth={1.5}
                        className="size-3.5 shrink-0 text-muted-foreground"
                      />
                    </button>
                    <div className="flex items-center gap-1">
                      <Button
                        type="button"
                        variant="ghost"
                        size="icon"
                        onClick={() => startEdit(kb)}
                        aria-label="Rename knowledge base"
                      >
                        <HugeiconsIcon icon={Edit03Icon} className="size-3.5" />
                      </Button>
                      <Button
                        type="button"
                        variant="ghost"
                        size="icon"
                        onClick={() => setConfirmingDelete(kb)}
                        aria-label="Delete knowledge base"
                      >
                        <HugeiconsIcon icon={Delete02Icon} className="size-3.5" />
                      </Button>
                    </div>
                  </li>
                ))}
              </ul>
            )}
          </div>
        )}
      </DialogContent>
      <AlertDialog
        open={confirmingDelete !== null}
        onOpenChange={(next) => {
          if (!next) setConfirmingDelete(null);
        }}
      >
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete knowledge base</AlertDialogTitle>
            <AlertDialogDescription>
              Delete{" "}
              <span className="font-medium text-foreground">
                &quot;{confirmingDelete?.name}&quot;
              </span>{" "}
              and all its documents? This cannot be undone.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction
              variant="destructive"
              onClick={() => {
                const kb = confirmingDelete;
                setConfirmingDelete(null);
                if (kb) void removeKb(kb);
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

function KnowledgeBaseDocuments({
  kb,
  uploads,
  onUploadsStarted,
  onBack,
}: {
  kb: KnowledgeBase;
  uploads?: RagUploadItem[];
  onUploadsStarted: () => void;
  onBack: () => void;
}) {
  const lister = useCallback(() => listKnowledgeBaseDocuments(kb.id), [kb.id]);
  const { documents, loading, uploading, refresh, upload, remove } =
    useRagDocuments({ type: "kb", kbId: kb.id }, lister);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const handleLinkedSourcesChanged = useCallback(() => {
    void refresh({ quiet: true });
  }, [refresh]);
  const { dragging, dropProps, nativeDropTarget } = useSourceDrop({
    onItems: (items) => void upload(items),
    // upload() tracks one run at a time, so a second batch would clear the
    // in-flight guard the first one is still relying on.
    disabledReason: uploading
      ? "An upload is already running. Add these when it finishes."
      : undefined,
  });

  // Deferred a tick: the hook's unmount cleanup aborts an upload started on StrictMode's
  // first mount. Taking the files off the view, not the deps, stops a rerun.
  // A batch already uploading here goes first: upload() tracks one run at a time.
  useEffect(() => {
    if (!uploads?.length || uploading) return;
    const timer = window.setTimeout(() => {
      onUploadsStarted();
      void upload(uploads);
    }, 0);
    return () => window.clearTimeout(timer);
  }, [uploads, uploading, onUploadsStarted, upload]);

  return (
    <div
      className="flex min-w-0 flex-col gap-3"
      ref={nativeDropTarget}
      {...dropProps}
    >
      <div className="flex items-center justify-between">
        <Button variant="ghost" size="sm" onClick={onBack}>
          <ChevronLeftIcon className="size-4" />
          All knowledge bases
        </Button>
        <Button
          size="sm"
          onClick={() => fileInputRef.current?.click()}
          disabled={uploading}
        >
          {uploading ? <Spinner /> : <UploadIcon className="size-3.5" />}
          Upload
        </Button>
        <input
          ref={fileInputRef}
          type="file"
          multiple={true}
          accept={RAG_SOURCE_UPLOAD_ACCEPT}
          className="hidden"
          onChange={(e) => {
            if (e.target.files?.length) void upload(e.target.files);
            e.target.value = "";
          }}
        />
      </div>
      {loading && documents.length === 0 ? (
        <div className="flex justify-center py-6">
          <Spinner />
        </div>
      ) : documents.length === 0 ? (
        <div
          className={cn(
            "rounded-md border border-dashed py-6 text-center text-sm text-muted-foreground transition-colors",
            dragging && "border-primary/60 bg-primary/5 text-foreground",
          )}
        >
          No documents yet. Upload or drop documents, spreadsheets, slides,
          e-books, email, text or code.
        </div>
      ) : (
        <div
          className={cn(
            "flex max-h-[55dvh] flex-wrap gap-1.5 overflow-y-auto scroll-rounded rounded-md pr-0.5 transition-colors",
            dragging && "bg-primary/5 ring-1 ring-primary/60",
          )}
        >
          {documents.map((doc) => (
            <DocumentStatusChip
              key={doc.id}
              filename={doc.filename}
              status={doc.status}
              progress={doc.progress}
              stage={doc.stage}
              error={doc.error}
              onRemove={
                doc.id.startsWith("pending_") || isLinkedFolderManaged(doc)
                  ? undefined
                  : () => void remove(doc.id)
              }
            />
          ))}
        </div>
      )}
      <div className="border-t pt-3">
        <LinkedFoldersManager
          scope={{ type: "knowledge_base", id: kb.id }}
          compact={true}
          onSourcesChanged={handleLinkedSourcesChanged}
        />
      </div>
    </div>
  );
}
