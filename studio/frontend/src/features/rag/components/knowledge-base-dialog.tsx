// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useCallback, useEffect, useRef, useState } from "react";
import {
  Delete02Icon,
  Edit03Icon,
  PlusSignIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { ChevronLeftIcon, UploadIcon } from "lucide-react";

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

import {
  createKnowledgeBase,
  deleteKnowledgeBase,
  listKnowledgeBaseDocuments,
  listKnowledgeBases,
  updateKnowledgeBase,
} from "../api/rag-api";
import { RAG_UPLOAD_ACCEPT, type KnowledgeBase } from "../types/rag";
import { DocumentStatusChip } from "./document-status-chip";
import { useRagDocuments } from "./use-rag-documents";

type View =
  | { kind: "list" }
  | { kind: "create" }
  | { kind: "edit"; kb: KnowledgeBase }
  | { kind: "documents"; kb: KnowledgeBase };

export interface KnowledgeBaseDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export function KnowledgeBaseDialog({
  open,
  onOpenChange,
}: KnowledgeBaseDialogProps) {
  const [kbs, setKbs] = useState<KnowledgeBase[]>([]);
  const [loading, setLoading] = useState(false);
  const [view, setView] = useState<View>({ kind: "list" });
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [saving, setSaving] = useState(false);

  const refresh = useCallback(async () => {
    setLoading(true);
    try {
      setKbs(await listKnowledgeBases());
    } catch (err) {
      toast.error("Failed to load knowledge bases", {
        description: err instanceof Error ? err.message : String(err),
      });
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (!open) return;
    setView({ kind: "list" });
    void refresh();
  }, [open, refresh]);

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

  async function submitForm() {
    const trimmed = name.trim();
    if (!trimmed) {
      toast.error("Name is required");
      return;
    }
    setSaving(true);
    try {
      if (view.kind === "edit") {
        await updateKnowledgeBase(view.kb.id, {
          name: trimmed,
          description: description.trim(),
        });
        toast.success("Knowledge base updated");
      } else {
        await createKnowledgeBase({
          name: trimmed,
          description: description.trim() || undefined,
        });
        toast.success("Knowledge base created");
      }
      backToList();
      await refresh();
    } catch (err) {
      toast.error("Save failed", {
        description: err instanceof Error ? err.message : String(err),
      });
    } finally {
      setSaving(false);
    }
  }

  async function removeKb(kb: KnowledgeBase) {
    if (
      !window.confirm(
        `Delete knowledge base "${kb.name}" and all its documents?`,
      )
    ) {
      return;
    }
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
      <DialogContent className="max-w-2xl">
        <DialogHeader>
          <DialogTitle>
            {view.kind === "documents"
              ? view.kb.name
              : "Knowledge bases"}
          </DialogTitle>
          <DialogDescription>
            {view.kind === "documents"
              ? "Upload documents to index for retrieval in chat."
              : "Group documents into a reusable knowledge base for chat retrieval."}
          </DialogDescription>
        </DialogHeader>

        {view.kind === "documents" ? (
          <KnowledgeBaseDocuments kb={view.kb} onBack={backToList} />
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
              <Button onClick={submitForm} disabled={saving}>
                {saving ? <Spinner /> : null}
                {view.kind === "edit" ? "Save changes" : "Create"}
              </Button>
            </div>
          </div>
        ) : (
          <div className="flex min-w-0 flex-col gap-3">
            <div className="flex justify-end">
              <Button size="sm" onClick={startCreate}>
                <HugeiconsIcon icon={PlusSignIcon} size={14} />
                New knowledge base
              </Button>
            </div>
            {loading ? (
              <div className="flex justify-center py-6">
                <Spinner />
              </div>
            ) : kbs.length === 0 ? (
              <div className="rounded-md border border-dashed py-6 text-center text-sm text-muted-foreground">
                No knowledge bases yet.
              </div>
            ) : (
              <ul className="flex max-h-[60vh] flex-col divide-y overflow-y-auto rounded-md border">
                {kbs.map((kb) => (
                  <li
                    key={kb.id}
                    className="flex items-center justify-between gap-3 px-3 py-2"
                  >
                    <button
                      type="button"
                      onClick={() => setView({ kind: "documents", kb })}
                      className="min-w-0 flex-1 text-left"
                    >
                      <div className="truncate font-medium">{kb.name}</div>
                      <div className="truncate text-xs text-muted-foreground">
                        {kb.documentCount ?? 0} document
                        {(kb.documentCount ?? 0) === 1 ? "" : "s"}
                        {kb.description ? ` · ${kb.description}` : ""}
                      </div>
                    </button>
                    <div className="flex items-center gap-1">
                      <Button
                        type="button"
                        variant="ghost"
                        size="icon"
                        onClick={() => startEdit(kb)}
                        aria-label="Rename knowledge base"
                      >
                        <HugeiconsIcon icon={Edit03Icon} size={14} />
                      </Button>
                      <Button
                        type="button"
                        variant="ghost"
                        size="icon"
                        onClick={() => removeKb(kb)}
                        aria-label="Delete knowledge base"
                      >
                        <HugeiconsIcon icon={Delete02Icon} size={14} />
                      </Button>
                    </div>
                  </li>
                ))}
              </ul>
            )}
          </div>
        )}
      </DialogContent>
    </Dialog>
  );
}

function KnowledgeBaseDocuments({
  kb,
  onBack,
}: {
  kb: KnowledgeBase;
  onBack: () => void;
}) {
  const lister = useCallback(
    () => listKnowledgeBaseDocuments(kb.id),
    [kb.id],
  );
  const { documents, loading, uploading, upload, remove } = useRagDocuments(
    { type: "kb", kbId: kb.id },
    lister,
  );
  const fileInputRef = useRef<HTMLInputElement>(null);

  return (
    <div className="flex min-w-0 flex-col gap-3">
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
          multiple
          accept={RAG_UPLOAD_ACCEPT}
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
        <div className="rounded-md border border-dashed py-6 text-center text-sm text-muted-foreground">
          No documents yet. Upload a PDF, Markdown, DOCX, HTML, or text file.
        </div>
      ) : (
        <div className="flex max-h-[55vh] flex-wrap gap-1.5 overflow-y-auto pr-0.5">
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
      )}
    </div>
  );
}
