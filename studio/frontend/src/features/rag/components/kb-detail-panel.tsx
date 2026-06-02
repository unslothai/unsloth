// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { useChatRuntimeStore } from "@/features/chat/stores/chat-runtime-store";
import {
  acquireIndexSlot,
  releaseIndexSlot,
} from "@/features/chat/utils/rag-index-queue";
import { cn } from "@/lib/utils";
import { FileTextIcon, Trash2Icon, XIcon } from "lucide-react";
import { useState } from "react";
import type { KnowledgeBase, RagDocument } from "../api/rag-api";
import { subscribeToJobEvents } from "../api/rag-api";
import { useKBDocuments } from "../hooks/use-kb-documents";
import { useIndexProgressStore } from "../stores/index-progress-store";
import { usePreviewStore } from "../stores/preview-store";
import { useRagStore } from "../stores/rag-store";
import { DocumentUploadDropzone } from "./document-upload-dropzone";
import { KBReconfigureDialog } from "./kb-reconfigure-dialog";

function humanBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

const STATUS_LABEL: Record<RagDocument["status"], string> = {
  pending: "Queued",
  running: "Indexing…",
  completed: "Completed",
  failed: "Failed",
};

export function KBDetailPanel({
  kb,
  panel,
  onClose,
}: {
  kb: KnowledgeBase;
  panel: "upload" | "files";
  onClose: () => void;
}) {
  const { documents, loading, error, remove } = useKBDocuments(kb.id);
  const [reconfigureOpen, setReconfigureOpen] = useState(false);
  const openPreview = usePreviewStore((s) => s.open);

  // Route uploads through the shared aggregate indexing toast (like the chat
  // composer) so a multi-file upload shows ONE progress bar, not a per-row
  // spinner. uploadDocument's own job subscription refreshes the doc list;
  // here we only drive the toast + concurrency semaphore.
  const handleFiles = (files: File[]) => {
    const indexProgress = useIndexProgressStore.getState();
    const captionImages = useChatRuntimeStore.getState().ragCaptionImages;
    const uploadDocument = useRagStore.getState().uploadDocument;
    for (const file of files) {
      const chipId = crypto.randomUUID();
      indexProgress.add(chipId, file.name);
      void (async () => {
        await acquireIndexSlot();
        indexProgress.setIndexing(chipId);
        let released = false;
        const release = () => {
          if (!released) {
            released = true;
            releaseIndexSlot();
          }
        };
        try {
          const { jobId, alreadyIndexed } = await uploadDocument(
            { kind: "kb", kbId: kb.id },
            file,
            captionImages,
          );
          if (alreadyIndexed || !jobId) {
            indexProgress.setReady(chipId);
            release();
            return;
          }
          subscribeToJobEvents(jobId, {
            onEvent: (event) => {
              if (event.type === "progress") {
                indexProgress.setProgress(chipId, event.progress);
              } else if (event.type === "complete") {
                indexProgress.setReady(chipId, event.num_chunks);
                release();
              } else if (event.type === "cancelled") {
                release();
              } else if (event.type === "error") {
                indexProgress.setError(chipId);
                release();
              }
            },
          });
        } catch {
          indexProgress.setError(chipId);
          release();
        }
      })();
    }
  };

  return (
    <div className="flex h-full min-w-0 flex-col gap-4 pt-2">
      <div className="flex flex-col gap-1.5">
        <div className="flex items-center justify-between gap-3">
          <h2 className="truncate text-lg font-semibold">{kb.name}</h2>
          <div className="flex shrink-0 items-center gap-1">
            <Button
              variant="outline"
              size="sm"
              onClick={() => setReconfigureOpen(true)}
              disabled={documents.length === 0}
              title={
                documents.length === 0
                  ? "Upload at least one document before re-indexing"
                  : undefined
              }
            >
              Reconfigure…
            </Button>
            <Button
              variant="ghost"
              size="icon"
              aria-label="Close panel"
              className="h-7 w-7 text-muted-foreground hover:text-foreground"
              onClick={onClose}
            >
              <XIcon className="size-4" />
            </Button>
          </div>
        </div>
        {/* Own full-width row so the embedder id fits on one line instead of
            wrapping next to the action buttons. */}
        <p className="truncate text-xs text-muted-foreground">
          Embedder: <code>{kb.embedding_model}</code>
        </p>
      </div>

      {panel === "upload" ? (
        <DocumentUploadDropzone onFiles={handleFiles} />
      ) : (
        <>
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-medium">Documents</h3>
            <span className="text-xs text-muted-foreground">
              {loading ? "Loading…" : `${documents.length} total`}
            </span>
          </div>
          {error ? (
            <div className="text-xs text-destructive">{error}</div>
          ) : null}
          {/* Native scroll, not Radix ScrollArea (its display:table wrapper
              breaks truncation). overflow-x-hidden is explicit because WebKit
              (Tauri webview) does NOT auto-compute overflow-x to auto when
              only overflow-y is set, so a long filename would otherwise spill
              and widen the whole dialog. */}
          <div className="min-h-0 min-w-0 flex-1 overflow-y-auto overflow-x-hidden pr-1">
            {documents.length === 0 && !loading ? (
              <div className="rounded-md border border-dashed border-border/60 px-3 py-6 text-center text-xs text-muted-foreground">
                No documents yet. Use the upload button to add some.
              </div>
            ) : (
              <div className="flex flex-col gap-2">
                {documents.map((doc) => {
                  const previewable = doc.status === "completed";
                  return (
                    <div
                      key={doc.id}
                      className={cn(
                        // grid minmax(0,1fr)/auto (like the Connections rows)
                        // so the name column shrinks and the delete button
                        // stays put — never widening the panel.
                        "grid w-full grid-cols-[minmax(0,1fr)_auto] items-center gap-2 rounded-lg border border-foreground/20 bg-muted px-3 py-1.5 text-xs",
                        previewable && "cursor-pointer hover:bg-muted/70",
                      )}
                      onClick={
                        previewable
                          ? () => void openPreview({ documentId: doc.id })
                          : undefined
                      }
                    >
                      <div className="flex min-w-0 items-center gap-2">
                        <FileTextIcon className="size-3.5 shrink-0 text-muted-foreground" />
                        <div className="flex min-w-0 flex-col">
                          <span className="truncate" title={doc.filename}>
                            {doc.filename}
                          </span>
                          <span
                            className={cn(
                              "text-[10px] leading-tight text-muted-foreground",
                              doc.status === "failed" && "text-destructive",
                            )}
                          >
                            {humanBytes(doc.byte_size)} · {doc.num_chunks} chunks
                            {" · "}
                            {STATUS_LABEL[doc.status]}
                          </span>
                        </div>
                      </div>
                      <button
                        type="button"
                        className="shrink-0 text-muted-foreground hover:text-destructive"
                        aria-label={`Delete ${doc.filename}`}
                        onClick={(e) => {
                          e.stopPropagation();
                          void remove(doc.id);
                        }}
                      >
                        <Trash2Icon className="size-3.5" />
                      </button>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        </>
      )}

      <KBReconfigureDialog
        open={reconfigureOpen}
        onOpenChange={setReconfigureOpen}
        kb={kb}
        documentCount={documents.length}
      />
    </div>
  );
}
