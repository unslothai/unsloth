// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { useNativeFileDrop } from "@/features/native-intents";
import type { NativeIntent } from "@/features/native-intents";
import { cn } from "@/lib/utils";
import { FolderAddIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useCallback, useEffect, useRef } from "react";
import {
  announceProjectSourcesUpdated,
  invalidateProjectSources,
  listProjectDocuments,
  subscribeProjectSourcesUpdated,
} from "../api/rag-api";
import { RAG_UPLOAD_ACCEPT, isLinkedFolderManaged } from "../types/rag";
import { DocumentStatusChip } from "./document-status-chip";
import { LinkedFoldersManager } from "./linked-folders-manager";
import { fileItems, useRagDocuments } from "./use-rag-documents";
import type { RagUploadItem } from "./use-rag-documents";

/** Project "Sources" tab: documents indexed for retrieval in every chat that
 * belongs to the project. */
export function ProjectSourcesPanel({ projectId }: { projectId: string }) {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const lister = useCallback(
    () => listProjectDocuments(projectId),
    [projectId],
  );
  const { documents, loading, uploading, refresh, upload, remove } =
    useRagDocuments({ type: "project", projectId }, lister);

  // Invalidate the sources probe before each mutation so a chat sent mid-upload
  // cannot cache "no sources" for the probe's TTL, and announce after it, which
  // is the half other instances and other tabs listen for. Announcing before
  // would refetch and resurrect the row this panel has already dropped.
  const handleItems = useCallback(
    async (items: RagUploadItem[]) => {
      if (items.length === 0) return;
      invalidateProjectSources(projectId);
      await upload(items);
      announceProjectSourcesUpdated(projectId);
    },
    [projectId, upload],
  );

  const handleFiles = useCallback(
    (files: File[]) => handleItems(fileItems(files)),
    [handleItems],
  );

  // Desktop drops arrive as paths; the upload mints a lease per file rather
  // than reading a document through the webview.
  const handleNativeIntents = useCallback(
    (intents: NativeIntent[]) =>
      handleItems(
        intents.map((intent) => ({
          kind: "native" as const,
          token: intent.path.token,
          name: intent.path.displayLabel,
          sizeBytes: intent.path.sizeBytes,
          modifiedMs: intent.path.modifiedMs,
        })),
      ),
    [handleItems],
  );

  const handleRemove = useCallback(
    async (documentId: string) => {
      invalidateProjectSources(projectId);
      await remove(documentId);
      announceProjectSourcesUpdated(projectId);
    },
    [projectId, remove],
  );
  const handleLinkedSourcesChanged = useCallback(() => {
    announceProjectSourcesUpdated(projectId);
    void refresh({ quiet: true });
  }, [projectId, refresh]);

  // External mutators (sidebar/thread saves, deletes elsewhere) announce when
  // they are done; refresh the mounted list so a source saved from a chat shows
  // up here without a remount. The list only polls while a row it already knows
  // is indexing, so nothing else would ever fetch it.
  useEffect(
    () =>
      subscribeProjectSourcesUpdated(projectId, () => {
        void refresh({ quiet: true });
      }),
    [projectId, refresh],
  );

  const empty = documents.length === 0;

  // Tauri suppresses webview drop events, so the plain `onDrop` this panel
  // carried never fired on desktop: no border, file ignored (#9036).
  const { ref: dropRef, dragging, dragHandlers } = useNativeFileDrop({
    onFiles: handleFiles,
    onNativeIntents: handleNativeIntents,
    accept: RAG_UPLOAD_ACCEPT,
    disabled: uploading,
    disabledReason: "Wait for the current upload to finish, then drop again.",
  });

  return (
    <div className="mt-8" ref={dropRef} {...dragHandlers}>
      <input
        ref={fileInputRef}
        type="file"
        multiple={true}
        accept={RAG_UPLOAD_ACCEPT}
        className="hidden"
        onChange={(e) => {
          const files = Array.from(e.target.files ?? []);
          e.target.value = "";
          void handleFiles(files);
        }}
      />
      <div className="mb-4 rounded-[22px] bg-muted/30 px-5 py-4">
        <LinkedFoldersManager
          scope={{ type: "project", id: projectId }}
          compact={true}
          onSourcesChanged={handleLinkedSourcesChanged}
        />
      </div>
      {empty ? (
        <div
          className={cn(
            "flex flex-col items-center justify-center gap-3 rounded-[26px] border border-transparent bg-muted/30 px-6 py-16 text-center transition-colors",
            dragging && "border-primary/60 bg-primary/5",
          )}
        >
          <span className="flex size-12 items-center justify-center rounded-full bg-muted text-muted-foreground">
            <HugeiconsIcon
              icon={FolderAddIcon}
              strokeWidth={1.75}
              className="size-6"
            />
          </span>
          <div className="space-y-1">
            <p className="text-ui-15 font-semibold text-foreground">
              Give this project context
            </p>
            <p className="max-w-sm text-sm text-muted-foreground">
              Upload PDFs, docs, or text. Every chat in this project can use
              them.
            </p>
          </div>
          <Button
            type="button"
            variant="outline"
            className="mt-1 border-none bg-background text-foreground shadow-[0_2px_8px_-2px_rgba(0,0,0,0.16)] hover:bg-background/80 dark:bg-card dark:shadow-none dark:hover:bg-accent/50"
            disabled={uploading || loading}
            onClick={() => fileInputRef.current?.click()}
          >
            Add sources
          </Button>
          <p className="text-ui-11 text-muted-foreground">Or drop files here</p>
        </div>
      ) : (
        <div
          className={cn(
            "flex flex-col gap-4 rounded-[26px] border border-transparent bg-muted/30 px-6 py-5 transition-colors",
            dragging && "border-primary/60 bg-primary/5",
          )}
        >
          <div className="flex items-center justify-between gap-3">
            <p className="text-sm text-muted-foreground">
              {documents.length === 1
                ? "1 source"
                : `${documents.length} sources`}
            </p>
            <Button
              type="button"
              size="sm"
              variant="outline"
              className="border-none bg-background text-foreground shadow-[0_2px_8px_-2px_rgba(0,0,0,0.16)] hover:bg-background/80 dark:bg-card dark:shadow-none dark:hover:bg-accent/50"
              disabled={uploading}
              onClick={() => fileInputRef.current?.click()}
            >
              Add sources
            </Button>
          </div>
          <div className="flex flex-row flex-wrap items-center gap-1.5">
            {documents.map((doc) => (
              <DocumentStatusChip
                key={doc.id}
                filename={doc.filename}
                status={doc.status}
                progress={doc.progress}
                error={doc.error}
                onRemove={
                  doc.id.startsWith("pending_") || isLinkedFolderManaged(doc)
                    ? undefined
                    : () => void handleRemove(doc.id)
                }
              />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
