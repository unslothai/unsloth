// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { useNativeFileDrop } from "@/features/native-intents";
import type { NativeIntent } from "@/features/native-intents";
import { isTauri } from "@/lib/api-base";
import { MAX_FOLDER_FILES, openFolderPicker } from "@/lib/dropped-folders";
import { FolderPlusIcon } from "@/lib/hugeicons-derived";
import { toast } from "@/lib/toast";
import { cn } from "@/lib/utils";
import { HugeiconsIcon } from "@hugeicons/react";
import { useCallback, useEffect, useMemo, useRef } from "react";
import {
  announceProjectSourcesUpdated,
  invalidateProjectSources,
  listProjectDocuments,
  subscribeProjectSourcesUpdated,
} from "../api/rag-api";
import { isLinkedFolderManaged } from "../types/rag";
import { DocumentStatusChip } from "./document-status-chip";
import { groupByLinkedFolder, useLinkedFolderNames } from "./linked-folder-groups";
import { LinkedFoldersManager } from "./linked-folders-manager";
import {
  RAG_SOURCE_UPLOAD_ACCEPT,
  SUPPORTED_SOURCES_HINT,
  isSupportedSourceName,
} from "./source-drop-policy";
import {
  type RagUploadItem,
  fileItems,
  useRagDocuments,
} from "./use-rag-documents";
import { useUploadQueue } from "./use-upload-queue";

/** Project "Sources" tab: documents indexed for retrieval in every chat that
 * belongs to the project. */
export function ProjectSourcesPanel({ projectId }: { projectId: string }) {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const lister = useCallback(
    () => listProjectDocuments(projectId),
    [projectId],
  );
  const {
    documents,
    loading,
    uploading,
    refresh,
    upload,
    remove,
    retry,
    canRetry,
  } = useRagDocuments({ type: "project", projectId }, lister);

  // Invalidate the sources probe before each mutation so a chat sent mid-upload
  // cannot cache "no sources" for the probe's TTL, and announce after it, which
  // is the half other instances and other tabs listen for. Announcing before
  // would refetch and resurrect the row this panel has already dropped.
  const uploadNow = useCallback(
    (items: RagUploadItem[]) => {
      if (items.length === 0) return;
      invalidateProjectSources(projectId);
      void upload(items).finally(() => announceProjectSourcesUpdated(projectId));
    },
    [projectId, upload],
  );
  // A drop or pick during an upload waits for it rather than being refused.
  const { enqueue: handleItems } = useUploadQueue(uploadNow, uploading, projectId);

  const handleRetry = useCallback(
    (documentId: string) => {
      invalidateProjectSources(projectId);
      void retry(documentId).finally(() =>
        announceProjectSourcesUpdated(projectId),
      );
    },
    [projectId, retry],
  );

  const pickFolder = useCallback(() => {
    openFolderPicker(({ files, truncated }) => {
      const supported = files.filter((file) => isSupportedSourceName(file.name));
      if (supported.length === 0) {
        toast.info("No supported files in that folder", {
          description: SUPPORTED_SOURCES_HINT,
        });
        return;
      }
      if (truncated > 0) {
        toast.info(`Added the first ${MAX_FOLDER_FILES} files`, {
          description: `${truncated} more were left out. Pick a smaller folder for the rest.`,
        });
      }
      handleItems(fileItems(supported));
    });
  }, [handleItems]);

  const groups = useMemo(() => groupByLinkedFolder(documents), [documents]);
  const folderNames = useLinkedFolderNames(
    projectId,
    groups.folders.map(([id]) => id).join(","),
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

  // External mutators (sidebar/thread saves, deletes elsewhere) announce when they are done;
  // refresh the mounted list so a source saved from a chat shows up here without a remount. The
  // list only polls while a row it already knows is indexing, so nothing else would ever fetch it.
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
  const {
    ref: dropRef,
    dragging,
    dragHandlers,
  } = useNativeFileDrop({
    onFiles: handleFiles,
    onNativeIntents: handleNativeIntents,
    accept: RAG_SOURCE_UPLOAD_ACCEPT,
    folders: true,
  });

  // A browser upload copies a folder in once; the desktop links it, which also keeps it in sync.
  const addFolderButton = isTauri ? null : (
    <Button
      type="button"
      size="sm"
      variant="ghost"
      className="text-muted-foreground"
      disabled={loading}
      onClick={pickFolder}
      title="Upload every supported file in a folder. Linking a folder that stays in sync needs the desktop app."
    >
      Add folder
    </Button>
  );

  return (
    <div className="mt-8" ref={dropRef} {...dragHandlers}>
      <input
        ref={fileInputRef}
        type="file"
        multiple={true}
        accept={RAG_SOURCE_UPLOAD_ACCEPT}
        className="hidden"
        onChange={(e) => {
          const files = Array.from(e.target.files ?? []);
          e.target.value = "";
          void handleItems(fileItems(files));
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
              icon={FolderPlusIcon}
              strokeWidth={1.75}
              className="size-6"
            />
          </span>
          <div className="space-y-1">
            <p className="text-ui-15 font-semibold text-foreground">
              Give this project context
            </p>
            <p className="max-w-sm text-sm text-muted-foreground">
              Add documents, spreadsheets, slides, e-books, email, text or
              code. Every chat in this project can use them.
            </p>
          </div>
          <div className="mt-1 flex items-center gap-1">
            <Button
              type="button"
              variant="outline"
              className="border-none bg-background text-foreground shadow-[0_2px_8px_-2px_rgba(0,0,0,0.16)] hover:bg-background/80 dark:bg-card dark:shadow-none dark:hover:bg-accent/50"
              disabled={loading}
              onClick={() => fileInputRef.current?.click()}
            >
              Add sources
            </Button>
            {addFolderButton}
          </div>
          <p className="text-ui-11 text-muted-foreground">
            {isTauri ? "Or drop files here" : "Or drop files or a folder here"}
          </p>
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
            <div className="flex items-center gap-1">
              {addFolderButton}
              <Button
                type="button"
                size="sm"
                variant="outline"
                className="border-none bg-background text-foreground shadow-[0_2px_8px_-2px_rgba(0,0,0,0.16)] hover:bg-background/80 dark:bg-card dark:shadow-none dark:hover:bg-accent/50"
                onClick={() => fileInputRef.current?.click()}
              >
                Add sources
              </Button>
            </div>
          </div>
          <div className="flex flex-row flex-wrap items-center gap-1.5">
            {/* A linked folder is one chip: it can hold thousands of files. */}
            {groups.folders.map(([folderId, docs]) => {
              const indexing = docs.filter(
                (doc) => doc.status === "pending" || doc.status === "running",
              ).length;
              const name = folderNames.get(folderId) ?? "Linked folder";
              return (
                <DocumentStatusChip
                  key={`folder:${folderId}`}
                  filename={`${name} · ${docs.length} file${docs.length === 1 ? "" : "s"}`}
                  status={indexing > 0 ? "running" : "completed"}
                  progress={
                    indexing > 0 ? (docs.length - indexing) / docs.length : null
                  }
                  shared={true}
                />
              );
            })}
            {groups.loose.map((doc) => (
              <DocumentStatusChip
                key={doc.id}
                filename={doc.filename}
                status={doc.status}
                progress={doc.progress}
                stage={doc.stage}
                error={doc.error}
                onRemove={
                  isLinkedFolderManaged(doc) ||
                  (doc.id.startsWith("pending_") && doc.status !== "failed")
                    ? undefined
                    : () => void handleRemove(doc.id)
                }
                onRetry={canRetry(doc.id) ? () => handleRetry(doc.id) : undefined}
              />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
