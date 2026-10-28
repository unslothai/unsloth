// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { useNativeFileDrop } from "@/features/native-intents";
import type { NativeIntent } from "@/features/native-intents";
import { isTauri } from "@/lib/api-base";
import { MAX_FOLDER_FILES, openFolderPicker } from "@/lib/dropped-folders";
import { toast } from "@/lib/toast";
import { cn } from "@/lib/utils";
import { FolderAddIcon, PlusSignIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useCallback, useMemo, useRef } from "react";
import {
  announceProjectSourcesUpdated,
  invalidateProjectSources,
  listProjectDocuments,
} from "../api/rag-api";
import { isLinkedFolderManaged } from "../types/rag";
import { DocumentStatusChip } from "./document-status-chip";
import {
  groupByLinkedFolder,
  useLinkedFolderNames,
} from "./linked-folder-groups";
import { LinkedFoldersManager } from "./linked-folders-manager";
import { ProjectSourceChip } from "./project-source-chip";
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
      void upload(items).finally(() =>
        announceProjectSourcesUpdated(projectId),
      );
    },
    [projectId, upload],
  );
  const { enqueue: handleItems } = useUploadQueue(
    uploadNow,
    uploading,
    projectId,
  );

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
    openFolderPicker(({ files: supported, truncated }) => {
      if (supported.length === 0) {
        toast.info("No supported files in that folder", {
          description: SUPPORTED_SOURCES_HINT,
        });
        return;
      }
      if (truncated > 0) {
        toast.info(`Added the first ${MAX_FOLDER_FILES} files`, {
          description:
            "The rest were left out. Pick a smaller folder for them.",
        });
      }
      handleItems(fileItems(supported));
    }, isSupportedSourceName);
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
  // No refresh here: useRagDocuments refetches on this announce, and a second one doubles each sync tick.
  const handleLinkedSourcesChanged = useCallback(() => {
    announceProjectSourcesUpdated(projectId);
  }, [projectId]);

  const empty = documents.length === 0;

  // Tauri suppresses webview drop events, so a plain onDrop never fires on desktop.
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
      <HugeiconsIcon icon={FolderAddIcon} data-icon="inline-start" />
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
      <div className="mb-4 rounded-2xl bg-muted/30 px-5 py-4">
        <LinkedFoldersManager
          scope={{ type: "project", id: projectId }}
          compact={true}
          onSourcesChanged={handleLinkedSourcesChanged}
        />
      </div>
      {empty ? (
        <div
          className={cn(
            "flex flex-col items-center justify-center gap-3 rounded-2xl border border-transparent bg-muted/30 px-6 py-16 text-center transition-colors",
            dragging &&
              "border-wash-30 bg-wash-3 dark:border-wash-60 dark:bg-wash-5",
          )}
        >
          <div className="space-y-1">
            <p className="text-ui-15 font-semibold text-foreground">
              Give this project context
            </p>
            <p className="max-w-sm text-sm text-muted-foreground">
              Add documents, spreadsheets, slides, e-books, email, text or code.
              Every chat in this project can use them.
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
              <HugeiconsIcon icon={PlusSignIcon} data-icon="inline-start" />
              Add sources
            </Button>
          </div>
          <p className="text-ui-11 text-muted-foreground">
            {isTauri ? "Or drop files here" : "Or drop files or a folder here"}
          </p>
        </div>
      ) : (
        <div
          className={cn(
            "flex flex-col gap-4 rounded-2xl border border-transparent bg-muted/30 px-6 py-5 transition-colors",
            dragging &&
              "border-wash-30 bg-wash-3 dark:border-wash-60 dark:bg-wash-5",
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
                <HugeiconsIcon icon={PlusSignIcon} data-icon="inline-start" />
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
              <ProjectSourceChip
                key={doc.id}
                doc={doc}
                onRemove={
                  isLinkedFolderManaged(doc) ||
                  (doc.id.startsWith("pending_") && doc.status !== "failed")
                    ? undefined
                    : () => void handleRemove(doc.id)
                }
                onRetry={
                  canRetry(doc.id) ? () => handleRetry(doc.id) : undefined
                }
              />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
