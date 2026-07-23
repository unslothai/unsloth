// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useCallback, useRef } from "react";
import { HugeiconsIcon } from "@hugeicons/react";
import { FolderAddIcon } from "@hugeicons/core-free-icons";
import { Button } from "@/components/ui/button";
import {
  invalidateProjectSources,
  listProjectDocuments,
} from "../api/rag-api";
import { RAG_UPLOAD_ACCEPT } from "../types/rag";
import { DocumentStatusChip } from "./document-status-chip";
import { useRagDocuments } from "./use-rag-documents";

/** Project "Sources" tab: documents indexed for retrieval in every chat that
 * belongs to the project. */
export function ProjectSourcesPanel({ projectId }: { projectId: string }) {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const lister = useCallback(() => listProjectDocuments(projectId), [projectId]);
  const { documents, loading, uploading, upload, remove } = useRagDocuments(
    { type: "project", projectId },
    lister,
  );

  // Invalidate the sources probe before and after each mutation: a chat sent
  // mid-upload must not cache "no sources" for the probe's TTL.
  const handleFiles = useCallback(
    async (files: File[]) => {
      if (files.length === 0) return;
      invalidateProjectSources(projectId);
      await upload(files);
      invalidateProjectSources(projectId);
    },
    [projectId, upload],
  );

  const handleRemove = useCallback(
    async (documentId: string) => {
      invalidateProjectSources(projectId);
      await remove(documentId);
      invalidateProjectSources(projectId);
    },
    [projectId, remove],
  );

  const empty = documents.length === 0;

  return (
    <div
      className="mt-8"
      onDragOver={(e) => e.preventDefault()}
      onDrop={(e) => {
        e.preventDefault();
        void handleFiles(Array.from(e.dataTransfer.files ?? []));
      }}
    >
      <input
        ref={fileInputRef}
        type="file"
        multiple
        accept={RAG_UPLOAD_ACCEPT}
        className="hidden"
        onChange={(e) => {
          const files = Array.from(e.target.files ?? []);
          e.target.value = "";
          void handleFiles(files);
        }}
      />
      {empty ? (
        <div className="flex flex-col items-center justify-center gap-3 rounded-[26px] bg-muted/30 px-6 py-16 text-center">
          <span className="flex size-12 items-center justify-center rounded-full bg-muted text-muted-foreground">
            <HugeiconsIcon
              icon={FolderAddIcon}
              strokeWidth={1.75}
              className="size-6"
            />
          </span>
          <div className="space-y-1">
            <p className="text-[0.9375rem] font-semibold text-foreground">
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
          <p className="text-[0.6875rem] text-muted-foreground">Or drop files here</p>
        </div>
      ) : (
        <div className="flex flex-col gap-4 rounded-[26px] bg-muted/30 px-6 py-5">
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
                  doc.id.startsWith("pending_")
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
