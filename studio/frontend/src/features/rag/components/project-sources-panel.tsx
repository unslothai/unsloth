// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { HugeiconsIcon } from "@hugeicons/react";
import { FileDatabaseIcon } from "@hugeicons/core-free-icons";
import { UploadIcon } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Spinner } from "@/components/ui/spinner";
import { toast } from "@/lib/toast";
import { useChatRuntimeStore } from "@/features/chat/stores/chat-runtime-store";

import { listProjectDocuments } from "../api/rag-api";
import {
  hasProjectRagSourcePreference,
  loadProjectRagSource,
  saveProjectRagSource,
} from "../project-rag-preferences";
import { RAG_UPLOAD_ACCEPT } from "../types/rag";
import { DocumentStatusChip } from "./document-status-chip";
import { useRagDocuments } from "./use-rag-documents";

export function ProjectSourcesPanel({ projectId }: { projectId: string }) {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const hadDocumentsRef = useRef(false);
  const setRagSource = useChatRuntimeStore((s) => s.setRagSource);
  const [preferenceVersion, setPreferenceVersion] = useState(0);
  const selectedSource = useMemo(
    () => loadProjectRagSource(projectId),
    [preferenceVersion, projectId],
  );
  const selectedDefault =
    selectedSource?.type === "project" && selectedSource.projectId === projectId
      ? "project"
      : selectedSource?.type === "kb"
        ? "kb"
      : "thread";
  const lister = useCallback(
    () => listProjectDocuments(projectId),
    [projectId],
  );
  const { documents, loading, uploading, upload, remove } = useRagDocuments(
    { type: "project", projectId },
    lister,
  );

  const setProjectDefault = useCallback(
    (value: "project" | "thread") => {
      const source =
        value === "project"
          ? ({ type: "project", projectId } as const)
          : ({ type: "thread" } as const);
      saveProjectRagSource(projectId, source);
      setRagSource(source);
      setPreferenceVersion((version) => version + 1);
    },
    [projectId, setRagSource],
  );

  useEffect(() => {
    setPreferenceVersion((version) => version + 1);
  }, [projectId]);

  const promptForDefault = useCallback(() => {
    if (hasProjectRagSourcePreference(projectId)) return;
    toast("Use project sources by default?", {
      description: "RAG chats in this project can retrieve from these sources.",
      action: {
        label: "Use by default",
        onClick: () => setProjectDefault("project"),
      },
      cancel: {
        label: "Not now",
        onClick: () => setProjectDefault("thread"),
      },
    });
  }, [projectId, setProjectDefault]);

  useEffect(() => {
    const hasDocuments = documents.length > 0;
    if (hasDocuments && !hadDocumentsRef.current) {
      promptForDefault();
    }
    hadDocumentsRef.current = hasDocuments;
  }, [documents.length, promptForDefault]);

  return (
    <div className="mt-8 flex min-w-0 flex-col gap-5">
      <input
        ref={fileInputRef}
        type="file"
        multiple
        accept={RAG_UPLOAD_ACCEPT}
        className="hidden"
        onChange={(event) => {
          const files = Array.from(event.target.files ?? []);
          event.target.value = "";
          if (files.length === 0) return;
          void upload(files);
        }}
      />

      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="min-w-0">
          <h2 className="text-[16px] font-semibold text-foreground">
            Project sources
          </h2>
          <p className="mt-1 text-sm text-muted-foreground">
            Upload documents for retrieval across chats in this project.
          </p>
        </div>
        <Button
          type="button"
          onClick={() => fileInputRef.current?.click()}
          disabled={uploading}
        >
          {uploading ? <Spinner /> : <UploadIcon className="size-4" />}
          Add sources
        </Button>
      </div>

      <div className="flex flex-wrap items-center gap-2">
        <span className="mr-1 text-sm font-medium text-foreground">
          Default for project chats
        </span>
        <button
          type="button"
          onClick={() => setProjectDefault("project")}
          data-active={selectedDefault === "project"}
          className="h-8 rounded-full border px-3 text-sm font-medium transition-colors data-[active=true]:border-border data-[active=true]:bg-muted data-[active=true]:text-foreground data-[active=false]:border-transparent data-[active=false]:text-muted-foreground data-[active=false]:hover:bg-nav-surface-hover"
        >
          Project sources
        </button>
        <button
          type="button"
          onClick={() => setProjectDefault("thread")}
          data-active={selectedDefault === "thread"}
          className="h-8 rounded-full border px-3 text-sm font-medium transition-colors data-[active=true]:border-border data-[active=true]:bg-muted data-[active=true]:text-foreground data-[active=false]:border-transparent data-[active=false]:text-muted-foreground data-[active=false]:hover:bg-nav-surface-hover"
        >
          Thread documents
        </button>
        {selectedDefault === "kb" ? (
          <span className="text-sm text-muted-foreground">
            Knowledge base selected in the RAG picker
          </span>
        ) : null}
      </div>

      {loading ? (
        <div className="flex justify-center py-10">
          <Spinner />
        </div>
      ) : documents.length === 0 ? (
        <div className="flex flex-col items-center justify-center gap-3 rounded-[16px] border border-dashed border-border/70 bg-muted/30 px-6 py-14 text-center">
          <span className="flex size-12 items-center justify-center rounded-full bg-muted text-muted-foreground">
            <HugeiconsIcon
              icon={FileDatabaseIcon}
              strokeWidth={1.75}
              className="size-6"
            />
          </span>
          <div className="space-y-1">
            <p className="text-[15px] font-semibold text-foreground">
              Give this project context
            </p>
            <p className="max-w-sm text-sm text-muted-foreground">
              Upload PDFs, documents, or other text. The model can reference
              them in chats when RAG uses project sources.
            </p>
          </div>
          <Button
            type="button"
            className="mt-1"
            onClick={() => fileInputRef.current?.click()}
            disabled={uploading}
          >
            {uploading ? <Spinner /> : null}
            Add sources
          </Button>
        </div>
      ) : (
        <div className="flex max-h-[360px] flex-col gap-2 overflow-y-auto rounded-lg border border-border/70 p-2">
          {documents.map((doc) => (
            <div
              key={doc.id}
              className="flex min-h-11 items-center justify-between gap-3 rounded-md px-2 py-1.5 hover:bg-nav-surface-hover"
            >
              <div className="flex min-w-0 items-center gap-2">
                <HugeiconsIcon
                  icon={FileDatabaseIcon}
                  strokeWidth={1.75}
                  className="size-4 shrink-0 text-muted-foreground"
                />
                <span className="truncate text-sm font-medium text-foreground">
                  {doc.filename}
                </span>
              </div>
              <DocumentStatusChip
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
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
