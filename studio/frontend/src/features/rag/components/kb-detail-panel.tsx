// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import { useState } from "react";
import type { KnowledgeBase } from "../api/rag-api";
import { useKBDocuments } from "../hooks/use-kb-documents";
import { DocumentRow } from "./document-row";
import { DocumentUploadDropzone } from "./document-upload-dropzone";
import { IngestionProgress } from "./ingestion-progress";

export function KBDetailPanel({ kb }: { kb: KnowledgeBase }) {
  const { documents, loading, error, upload, remove } = useKBDocuments(kb.id);
  const [activeJobsByDoc, setActiveJobsByDoc] = useState<Record<string, string>>(
    {},
  );

  const handleFiles = async (files: File[]) => {
    for (const file of files) {
      try {
        const { documentId, jobId } = await upload(file);
        setActiveJobsByDoc((prev) => ({ ...prev, [documentId]: jobId }));
      } catch (err) {
        console.error("upload failed", err);
      }
    }
  };

  return (
    <div className="flex h-full flex-col gap-4">
      <div className="flex flex-col gap-1">
        <h2 className="text-lg font-semibold">{kb.name}</h2>
        {kb.description ? (
          <p className="text-sm text-muted-foreground">{kb.description}</p>
        ) : null}
        <p className="text-xs text-muted-foreground">
          Embedding model: <code>{kb.embedding_model}</code>
        </p>
      </div>

      <DocumentUploadDropzone onFiles={handleFiles} />

      <Separator />

      <div className="flex items-center justify-between">
        <h3 className="text-sm font-medium">Documents</h3>
        <span className="text-xs text-muted-foreground">
          {loading ? "Loading…" : `${documents.length} total`}
        </span>
      </div>

      {error ? (
        <div className="text-xs text-destructive">{error}</div>
      ) : null}

      <ScrollArea className="flex-1">
        <div className="flex flex-col gap-2 pr-2">
          {documents.length === 0 && !loading ? (
            <div className="rounded-md border border-dashed border-border/60 px-3 py-6 text-center text-xs text-muted-foreground">
              No documents yet. Drop some files above to get started.
            </div>
          ) : null}
          {documents.map((doc) => {
            const jobId = activeJobsByDoc[doc.id];
            const showProgress =
              jobId && (doc.status === "pending" || doc.status === "running");
            return (
              <DocumentRow
                key={doc.id}
                doc={doc}
                onDelete={() => {
                  void remove(doc.id);
                }}
                rightSlot={
                  showProgress ? (
                    <IngestionProgress jobId={jobId} className="mt-1" />
                  ) : null
                }
              />
            );
          })}
        </div>
      </ScrollArea>
    </div>
  );
}
