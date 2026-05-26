// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useCallback, useState } from "react";
import { toast } from "sonner";

import { subscribeToJobEvents } from "@/features/rag/api/rag-api";
import { useRagStore } from "@/features/rag/stores/rag-store";
import { useChatRuntimeStore } from "../stores/chat-runtime-store";

export type PendingDoc = {
  id: string;
  file: File;
  status: "uploading" | "ingesting" | "ready" | "error";
  jobId?: string;
  documentId?: string;
  errorMessage?: string;
};

const DOCUMENT_EXTENSIONS = new Set([
  ".pdf",
  ".txt",
  ".md",
  ".markdown",
  ".docx",
  ".html",
  ".htm",
]);

export const DOCUMENT_ACCEPT =
  ".pdf,.txt,.md,.markdown,.docx,.html,.htm,application/pdf,text/plain,text/markdown,application/vnd.openxmlformats-officedocument.wordprocessingml.document,text/html";

export function isDocumentFile(file: File): boolean {
  const lower = file.name.toLowerCase();
  const dot = lower.lastIndexOf(".");
  if (dot < 0) return false;
  return DOCUMENT_EXTENSIONS.has(lower.slice(dot));
}

export interface UseThreadDocUploadsResult {
  pendingDocs: PendingDoc[];
  addDoc: (file: File) => void;
  removeDoc: (id: string) => void;
  clearDocs: () => void;
  isIndexing: boolean;
}

/** Per-thread RAG upload: file → POST → SSE → chip status. */
export function useThreadDocUploads(): UseThreadDocUploadsResult {
  const activeThreadId = useChatRuntimeStore((s) => s.activeThreadId);
  const [pendingDocs, setPendingDocs] = useState<PendingDoc[]>([]);

  const addDoc = useCallback(
    (file: File) => {
      if (!activeThreadId) {
        toast.error("Attach to a thread first");
        return;
      }
      const id = crypto.randomUUID();
      setPendingDocs((prev) => [...prev, { id, file, status: "uploading" }]);
      const uploadDocument = useRagStore.getState().uploadDocument;
      uploadDocument({ kind: "thread", threadId: activeThreadId }, file)
        .then(({ documentId, jobId }) => {
          setPendingDocs((prev) =>
            prev.map((d) =>
              d.id === id
                ? { ...d, status: "ingesting", jobId, documentId }
                : d,
            ),
          );
          subscribeToJobEvents(jobId, {
            onEvent: (event) => {
              if (event.type === "complete") {
                setPendingDocs((prev) =>
                  prev.map((d) =>
                    d.id === id ? { ...d, status: "ready" } : d,
                  ),
                );
                // First ingest on an off-source thread → flip to 'thread'.
                if (useChatRuntimeStore.getState().ragSource.kind === "off") {
                  useChatRuntimeStore
                    .getState()
                    .setRagSource({ kind: "thread" });
                }
              } else if (event.type === "error") {
                setPendingDocs((prev) =>
                  prev.map((d) =>
                    d.id === id
                      ? { ...d, status: "error", errorMessage: event.error }
                      : d,
                  ),
                );
              }
            },
          });
        })
        .catch((err: unknown) => {
          const message = err instanceof Error ? err.message : "Upload failed";
          setPendingDocs((prev) =>
            prev.map((d) =>
              d.id === id
                ? { ...d, status: "error", errorMessage: message }
                : d,
            ),
          );
          toast.error(`Document upload failed: ${message}`);
        });
    },
    [activeThreadId],
  );

  const removeDoc = useCallback(
    (id: string) => {
      setPendingDocs((prev) => {
        const doc = prev.find((d) => d.id === id);
        if (doc?.documentId) {
          void useRagStore
            .getState()
            .deleteDocument(doc.documentId, `thread:${activeThreadId ?? ""}`)
            .catch(() => {});
        }
        return prev.filter((d) => d.id !== id);
      });
    },
    [activeThreadId],
  );

  const clearDocs = useCallback(() => setPendingDocs([]), []);

  const isIndexing = pendingDocs.some(
    (d) => d.status === "uploading" || d.status === "ingesting",
  );

  return { pendingDocs, addDoc, removeDoc, clearDocs, isIndexing };
}
