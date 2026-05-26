// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useAui } from "@assistant-ui/react";
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
  const aui = useAui();
  const activeThreadId = useChatRuntimeStore((s) => s.activeThreadId);
  const [pendingDocs, setPendingDocs] = useState<PendingDoc[]>([]);

  // Brand-new chats have no backend thread until the first message is sent.
  // Initialize the current local thread to mint a remoteId so RAG uploads
  // can attach before-send; the saved thread row gets created on first send.
  const ensureThreadId = useCallback(async (): Promise<string | null> => {
    const stored = useChatRuntimeStore.getState().activeThreadId;
    if (stored) return stored;
    try {
      const runtime = aui.threads().__internal_getAssistantRuntime?.();
      if (!runtime) return null;
      const localId = runtime.threads.getState().mainThreadId;
      if (!localId) return null;
      const { remoteId } = await runtime.threads
        .getItemById(localId)
        .initialize();
      useChatRuntimeStore.getState().setActiveThreadId(remoteId);
      return remoteId;
    } catch {
      return null;
    }
  }, [aui]);

  const addDoc = useCallback(
    (file: File) => {
      const localChipId = crypto.randomUUID();
      setPendingDocs((prev) => [
        ...prev,
        { id: localChipId, file, status: "uploading" },
      ]);

      void (async () => {
        const threadId = await ensureThreadId();
        if (!threadId) {
          setPendingDocs((prev) =>
            prev.map((d) =>
              d.id === localChipId
                ? {
                    ...d,
                    status: "error",
                    errorMessage: "Could not create thread for upload",
                  }
                : d,
            ),
          );
          toast.error("Could not create thread for upload");
          return;
        }
        const uploadDocument = useRagStore.getState().uploadDocument;
        try {
          const { documentId, jobId } = await uploadDocument(
            { kind: "thread", threadId },
            file,
          );
          setPendingDocs((prev) =>
            prev.map((d) =>
              d.id === localChipId
                ? { ...d, status: "ingesting", jobId, documentId }
                : d,
            ),
          );
          subscribeToJobEvents(jobId, {
            onEvent: (event) => {
              if (event.type === "complete") {
                setPendingDocs((prev) =>
                  prev.map((d) =>
                    d.id === localChipId ? { ...d, status: "ready" } : d,
                  ),
                );
                if (useChatRuntimeStore.getState().ragSource.kind === "off") {
                  useChatRuntimeStore
                    .getState()
                    .setRagSource({ kind: "thread" });
                }
              } else if (event.type === "error") {
                setPendingDocs((prev) =>
                  prev.map((d) =>
                    d.id === localChipId
                      ? { ...d, status: "error", errorMessage: event.error }
                      : d,
                  ),
                );
              }
            },
          });
        } catch (err: unknown) {
          const message = err instanceof Error ? err.message : "Upload failed";
          setPendingDocs((prev) =>
            prev.map((d) =>
              d.id === localChipId
                ? { ...d, status: "error", errorMessage: message }
                : d,
            ),
          );
          toast.error(`Document upload failed: ${message}`);
        }
      })();
    },
    [ensureThreadId],
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
