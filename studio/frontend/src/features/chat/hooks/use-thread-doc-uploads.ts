// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useAui } from "@assistant-ui/react";
import { useCallback, useState } from "react";
import { toast } from "sonner";

import { cancelJob, subscribeToJobEvents } from "@/features/rag/api/rag-api";
import { useIndexProgressStore } from "@/features/rag/stores/index-progress-store";
import { useRagStore } from "@/features/rag/stores/rag-store";
import { useChatRuntimeStore } from "../stores/chat-runtime-store";
import { acquireIndexSlot, releaseIndexSlot } from "../utils/rag-index-queue";

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

/** RAG upload from the composer "+" button. Routes by the Retrieval
 *  dropdown source: KB → that KB; thread/off → the current chat thread
 *  (lazy-initialized), flipping source to "thread" on first ingest if
 *  it was "off". */
export function useThreadDocUploads(): UseThreadDocUploadsResult {
  const aui = useAui();
  const activeThreadId = useChatRuntimeStore((s) => s.activeThreadId);
  const [pendingDocs, setPendingDocs] = useState<PendingDoc[]>([]);
  // Track scope key per chip so removeDoc dispatches the right delete
  // (KB vs thread docs use different scope keys in the store).
  const [chipScopeKeys, setChipScopeKeys] = useState<Record<string, string>>(
    {},
  );

  // Brand-new chats have no backend thread until the first message.
  // Initialize the local thread to mint a remoteId so RAG uploads can
  // attach before-send; the saved row is created on first send.
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
      // Lifecycle state shared by the upload flow and cancel thunk. The
      // thunk closes over these `let`s by reference, so it always sees
      // the latest job/document ids whenever the user cancels.
      const abort = new AbortController();
      let jobId: string | undefined;
      let documentId: string | undefined;
      let scopeKey: string | null = null;
      let unsubscribe: (() => void) | undefined;
      let slotAcquired = false;
      let slotReleased = false;
      let cleaned = false;
      const releaseSlot = () => {
        if (slotAcquired && !slotReleased) {
          slotReleased = true;
          releaseIndexSlot();
        }
      };
      const removeChip = () => {
        setPendingDocs((prev) => prev.filter((d) => d.id !== localChipId));
        setChipScopeKeys((m) => {
          const { [localChipId]: _gone, ...rest } = m;
          return rest;
        });
      };
      // Stop the backend job (if started) and delete its document to
      // reset the index. Idempotent: both a late in-flight abort and the
      // toast cancel can reach here.
      const cleanupBackend = async () => {
        if (cleaned) return;
        cleaned = true;
        if (jobId) await cancelJob(jobId);
        if (documentId && scopeKey) {
          try {
            await useRagStore.getState().deleteDocument(documentId, scopeKey);
          } catch {}
        }
      };

      setPendingDocs((prev) => [
        ...prev,
        { id: localChipId, file, status: "uploading" },
      ]);
      // Register in the aggregate-progress store now (synchronously, for
      // the whole batch) so the single toast counts queued files too.
      const indexProgress = useIndexProgressStore.getState();
      indexProgress.add(localChipId, file.name);
      indexProgress.setCancel(localChipId, async () => {
        abort.abort();
        unsubscribe?.();
        releaseSlot();
        await cleanupBackend();
        removeChip();
      });

      void (async () => {
        // Hold an indexing slot for this doc's whole lifecycle so bulk /
        // folder uploads drain at the configured concurrency instead of
        // spawning every ingestion at once. Released on every terminal path.
        await acquireIndexSlot();
        slotAcquired = true;
        if (abort.signal.aborted) {
          releaseSlot();
          return;
        }
        indexProgress.setIndexing(localChipId);
        const ragSource = useChatRuntimeStore.getState().ragSource;
        let scope:
          | { kind: "kb"; kbId: string }
          | { kind: "thread"; threadId: string }
          | null = null;
        if (ragSource.kind === "kb") {
          scope = { kind: "kb", kbId: ragSource.kbId };
          scopeKey = `kb:${ragSource.kbId}`;
        } else {
          // ragSource is "thread" or "off" — use thread.
          const threadId = await ensureThreadId();
          if (abort.signal.aborted) {
            releaseSlot();
            return;
          }
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
            indexProgress.setError(localChipId);
            releaseSlot();
            return;
          }
          scope = { kind: "thread", threadId };
          scopeKey = `thread:${threadId}`;
        }
        setChipScopeKeys((m) => ({ ...m, [localChipId]: scopeKey as string }));
        const uploadDocument = useRagStore.getState().uploadDocument;
        const captionImages =
          useChatRuntimeStore.getState().ragCaptionImages;
        try {
          const {
            documentId: did,
            jobId: jid,
            alreadyIndexed,
          } = await uploadDocument(scope, file, captionImages);
          documentId = did;
          jobId = jid;
          if (abort.signal.aborted) {
            // Cancelled mid-upload: the doc now exists on the backend,
            // so tear it down here.
            releaseSlot();
            await cleanupBackend();
            removeChip();
            return;
          }
          if (alreadyIndexed) {
            // Identical file already in this scope — no re-index. If a
            // chip for this doc already exists, drop the one we just
            // added so the composer doesn't show it twice; otherwise mark
            // this chip ready.
            setPendingDocs((prev) => {
              const dupExists = prev.some(
                (d) => d.id !== localChipId && d.documentId === did,
              );
              if (dupExists) {
                return prev.filter((d) => d.id !== localChipId);
              }
              return prev.map((d) =>
                d.id === localChipId
                  ? { ...d, status: "ready", documentId: did }
                  : d,
              );
            });
            toast.info(`${file.name} is already indexed`);
            if (
              scope?.kind === "thread" &&
              useChatRuntimeStore.getState().ragSource.kind === "off"
            ) {
              useChatRuntimeStore.getState().setRagSource({ kind: "thread" });
            }
            indexProgress.setReady(localChipId);
            releaseSlot();
            return;
          }
          setPendingDocs((prev) =>
            prev.map((d) =>
              d.id === localChipId
                ? { ...d, status: "ingesting", jobId: jid, documentId: did }
                : d,
            ),
          );
          unsubscribe = subscribeToJobEvents(jid, {
            onEvent: (event) => {
              if (event.type === "progress") {
                indexProgress.setProgress(localChipId, event.progress);
              } else if (event.type === "complete") {
                setPendingDocs((prev) =>
                  prev.map((d) =>
                    d.id === localChipId ? { ...d, status: "ready" } : d,
                  ),
                );
                // First ingest in an off-source chat: flip to thread so
                // the model has somewhere to search. KB uploads skip this
                // — source is already a KB.
                if (
                  scope?.kind === "thread" &&
                  useChatRuntimeStore.getState().ragSource.kind === "off"
                ) {
                  useChatRuntimeStore
                    .getState()
                    .setRagSource({ kind: "thread" });
                }
                indexProgress.setReady(localChipId, event.num_chunks);
                releaseSlot();
              } else if (event.type === "cancelled") {
                releaseSlot();
              } else if (event.type === "error") {
                setPendingDocs((prev) =>
                  prev.map((d) =>
                    d.id === localChipId
                      ? { ...d, status: "error", errorMessage: event.error }
                      : d,
                  ),
                );
                indexProgress.setError(localChipId);
                releaseSlot();
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
          indexProgress.setError(localChipId);
          releaseSlot();
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
          const scopeKey =
            chipScopeKeys[id] ?? `thread:${activeThreadId ?? ""}`;
          void useRagStore
            .getState()
            .deleteDocument(doc.documentId, scopeKey)
            .catch(() => {});
        }
        return prev.filter((d) => d.id !== id);
      });
      setChipScopeKeys((m) => {
        const { [id]: _gone, ...rest } = m;
        return rest;
      });
    },
    [activeThreadId, chipScopeKeys],
  );

  const clearDocs = useCallback(() => setPendingDocs([]), []);

  const isIndexing = pendingDocs.some(
    (d) => d.status === "uploading" || d.status === "ingesting",
  );

  return { pendingDocs, addDoc, removeDoc, clearDocs, isIndexing };
}
