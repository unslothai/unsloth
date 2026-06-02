// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useCallback, useEffect, useRef, useState } from "react";
import { toast } from "@/lib/toast";
import {
  deleteDocument,
  getJob,
  streamJobEvents,
  uploadKnowledgeBaseDocument,
  uploadThreadDocument,
} from "../api/rag-api";
import type { DocumentStatus, RagDocument } from "../types/rag";

/** Local view of a document with live indexing progress. */
export interface TrackedDocument extends RagDocument {
  progress?: number | null;
}

/**
 * Client-side dedup identity: name + size + last-modified. An edit counts as
 * new; an identical re-selection is skipped pre-upload. Backend dedups by
 * content hash (authoritative).
 */
function fileSignature(file: File): string {
  return `${file.name}|${file.size}|${file.lastModified}`;
}

export type RagDocumentScope =
  | { type: "kb"; kbId: string }
  | { type: "thread"; threadId: string };

type Lister = () => Promise<RagDocument[]>;

/**
 * Manage one scope's documents (KB or thread): list, upload, delete, with live
 * indexing status via SSE (getJob polling on error).
 */
export function useRagDocuments(
  scope: RagDocumentScope | null,
  lister: Lister,
) {
  const [documents, setDocuments] = useState<TrackedDocument[]>([]);
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  // jobId -> abort, to avoid double-subscribing.
  const trackedJobs = useRef<Map<string, AbortController>>(new Map());
  // Live mirror of `documents` for synchronous dedup in upload().
  const documentsRef = useRef<TrackedDocument[]>([]);
  useEffect(() => {
    documentsRef.current = documents;
  }, [documents]);
  // documentId -> file signature for this scope's files. Keyed by id so a
  // re-selection is skipped, yet the record is forgotten on delete (re-upload
  // then re-indexes). Cleared on scope change.
  const sigByDocId = useRef<Map<string, string>>(new Map());
  const sigAttached = useCallback(
    (sig: string) => {
      for (const s of sigByDocId.current.values()) if (s === sig) return true;
      return false;
    },
    [],
  );
  // True while upload() runs. Lets the scope-change effect tell a real context
  // switch from lazy thread materialization mid-upload (which must not reset).
  const uploadInFlightRef = useRef(false);

  const scopeKey = scope
    ? scope.type === "kb"
      ? `kb:${scope.kbId}`
      : `thread:${scope.threadId}`
    : null;
  const prevScopeKeyRef = useRef<string | null>(null);

  const patchDoc = useCallback(
    (documentId: string, patch: Partial<TrackedDocument>) => {
      setDocuments((rows) =>
        rows.map((row) =>
          row.id === documentId ? { ...row, ...patch } : row,
        ),
      );
    },
    [],
  );

  const trackJob = useCallback(
    (jobId: string, documentId: string, filename: string) => {
      if (trackedJobs.current.has(jobId)) return;
      const controller = new AbortController();
      trackedJobs.current.set(jobId, controller);

      const finish = (status: DocumentStatus, error?: string | null) => {
        if (status === "failed") {
          // Drop the chip instead of showing "Failed"; warn via toast.
          setDocuments((rows) => rows.filter((row) => row.id !== documentId));
          toast.error(`Couldn't index ${filename}`, {
            description: error ?? "Indexing failed",
          });
        } else {
          patchDoc(documentId, { status, error: null, progress: 1 });
        }
        trackedJobs.current.delete(jobId);
      };

      (async () => {
        try {
          for await (const ev of streamJobEvents(jobId, controller.signal)) {
            if (ev.type === "progress") {
              patchDoc(documentId, {
                status: "running",
                progress: ev.progress ?? null,
              });
            } else if (ev.type === "complete") {
              finish("completed");
              return;
            } else if (ev.type === "error") {
              finish("failed", ev.error ?? "Indexing failed");
              return;
            }
          }
          // Stream ended with no terminal frame: reconcile final state.
          const job = await getJob(jobId);
          finish(
            job.status === "completed"
              ? "completed"
              : job.status === "failed"
                ? "failed"
                : "completed",
            job.error,
          );
        } catch {
          if (controller.signal.aborted) {
            trackedJobs.current.delete(jobId);
            return;
          }
          // SSE unavailable: poll to a terminal state.
          try {
            for (let i = 0; i < 600; i++) {
              if (controller.signal.aborted) break;
              const job = await getJob(jobId);
              if (job.status === "completed") return finish("completed");
              if (job.status === "failed") {
                return finish("failed", job.error ?? "Indexing failed");
              }
              patchDoc(documentId, {
                status: job.status === "running" ? "running" : "pending",
                progress: job.progress ?? null,
              });
              await new Promise((r) => setTimeout(r, 1500));
            }
          } catch {
            trackedJobs.current.delete(jobId);
          }
        }
      })();
    },
    [patchDoc],
  );

  const refresh = useCallback(async () => {
    if (!scope) return;
    setLoading(true);
    try {
      // Merge server truth with local progress so a refresh mid-index keeps a
      // live "running %" chip. Failed docs are hidden (toast warned at upload).
      const rows = (await lister()).filter((row) => row.status !== "failed");
      setDocuments((prev) =>
        rows.map((row) => {
          const tracked = prev.find((p) => p.id === row.id);
          return tracked && tracked.progress != null && row.status !== "completed"
            ? { ...row, progress: tracked.progress }
            : row;
        }),
      );
    } catch (err) {
      toast.error("Failed to load documents", {
        description: err instanceof Error ? err.message : String(err),
      });
    } finally {
      setLoading(false);
    }
  }, [scope, lister]);

  // On scope change: a real switch (thread/KB swap) resets + reloads; first
  // acquiring a scope just loads. Skip both during lazy thread materialization
  // mid-upload (scope goes null -> new thread while upload() runs) so we don't
  // abort the upload's job tracking or wipe its optimistic chips.
  useEffect(() => {
    const prev = prevScopeKeyRef.current;
    prevScopeKeyRef.current = scopeKey;
    if (prev !== null && prev !== scopeKey) {
      for (const controller of trackedJobs.current.values()) controller.abort();
      trackedJobs.current.clear();
      sigByDocId.current.clear();
      setDocuments([]);
      if (scope) void refresh();
    } else if (prev === null && scope && !uploadInFlightRef.current) {
      void refresh();
    }
    return () => {
      // Preserve an in-flight upload's tracking when the cleanup is the lazy
      // materialization flip rather than a real switch/unmount.
      if (uploadInFlightRef.current) return;
      for (const controller of trackedJobs.current.values()) controller.abort();
      trackedJobs.current.clear();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [scopeKey]);

  // Upload one file: optimistic chip -> POST -> swap to the real id. If the
  // backend deduped to an existing document, drop the chip. `seenIds` holds
  // ids present/added this batch.
  const uploadOne = useCallback(
    async (file: File, seenIds: Set<string>, activeScope: RagDocumentScope) => {
      const tempId = `pending_${Math.random().toString(36).slice(2)}`;
      setDocuments((rows) => [
        ...rows,
        { id: tempId, filename: file.name, status: "pending", progress: null },
      ]);
      try {
        const result =
          activeScope.type === "kb"
            ? await uploadKnowledgeBaseDocument(activeScope.kbId, file)
            : await uploadThreadDocument(activeScope.threadId, file);
        sigByDocId.current.set(result.documentId, fileSignature(file));
        if (seenIds.has(result.documentId)) {
          setDocuments((rows) => rows.filter((row) => row.id !== tempId));
          toast.info(`${result.filename || file.name} is already indexed - skipping`);
          return;
        }
        seenIds.add(result.documentId);
        setDocuments((rows) =>
          rows.map((row) =>
            row.id === tempId
              ? {
                  ...row,
                  id: result.documentId,
                  filename: result.filename || row.filename,
                  status: "running",
                }
              : row,
          ),
        );
        trackJob(result.jobId, result.documentId, result.filename || file.name);
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        // Drop the chip instead of showing "Failed"; warn via toast.
        setDocuments((rows) => rows.filter((row) => row.id !== tempId));
        toast.error(`Couldn't upload ${file.name}`, { description: message });
      }
    },
    [trackJob],
  );

  // `overrideScope` lets a caller pass a freshly-resolved scope (the thread bar
  // materializes its thread id at upload time, so the hook's `scope` prop is
  // still null on the first click); falls back to the hook scope otherwise.
  const upload = useCallback(
    async (files: FileList | File[], overrideScope?: RagDocumentScope) => {
      const activeScope = overrideScope ?? scope;
      if (!activeScope) return;
      uploadInFlightRef.current = true;
      setUploading(true);
      const seenIds = new Set(documentsRef.current.map((d) => d.id));
      try {
        for (const file of Array.from(files)) {
          // Skip an identical re-selection pre-upload; an edited same-name file
          // still uploads.
          if (sigAttached(fileSignature(file))) {
            toast.info(`${file.name} is already indexed - skipping`);
            continue;
          }
          await uploadOne(file, seenIds, activeScope);
        }
      } finally {
        setUploading(false);
        uploadInFlightRef.current = false;
      }
    },
    [scope, uploadOne, sigAttached],
  );

  const remove = useCallback(
    async (documentId: string) => {
      const prev = documents;
      setDocuments((rows) => rows.filter((row) => row.id !== documentId));
      // Forget the dedup signature so re-uploading re-indexes.
      const prevSig = sigByDocId.current.get(documentId);
      sigByDocId.current.delete(documentId);
      try {
        await deleteDocument(documentId);
      } catch (err) {
        setDocuments(prev);
        if (prevSig !== undefined) sigByDocId.current.set(documentId, prevSig);
        toast.error("Delete failed", {
          description: err instanceof Error ? err.message : String(err),
        });
      }
    },
    [documents],
  );

  return { documents, loading, uploading, refresh, upload, remove };
}
