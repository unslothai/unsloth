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
 * Per-file identity for client-side dedup: name + size + last-modified. An
 * edited file (same name, different size/date) is treated as new; an identical
 * re-selection is skipped before any upload. The backend still dedups by
 * content hash as the authoritative cross-session check.
 */
function fileSignature(file: File): string {
  return `${file.name}|${file.size}|${file.lastModified}`;
}

export type RagDocumentScope =
  | { type: "kb"; kbId: string }
  | { type: "thread"; threadId: string };

type Lister = () => Promise<RagDocument[]>;

/**
 * Manage one scope's documents (a KB or thread): list, upload, delete, and keep
 * indexing status live via /jobs/{jobId}/events (SSE), polling getJob on error.
 */
export function useRagDocuments(
  scope: RagDocumentScope | null,
  lister: Lister,
) {
  const [documents, setDocuments] = useState<TrackedDocument[]>([]);
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  // Jobs we are already tracking (id -> abort) so we don't double-subscribe.
  const trackedJobs = useRef<Map<string, AbortController>>(new Map());
  // Live mirror of `documents` for synchronous dedup checks inside upload().
  const documentsRef = useRef<TrackedDocument[]>([]);
  useEffect(() => {
    documentsRef.current = documents;
  }, [documents]);
  // Signatures of files attached this scope so an identical re-selection is
  // skipped without an upload or a chip. Cleared when the scope changes.
  const uploadedSigs = useRef<Set<string>>(new Set());

  const scopeKey = scope
    ? scope.type === "kb"
      ? `kb:${scope.kbId}`
      : `thread:${scope.threadId}`
    : null;

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
    (jobId: string, documentId: string) => {
      if (trackedJobs.current.has(jobId)) return;
      const controller = new AbortController();
      trackedJobs.current.set(jobId, controller);

      const finish = (status: DocumentStatus, error?: string | null) => {
        patchDoc(documentId, { status, error: error ?? null, progress: 1 });
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
          // Stream ended without an explicit terminal frame: reconcile
          // against the job's final state.
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
          // SSE unavailable: poll the job to a terminal state.
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
      // Merge server truth with any locally-tracked progress so a refresh
      // mid-index doesn't blow away a live "running %" chip.
      const rows = await lister();
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

  // Reset + load when the scope changes. Abort any in-flight job streams.
  useEffect(() => {
    for (const controller of trackedJobs.current.values()) controller.abort();
    trackedJobs.current.clear();
    uploadedSigs.current.clear();
    setDocuments([]);
    if (scope) void refresh();
    return () => {
      for (const controller of trackedJobs.current.values()) controller.abort();
      trackedJobs.current.clear();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [scopeKey]);

  // Upload one file: optimistic chip -> POST -> swap to the real id. If the
  // backend deduped to an existing document, drop the optimistic chip so a
  // duplicate never shows (never as a "Failed"/red chip). `seenIds` carries ids
  // already present/added this batch.
  const uploadOne = useCallback(
    async (file: File, seenIds: Set<string>) => {
      if (!scope) return;
      const tempId = `pending_${Math.random().toString(36).slice(2)}`;
      setDocuments((rows) => [
        ...rows,
        { id: tempId, filename: file.name, status: "pending", progress: null },
      ]);
      try {
        const result =
          scope.type === "kb"
            ? await uploadKnowledgeBaseDocument(scope.kbId, file)
            : await uploadThreadDocument(scope.threadId, file);
        uploadedSigs.current.add(fileSignature(file));
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
        trackJob(result.jobId, result.documentId);
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        setDocuments((rows) =>
          rows.map((row) =>
            row.id === tempId ? { ...row, status: "failed", error: message } : row,
          ),
        );
        toast.error(`Upload failed: ${file.name}`, { description: message });
      }
    },
    [scope, trackJob],
  );

  const upload = useCallback(
    async (files: FileList | File[]) => {
      if (!scope) return;
      setUploading(true);
      const seenIds = new Set(documentsRef.current.map((d) => d.id));
      try {
        for (const file of Array.from(files)) {
          // Skip an identical re-selection (name + size + last-modified) before
          // any upload or chip; an edited file with the same name still uploads.
          if (uploadedSigs.current.has(fileSignature(file))) {
            toast.info(`${file.name} is already indexed - skipping`);
            continue;
          }
          await uploadOne(file, seenIds);
        }
      } finally {
        setUploading(false);
      }
    },
    [scope, uploadOne],
  );

  const remove = useCallback(
    async (documentId: string) => {
      const prev = documents;
      setDocuments((rows) => rows.filter((row) => row.id !== documentId));
      try {
        await deleteDocument(documentId);
      } catch (err) {
        setDocuments(prev);
        toast.error("Delete failed", {
          description: err instanceof Error ? err.message : String(err),
        });
      }
    },
    [documents],
  );

  return { documents, loading, uploading, refresh, upload, remove };
}
