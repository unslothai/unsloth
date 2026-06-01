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

export type RagDocumentScope =
  | { type: "kb"; kbId: string }
  | { type: "thread"; threadId: string };

type Lister = () => Promise<RagDocument[]>;

/**
 * Manage a document set for one scope (a KB or a thread): list, upload,
 * delete, and keep per-document indexing status fresh. While a job is in
 * flight it subscribes to /jobs/{jobId}/events (SSE) and falls back to
 * polling getJob on stream error. Returns the tracked rows + actions.
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
    setDocuments([]);
    if (scope) void refresh();
    return () => {
      for (const controller of trackedJobs.current.values()) controller.abort();
      trackedJobs.current.clear();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [scopeKey]);

  const upload = useCallback(
    async (files: FileList | File[]) => {
      if (!scope) return;
      setUploading(true);
      try {
        for (const file of Array.from(files)) {
          // Optimistic pending row so the chip shows immediately.
          const tempId = `pending_${Math.random().toString(36).slice(2)}`;
          setDocuments((rows) => [
            ...rows,
            {
              id: tempId,
              filename: file.name,
              status: "pending",
              progress: null,
            },
          ]);
          try {
            const result =
              scope.type === "kb"
                ? await uploadKnowledgeBaseDocument(scope.kbId, file)
                : await uploadThreadDocument(scope.threadId, file);
            // Swap the temp row for the real document id.
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
            setDocuments((rows) =>
              rows.map((row) =>
                row.id === tempId
                  ? {
                      ...row,
                      status: "failed",
                      error: err instanceof Error ? err.message : String(err),
                    }
                  : row,
              ),
            );
            toast.error(`Upload failed: ${file.name}`, {
              description: err instanceof Error ? err.message : String(err),
            });
          }
        }
      } finally {
        setUploading(false);
      }
    },
    [scope, trackJob],
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
