// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { consumeNativePathToken } from "@/features/native-intents";
import { toast } from "@/lib/toast";
import { useCallback, useEffect, useRef, useState } from "react";
import {
  PROJECT_SOURCES_CHANGED_EVENT,
  PROJECT_WORK_CHANGED_EVENT,
  deleteDocument,
  getJob,
  noteProjectWork,
  projectWorkCount,
  reconcileProjectFolderJobs,
  streamJobEvents,
  subscribeProjectSourcesBroadcast,
  uploadKnowledgeBaseDocument,
  uploadProjectDocument,
  uploadThreadDocument,
} from "../api/rag-api";
import { useRagAvailabilityStore } from "../api/rag-availability";
import {
  type RagDocument,
  type TerminalJobStatus,
  terminalJobStatus,
} from "../types/rag";
import { resolveVisionOverrides } from "./vision-overrides";

export interface TrackedDocument extends RagDocument {
  progress?: number | null;
}

/** Matches the backend's folder scan interval, so a job it starts is counted
 * within one period of appearing. */
const FOLDER_RECONCILE_INTERVAL_MS = 30_000;

/** A browser File, or a desktop drop addressed by its native path token. */
export type RagUploadItem =
  | { kind: "file"; file: File }
  | {
      kind: "native";
      token: string;
      name: string;
      sizeBytes?: number | null;
      modifiedMs?: number | null;
    };

export function fileItems(files: FileList | File[]): RagUploadItem[] {
  return Array.from(files).map((file) => ({ kind: "file" as const, file }));
}

function itemName(item: RagUploadItem): string {
  return item.kind === "file" ? item.file.name : item.name;
}

// Client-side dedup key; backend dedups authoritatively by content hash. Native drops
// carry the size/mtime Rust stat'd, so same-named files from different folders are
// still distinct; without them the drop is never blocked client-side.
function itemSignature(item: RagUploadItem): string {
  if (item.kind === "file") {
    return `${item.file.name}|${item.file.size}|${item.file.lastModified}`;
  }
  if (item.sizeBytes == null || item.modifiedMs == null) {
    return `${item.name}|native|${item.token}`;
  }
  return `${item.name}|${item.sizeBytes}|${item.modifiedMs}`;
}

export type RagDocumentScope =
  | { type: "kb"; kbId: string }
  | { type: "thread"; threadId: string }
  | { type: "project"; projectId: string };

type Lister = () => Promise<RagDocument[]>;

/** Attempts a mutation's reconciling list gets before the gate is released. */
const REFRESH_RETRIES = 3;

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
  // documentId -> signature; forgotten on delete, cleared on scope change.
  const sigByDocId = useRef<Map<string, string>>(new Map());
  // Skip a re-selected file only if a matching doc is healthy or still indexing. A doc
  // that completed with 0 chunks is re-ingestable (e.g. a scan attached before a vision
  // model loaded); the backend re-ingests on the same hash, so let it through.
  const sigBlocksReupload = useCallback((sig: string) => {
    const ids = new Set<string>();
    for (const [id, s] of sigByDocId.current) if (s === sig) ids.add(id);
    if (ids.size === 0) return false;
    const docs = documentsRef.current.filter((d) => ids.has(d.id));
    if (docs.length === 0) return false; // sig tracked but doc gone -> allow re-upload
    return docs.some((d) => d.status !== "completed" || (d.numChunks ?? 0) > 0);
  }, []);
  // True while upload() runs, so the scope-change effect can tell a real switch
  // from lazy thread materialization mid-upload (which must not reset).
  const uploadInFlightRef = useRef(false);
  // Refreshes fire from several places at once (a mutation invalidates before
  // and after, the poll ticks, the scope changes) and can complete out of order,
  // so only the newest publishes: an earlier one landing last would restore the
  // pre-mutation list and drop the indexing state sends are gated on.
  const refreshSeq = useRef(0);
  const refreshInFlight = useRef(false);

  const scopeKey = scope
    ? scope.type === "kb"
      ? `kb:${scope.kbId}`
      : scope.type === "project"
        ? `project:${scope.projectId}`
        : `thread:${scope.threadId}`
    : null;
  const prevScopeKeyRef = useRef<string | null>(null);
  // The scope on screen now, read by the async loops that outlive a navigation.
  const liveScopeKeyRef = useRef<string | null>(scopeKey);
  useEffect(() => {
    liveScopeKeyRef.current = scopeKey;
  }, [scopeKey]);

  const patchDoc = useCallback(
    (documentId: string, patch: Partial<TrackedDocument>) => {
      setDocuments((rows) =>
        rows.map((row) => (row.id === documentId ? { ...row, ...patch } : row)),
      );
    },
    [],
  );

  const trackJob = useCallback(
    (jobId: string, documentId: string, filename: string) => {
      if (trackedJobs.current.has(jobId)) return;
      const controller = new AbortController();
      trackedJobs.current.set(jobId, controller);

      const finish = (
        status: TerminalJobStatus,
        error?: string | null,
        numChunks?: number | null,
      ) => {
        if (status === "cancelled") {
          sigByDocId.current.delete(documentId);
          setDocuments((rows) => rows.filter((row) => row.id !== documentId));
        } else if (status === "failed") {
          // Drop the chip rather than show "Failed"; warn via toast.
          sigByDocId.current.delete(documentId);
          setDocuments((rows) => rows.filter((row) => row.id !== documentId));
          toast.error(`Couldn't index ${filename}`, {
            description: error ?? "Indexing failed",
          });
        } else {
          // Record numChunks so re-selecting this file dedups (vs a 0-chunk doc, which
          // stays re-ingestable); the SSE "complete" frame carries it.
          patchDoc(documentId, {
            status,
            error: null,
            progress: 1,
            ...(numChunks != null ? { numChunks } : {}),
          });
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
              finish("completed", null, ev.num_chunks);
              return;
            } else if (ev.type === "error") {
              finish(
                ev.stage === "cancelled" ? "cancelled" : "failed",
                ev.error ?? "Indexing failed",
              );
              return;
            }
          }
          // Stream ended with no terminal frame: reconcile.
          const job = await getJob(jobId);
          const terminal = terminalJobStatus(job.status);
          finish(terminal ?? "completed", job.error, job.numChunks);
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
              const terminal = terminalJobStatus(job.status);
              if (terminal) {
                return finish(
                  terminal,
                  terminal === "failed"
                    ? (job.error ?? "Indexing failed")
                    : job.error,
                  job.numChunks,
                );
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

  /** Resolves to whether the list is now known: true when this request
   * published it, or when a newer one took over and owns the answer. False
   * only when the request still being awaited failed. */
  const refresh = useCallback(
    async (opts?: { quiet?: boolean; silentErrors?: boolean }) => {
      if (!scopeKey) return true;
      const requestId = ++refreshSeq.current;
      refreshInFlight.current = true;
      if (!opts?.quiet) setLoading(true);
      try {
        // Merge server truth with local progress so a refresh mid-index keeps a
        // live "running %" chip. Failed docs hidden (toast warned at upload).
        const rows = (await lister()).filter((row) => row.status !== "failed");
        if (refreshSeq.current !== requestId) return true;
        setDocuments((prev) => {
          const merged = rows.map((row) => {
            const tracked = prev.find((p) => p.id === row.id);
            return tracked &&
              tracked.progress != null &&
              row.status !== "completed"
              ? { ...row, progress: tracked.progress }
              : row;
          });
          // Keep optimistic chips (not yet listed) so a refresh racing an upload
          // can't make them vanish.
          const serverIds = new Set(rows.map((row) => row.id));
          const pendingLocal = prev.filter(
            (row) => row.id.startsWith("pending_") && !serverIds.has(row.id),
          );
          return [...merged, ...pendingLocal];
        });
        return true;
      } catch (err) {
        // A superseded failure describes a scope no longer shown, and a host
        // without RAG 503s every one of these: no toast per composer opened.
        if (refreshSeq.current !== requestId) return true;
        if (
          !opts?.silentErrors &&
          !useRagAvailabilityStore.getState().isUnavailable()
        ) {
          toast.error("Failed to load documents", {
            description: err instanceof Error ? err.message : String(err),
          });
        }
        return false;
      } finally {
        // The newest clears it: a superseded request would report the list as
        // known while the one that will publish is still out.
        if (refreshSeq.current === requestId) {
          refreshInFlight.current = false;
          setLoading(false);
        }
      }
    },
    [scopeKey, lister],
  );

  // Retry a project list whose request failed: one failed read leaves the
  // composer with no rows, nothing indexing and nothing polling while a source
  // is still being chunked. Counted as work, so a send waits for the answer.
  const loadProjectSources = useCallback(
    async (projectId: string, opts?: { quiet?: boolean }) => {
      const startedFor = `project:${projectId}`;
      noteProjectWork(projectId, 1);
      try {
        for (let attempt = 0; attempt < REFRESH_RETRIES; attempt += 1) {
          const last = attempt === REFRESH_RETRIES - 1;
          // True for a request that published, and for one a newer request has
          // already outranked.
          if (await refresh({ quiet: opts?.quiet, silentErrors: !last })) return;
          if (last) break;
          await new Promise((resolve) =>
            setTimeout(resolve, 1000 * (attempt + 1)),
          );
          // This closure keeps the lister of the scope it started for, so a
          // retry after the user moves on would take a ticket behind the new
          // scope's request and publish into the composer showing it.
          if (liveScopeKeyRef.current !== startedFor) return;
        }
      } finally {
        noteProjectWork(projectId, -1);
      }
    },
    [refresh],
  );

  // A real switch (thread/KB swap) resets + reloads; first acquiring a scope just
  // loads. Skip both during materialization mid-upload (scope null -> new thread
  // while upload() runs) so we don't abort tracking or wipe optimistic chips.
  useEffect(() => {
    const jobs = trackedJobs.current;
    const prev = prevScopeKeyRef.current;
    prevScopeKeyRef.current = scopeKey;
    if (prev !== null && prev !== scopeKey) {
      for (const controller of jobs.values()) controller.abort();
      jobs.clear();
      sigByDocId.current.clear();
      // Stand down any refresh still in flight for the old scope. Clearing to a
      // null scope starts no replacement request to outrank it, so without this
      // its response would repopulate the list that is about to be cleared.
      refreshSeq.current += 1;
      // Scope changes intentionally clear the old scope before fetching the new
      // one. Keep this synchronous so React StrictMode's setup/cleanup replay
      // cannot cancel the only refresh after prevScopeKeyRef has advanced.
      setDocuments([]);
      if (scope) {
        // eslint-disable-next-line react-hooks/set-state-in-effect
        void (scope.type === "project"
          ? loadProjectSources(scope.projectId)
          : refresh());
      } else {
        // Nothing is coming for the scope just dropped, and the request that
        // was is now behind the sequence, so it will not clear these itself.
        // Left set, the composer reads the list as still unknown and holds
        // every send.
        refreshInFlight.current = false;
        // eslint-disable-next-line react-hooks/set-state-in-effect
        setLoading(false);
      }
    } else if (prev === null && scope && !uploadInFlightRef.current) {
      void (scope.type === "project"
        ? loadProjectSources(scope.projectId)
        : refresh());
    }
    return () => {
      // Preserve in-flight tracking when cleanup is the materialization flip,
      // not a real switch/unmount.
      if (uploadInFlightRef.current) return;
      for (const controller of jobs.values()) controller.abort();
      jobs.clear();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [scopeKey]);

  // Safety net: a big upload opens one SSE stream per doc, but HTTP/1.1 caps
  // concurrent connections, so streams past the cap may never deliver a terminal
  // frame and leave a chip spinning. While anything is indexing, reconcile against
  // the document list (one request covers every doc) so chips always resolve.
  // Work on this project running elsewhere: an upload from the other instance,
  // or a folder sync. Neither has a row here until it lands.
  const [workElsewhere, setWorkElsewhere] = useState(0);
  const workScopeId = scope?.type === "project" ? scope.projectId : null;
  useEffect(() => {
    if (!workScopeId) {
      setWorkElsewhere(0);
      return;
    }
    const read = () => setWorkElsewhere(projectWorkCount(workScopeId));
    read();
    // Before the reconcile below: it takes its own lease before its first await,
    // and an event fired ahead of this listener would be missed, leaving the
    // composer sendable for the length of the lookup.
    window.addEventListener(PROJECT_WORK_CHANGED_EVENT, read);
    // A sync already running on this project has no watcher here after a
    // reload, and its rows do not exist yet, so nothing else would gate on it.
    void reconcileProjectFolderJobs(workScopeId);
    // The backend also enqueues a job per auto-syncing folder on its own timer,
    // with nothing to announce it. Keep asking while the project is open, or a
    // scan that starts after this mount gates nothing.
    const reconcile = setInterval(() => {
      void reconcileProjectFolderJobs(workScopeId);
    }, FOLDER_RECONCILE_INTERVAL_MS);
    // Work in another tab arrives over the same channel.
    subscribeProjectSourcesBroadcast();
    return () => {
      clearInterval(reconcile);
      window.removeEventListener(PROJECT_WORK_CHANGED_EVENT, read);
    };
  }, [workScopeId]);

  const hasIndexing =
    workElsewhere > 0 ||
    documents.some((d) => d.status === "pending" || d.status === "running");
  useEffect(() => {
    if (!scopeKey || !hasIndexing) return;
    // Skip a tick while one is still out. Starting another would retire it
    // through the sequence gate, and a list slower than the interval would
    // then never publish: the row this is watching never reaches completed and
    // a queued send waits forever.
    const id = setInterval(() => {
      if (!refreshInFlight.current) {
        void refresh({ quiet: true });
      }
    }, 4000);
    return () => clearInterval(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [scopeKey, hasIndexing]);

  // A project's sources are shown by two independent instances (the composer's
  // bar and the Sources panel), so each has to pick up the other's mutations.
  const projectScopeId = scope?.type === "project" ? scope.projectId : null;
  useEffect(() => {
    if (!projectScopeId) return;
    const onChanged = (event: Event) => {
      const changed = (event as CustomEvent<{ projectId?: string }>).detail
        ?.projectId;
      if (changed !== projectScopeId) return;
      // The refresh an invalidation triggers is quiet, so it takes no loading
      // gate, and the mutation that fired it has released its own. Counted as
      // work while it runs, or nothing gates the send between the two.
      void loadProjectSources(projectScopeId, { quiet: true });
    };
    subscribeProjectSourcesBroadcast();
    window.addEventListener(PROJECT_SOURCES_CHANGED_EVENT, onChanged);
    return () =>
      window.removeEventListener(PROJECT_SOURCES_CHANGED_EVENT, onChanged);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [projectScopeId]);

  // POST one file, then swap its optimistic chip (`tempId`) to the real id; drop
  // the chip if the backend deduped. `seenIds` holds ids present/added this batch.
  const uploadOne = useCallback(
    async (
      item: RagUploadItem,
      seenIds: Set<string>,
      activeScope: RagDocumentScope,
      tempId: string,
    ) => {
      const name = itemName(item);
      try {
        const { ocr, caption } = await resolveVisionOverrides();
        // Leases are short-lived, so mint one per upload rather than at drop time.
        const file =
          item.kind === "file"
            ? item.file
            : {
                nativePathLease: (
                  await consumeNativePathToken(item.token, "attach")
                ).nativePathLease,
              };
        const result =
          activeScope.type === "kb"
            ? await uploadKnowledgeBaseDocument(
                activeScope.kbId,
                file,
                ocr,
                caption,
              )
            : activeScope.type === "project"
              ? await uploadProjectDocument(
                  activeScope.projectId,
                  file,
                  ocr,
                  caption,
                )
              : await uploadThreadDocument(
                  activeScope.threadId,
                  file,
                  ocr,
                  caption,
                );
        sigByDocId.current.set(result.documentId, itemSignature(item));
        if (seenIds.has(result.documentId)) {
          setDocuments((rows) => rows.filter((row) => row.id !== tempId));
          toast.info(
            `${result.filename || name} is already indexed - skipping`,
          );
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
        trackJob(result.jobId, result.documentId, result.filename || name);
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        // Drop the chip rather than show "Failed"; warn via toast.
        setDocuments((rows) => rows.filter((row) => row.id !== tempId));
        toast.error(`Couldn't upload ${name}`, { description: message });
      }
    },
    [trackJob],
  );

  // `overrideScope` lets a caller pass a freshly-resolved scope (or a promise of
  // one), since the thread bar's id is still null on the first click; falls back to
  // the hook scope.
  const upload = useCallback(
    async (
      files: FileList | File[] | RagUploadItem[],
      overrideScope?: RagDocumentScope | Promise<RagDocumentScope | null>,
    ) => {
      // Flip the in-flight guard synchronously, before awaiting a thread id that
      // may still be materializing, so the scope-change effect reads it and leaves
      // job tracking and optimistic chips alone.
      uploadInFlightRef.current = true;
      setUploading(true);
      // Published before the first await so the other instance gates from the
      // moment the upload starts, not once the bytes are in.
      // The composer passes its project scope explicitly, since the hook's own
      // can still be null on the render that starts the upload.
      const knownScope =
        overrideScope instanceof Promise ? null : (overrideScope ?? scope);
      const uploadingProjectId =
        knownScope?.type === "project" ? knownScope.projectId : null;
      if (uploadingProjectId) {
        noteProjectWork(uploadingProjectId, 1);
      }
      try {
        // Show an optimistic chip per file before awaiting the thread id;
        // materialization is a round-trip and gating chips behind it makes a slow
        // one look like nothing happened. Dedup re-selections up front.
        const fresh: Array<{ tempId: string; item: RagUploadItem }> = [];
        const entries: Array<File | RagUploadItem> = Array.isArray(files)
          ? files
          : Array.from(files);
        for (const entry of entries) {
          const item: RagUploadItem =
            entry instanceof File ? { kind: "file", file: entry } : entry;
          if (sigBlocksReupload(itemSignature(item))) {
            toast.info(`${itemName(item)} is already indexed - skipping`);
            continue;
          }
          fresh.push({
            tempId: `pending_${Math.random().toString(36).slice(2)}`,
            item,
          });
        }
        if (fresh.length === 0) return;
        setDocuments((rows) => [
          ...rows,
          ...fresh.map(({ tempId, item }) => ({
            id: tempId,
            filename: itemName(item),
            status: "pending" as const,
            managed: false,
            progress: null,
          })),
        ]);

        const resolved =
          overrideScope instanceof Promise
            ? await overrideScope
            : overrideScope;
        const activeScope = resolved ?? scope;
        if (!activeScope) {
          // Materialization failed: drop the chips so they don't hang "pending".
          const tempIds = new Set(fresh.map((f) => f.tempId));
          setDocuments((rows) => rows.filter((row) => !tempIds.has(row.id)));
          toast.error("Couldn't attach documents", {
            description: "Could not start a chat to attach them to.",
          });
          return;
        }

        const seenIds = new Set(
          documentsRef.current
            .filter((d) => !d.id.startsWith("pending_"))
            .map((d) => d.id),
        );
        for (const { tempId, item } of fresh) {
          await uploadOne(item, seenIds, activeScope, tempId);
        }
      } finally {
        setUploading(false);
        uploadInFlightRef.current = false;
        if (uploadingProjectId) {
          noteProjectWork(uploadingProjectId, -1);
        }
      }
    },
    [scope, uploadOne, sigBlocksReupload],
  );

  const remove = useCallback(
    async (documentId: string) => {
      const prev = documents;
      setDocuments((rows) => rows.filter((row) => row.id !== documentId));
      // Forget the dedup signature so re-uploading re-indexes.
      const prevSig = sigByDocId.current.get(documentId);
      sigByDocId.current.delete(documentId);
      // The chip goes at once but the source is still there until the DELETE
      // returns, and the sources probe is only invalidated after it does, so a
      // send in between can still retrieve what the composer says is gone.
      const removingProjectId =
        scope?.type === "project" ? scope.projectId : null;
      if (removingProjectId) {
        noteProjectWork(removingProjectId, 1);
      }
      try {
        await deleteDocument(
          documentId,
          scope?.type === "project" ? scope.projectId : undefined,
        );
      } catch (err) {
        setDocuments(prev);
        if (prevSig !== undefined) sigByDocId.current.set(documentId, prevSig);
        toast.error("Delete failed", {
          description: err instanceof Error ? err.message : String(err),
        });
      } finally {
        if (removingProjectId) {
          noteProjectWork(removingProjectId, -1);
        }
      }
    },
    [documents, scope],
  );

  return {
    documents,
    loading,
    uploading,
    hasIndexing,
    refresh,
    upload,
    remove,
  };
}
