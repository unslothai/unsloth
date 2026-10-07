// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  consumeNativePathToken,
  type NativeIntent,
} from "@/features/native-intents";
import { toast } from "@/lib/toast";
import {
  isBackendDownForDesktopUpdate,
  isSilencedDesktopUpdateFailure,
} from "@/lib/desktop-update-activity";
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
  stage?: string | null;
}

/** Matches the backend's folder scan interval. */
const FOLDER_RECONCILE_INTERVAL_MS = 30_000;

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

export function uploadItemFromIntent(intent: NativeIntent): RagUploadItem {
  return {
    kind: "native",
    token: intent.path.token,
    name: intent.displayLabel,
    sizeBytes: intent.path.sizeBytes,
    modifiedMs: intent.path.modifiedMs,
  };
}

function itemName(item: RagUploadItem): string {
  return item.kind === "file" ? item.file.name : item.name;
}

// Client-side dedup key; the backend dedups authoritatively by content hash.
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

const REFRESH_RETRIES = 3;

export function useRagDocuments(
  scope: RagDocumentScope | null,
  lister: Lister,
) {
  const [documents, setDocuments] = useState<TrackedDocument[]>([]);
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const trackedJobs = useRef<Map<string, AbortController>>(new Map());
  const documentsRef = useRef<TrackedDocument[]>([]);
  useEffect(() => {
    documentsRef.current = documents;
  }, [documents]);
  const sigByDocId = useRef<Map<string, string>>(new Map());
  // A 0-chunk doc stays re-ingestable (e.g. a scan attached before a vision model loaded).
  const sigBlocksReupload = useCallback((sig: string) => {
    const ids = new Set<string>();
    for (const [id, s] of sigByDocId.current) if (s === sig) ids.add(id);
    if (ids.size === 0) return false;
    const docs = documentsRef.current.filter((d) => ids.has(d.id));
    if (docs.length === 0) return false;
    return docs.some((d) => d.status !== "completed" || (d.numChunks ?? 0) > 0);
  }, []);
  // Lets the scope-change effect tell a real switch from lazy thread materialization mid-upload.
  const uploadInFlightRef = useRef(false);
  const materializedKeyRef = useRef<string | null>(null);
  const uploadGenerationRef = useRef(0);
  const activeUploadsRef = useRef(new Set<object>());
  useEffect(
    () => () => {
      uploadGenerationRef.current += 1;
      activeUploadsRef.current.clear();
      uploadInFlightRef.current = false;
      for (const controller of trackedJobs.current.values()) controller.abort();
      trackedJobs.current.clear();
    },
    [],
  );
  // Refreshes can complete out of order; only the newest publishes, or a stale list would drop
  // the indexing state sends are gated on.
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
      const generation = uploadGenerationRef.current;
      const stale = () =>
        controller.signal.aborted || generation !== uploadGenerationRef.current;
      const forget = () => {
        if (trackedJobs.current.get(jobId) === controller)
          trackedJobs.current.delete(jobId);
      };

      const finish = (
        status: TerminalJobStatus,
        error?: string | null,
        numChunks?: number | null,
      ) => {
        if (stale()) return forget();
        if (status === "cancelled") {
          sigByDocId.current.delete(documentId);
          setDocuments((rows) => rows.filter((row) => row.id !== documentId));
        } else if (status === "failed") {
          sigByDocId.current.delete(documentId);
          setDocuments((rows) => rows.filter((row) => row.id !== documentId));
          toast.error(`Couldn't index ${filename}`, {
            description: error ?? "Indexing failed",
          });
        } else {
          patchDoc(documentId, {
            status,
            error: null,
            progress: 1,
            ...(numChunks != null ? { numChunks } : {}),
          });
        }
        forget();
      };

      (async () => {
        // Past the shared stream budget in rag-api, streamJobEvents throws and this falls back to polling.
        try {
          for await (const ev of streamJobEvents(jobId, controller.signal)) {
            if (stale()) return forget();
            if (ev.type === "progress") {
              patchDoc(documentId, {
                status: "running",
                progress: ev.progress ?? null,
                stage: ev.stage ?? null,
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
          if (stale()) return forget();
          const job = await getJob(jobId, controller.signal);
          const terminal = terminalJobStatus(job.status);
          if (terminal) {
            finish(terminal, job.error, job.numChunks);
            return;
          }
        } catch {
          if (stale()) {
            forget();
            return;
          }
        }
        try {
          while (!stale()) {
            const job = await getJob(jobId, controller.signal);
            if (stale()) return forget();
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
              stage: job.stage ?? null,
            });
            await new Promise((r) => setTimeout(r, 1500));
          }
        } catch {
          // A list refresh can restart tracking after a request fails.
        } finally {
          forget();
        }
      })();
    },
    [patchDoc],
  );

  /** True when the list is known (published here or by a newer request); false only on failure. */
  const refresh = useCallback(
    async (opts?: { quiet?: boolean; silentErrors?: boolean }) => {
      if (!scopeKey) return true;
      const downWhenIssued = isBackendDownForDesktopUpdate();
      const requestId = ++refreshSeq.current;
      refreshInFlight.current = true;
      if (!opts?.quiet) setLoading(true);
      try {
        // Merge server rows with local progress so a refresh mid-index keeps live progress.
        const rows = (await lister()).filter((row) => row.status !== "failed");
        if (refreshSeq.current !== requestId) return true;
        setDocuments((prev) => {
          const merged = rows.map((row) => {
            const tracked = prev.find((p) => p.id === row.id);
            return tracked &&
              tracked.progress != null &&
              row.status !== "completed"
              ? { ...row, progress: tracked.progress, stage: tracked.stage }
              : row;
          });
          // Keep optimistic chips so a refresh racing an upload cannot make them vanish.
          const serverIds = new Set(rows.map((row) => row.id));
          const pendingLocal = prev.filter(
            (row) => row.id.startsWith("pending_") && !serverIds.has(row.id),
          );
          return [...merged, ...pendingLocal];
        });
        return true;
      } catch (err) {
        // No toast for a superseded scope, or one per composer on a host without RAG.
        if (refreshSeq.current !== requestId) return true;
        if (isSilencedDesktopUpdateFailure(err, downWhenIssued)) return false;
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
        if (refreshSeq.current === requestId) {
          refreshInFlight.current = false;
          setLoading(false);
        }
      }
    },
    [scopeKey, lister],
  );

  // Retry a failed project list: one failed read leaves nothing indexing or polling.
  const loadProjectSources = useCallback(
    async (projectId: string, opts?: { quiet?: boolean }) => {
      const startedFor = `project:${projectId}`;
      noteProjectWork(projectId, 1);
      try {
        for (let attempt = 0; attempt < REFRESH_RETRIES; attempt += 1) {
          const last = attempt === REFRESH_RETRIES - 1;
          if (await refresh({ quiet: opts?.quiet, silentErrors: !last })) return;
          if (last) break;
          await new Promise((resolve) =>
            setTimeout(resolve, 1000 * (attempt + 1)),
          );
          // This closure keeps its scope's lister, so a retry after navigation would publish into the new scope.
          if (liveScopeKeyRef.current !== startedFor) return;
        }
      } finally {
        noteProjectWork(projectId, -1);
      }
    },
    [refresh],
  );

  // Skip reset during materialization mid-upload (null scope -> new thread).
  useEffect(() => {
    const jobs = trackedJobs.current;
    const prev = prevScopeKeyRef.current;
    prevScopeKeyRef.current = scopeKey;
    const materialized = materializedKeyRef.current;
    if (scopeKey !== null) materializedKeyRef.current = null;
    const leftForAnotherChat =
      prev === null &&
      scopeKey !== null &&
      !uploadInFlightRef.current &&
      (jobs.size > 0 || materialized !== null) &&
      materialized !== scopeKey;
    if ((prev !== null && prev !== scopeKey) || leftForAnotherChat) {
      for (const controller of jobs.values()) controller.abort();
      jobs.clear();
      sigByDocId.current.clear();
      // Stand down in-flight refreshes for the old scope, or one would repopulate the cleared list.
      refreshSeq.current += 1;
      uploadGenerationRef.current += 1;
      activeUploadsRef.current.clear();
      uploadInFlightRef.current = false;
      setUploading(false);
      // Keep synchronous so StrictMode's setup/cleanup replay cannot cancel the only refresh.
      setDocuments([]);
      if (scope) {
        // eslint-disable-next-line react-hooks/set-state-in-effect
        void (scope.type === "project"
          ? loadProjectSources(scope.projectId)
          : refresh());
      } else {
        // Left set, the composer reads the list as unknown and holds every send.
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
      // Keep in-flight tracking across the materialization flip (React can commit the new id after the
      // POST returns); the next setup decides. An unmount aborts in the unmount effect.
      if (uploadInFlightRef.current || scopeKey === null) return;
      for (const controller of jobs.values()) controller.abort();
      jobs.clear();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [scopeKey]);

  // Safety net: HTTP/1.1 caps connections, so per-doc SSE streams may never finish; reconcile
  // against the list while anything is indexing.
  const [workElsewhere, setWorkElsewhere] = useState(0);
  const workScopeId = scope?.type === "project" ? scope.projectId : null;
  useEffect(() => {
    if (!workScopeId) {
      setWorkElsewhere(0);
      return;
    }
    const read = () => setWorkElsewhere(projectWorkCount(workScopeId));
    read();
    // Before the reconcile below, or an event fired ahead of this listener would be missed.
    window.addEventListener(PROJECT_WORK_CHANGED_EVENT, read);
    void reconcileProjectFolderJobs(workScopeId);
    // The backend also enqueues per-folder jobs on its own timer with no announcement.
    const reconcile = setInterval(() => {
      void reconcileProjectFolderJobs(workScopeId);
    }, FOLDER_RECONCILE_INTERVAL_MS);
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
    // Skip a tick while one is out, or a list slower than the interval would never publish.
    const id = setInterval(() => {
      if (!refreshInFlight.current) {
        void refresh({ quiet: true });
      }
    }, 4000);
    return () => clearInterval(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [scopeKey, hasIndexing]);

  const projectScopeId = scope?.type === "project" ? scope.projectId : null;
  useEffect(() => {
    if (!projectScopeId) return;
    const onChanged = (event: Event) => {
      const changed = (event as CustomEvent<{ projectId?: string }>).detail
        ?.projectId;
      if (changed !== projectScopeId) return;
      // Counted as work while it runs, or nothing gates the send between mutation and refresh.
      void loadProjectSources(projectScopeId, { quiet: true });
    };
    subscribeProjectSourcesBroadcast();
    window.addEventListener(PROJECT_SOURCES_CHANGED_EVENT, onChanged);
    return () =>
      window.removeEventListener(PROJECT_SOURCES_CHANGED_EVENT, onChanged);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [projectScopeId]);

  const uploadOne = useCallback(
    async (
      item: RagUploadItem,
      activeScope: RagDocumentScope,
      tempId: string,
      generation: number,
    ) => {
      const name = itemName(item);
      try {
        const { ocr, caption } = await resolveVisionOverrides();
        if (generation !== uploadGenerationRef.current) return;
        // Leases are short-lived, so mint one per upload rather than at drop time.
        const file =
          item.kind === "file"
            ? item.file
            : {
                nativePathLease: (
                  await consumeNativePathToken(item.token, "attach")
                ).nativePathLease,
              };
        if (generation !== uploadGenerationRef.current) return;
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
        if (generation !== uploadGenerationRef.current) return;
        sigByDocId.current.set(result.documentId, itemSignature(item));
        setDocuments((rows) =>
          rows.some((row) => row.id === result.documentId)
            ? rows.filter((row) => row.id !== tempId)
            : rows.map((row) =>
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
        if (generation !== uploadGenerationRef.current) return;
        const message = err instanceof Error ? err.message : String(err);
        setDocuments((rows) => rows.filter((row) => row.id !== tempId));
        toast.error(`Couldn't upload ${name}`, { description: message });
      }
    },
    [trackJob],
  );

  // `overrideScope`: the thread bar's id is still null on the first click.
  const upload = useCallback(
    async (
      files: FileList | File[] | RagUploadItem[],
      overrideScope?:
        | RagDocumentScope
        | Promise<RagDocumentScope | null>
        | (() => Promise<RagDocumentScope | null>),
    ) => {
      const generation = uploadGenerationRef.current;
      const uploadToken = {};
      activeUploadsRef.current.add(uploadToken);
      // Set synchronously before awaiting the thread id so the scope-change effect leaves tracking alone.
      uploadInFlightRef.current = true;
      setUploading(true);
      // Published before the first await so the other instance gates from the moment the upload starts.
      const knownScope =
        overrideScope instanceof Promise || typeof overrideScope === "function"
          ? null
          : (overrideScope ?? scope);
      const uploadingProjectId =
        knownScope?.type === "project" ? knownScope.projectId : null;
      if (uploadingProjectId) {
        noteProjectWork(uploadingProjectId, 1);
      }
      try {
        const fresh: Array<{ tempId: string; item: RagUploadItem }> = [];
        const entries: Array<File | RagUploadItem> = Array.isArray(files)
          ? files
          : Array.from(files);
        for (const entry of entries) {
          const item: RagUploadItem =
            entry instanceof File ? { kind: "file", file: entry } : entry;
          if (sigBlocksReupload(itemSignature(item))) {
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

        let activeScope: RagDocumentScope | null;
        try {
          activeScope =
            overrideScope === undefined
              ? scope
              : typeof overrideScope === "function"
                ? await overrideScope()
                : await overrideScope;
          if (!activeScope) {
            throw new Error("Could not start a chat to attach them to.");
          }
        } catch (err) {
          if (generation !== uploadGenerationRef.current) return;
          const tempIds = new Set(fresh.map((f) => f.tempId));
          setDocuments((rows) => rows.filter((row) => !tempIds.has(row.id)));
          toast.error("Couldn't attach documents", {
            description: err instanceof Error ? err.message : String(err),
          });
          return;
        }

        // A null-scope batch is exempt from the generation bump, so check here that the user did not
        // navigate to another chat.
        const resolvedKey =
          activeScope.type === "kb"
            ? `kb:${activeScope.kbId}`
            : activeScope.type === "project"
              ? `project:${activeScope.projectId}`
              : `thread:${activeScope.threadId}`;
        const liveKey = liveScopeKeyRef.current;
        if (knownScope === null && liveKey !== null && liveKey !== resolvedKey) {
          const tempIds = new Set(fresh.map((f) => f.tempId));
          setDocuments((rows) => rows.filter((row) => !tempIds.has(row.id)));
          return;
        }
        // The job may be running before React commits the scope it belongs to.
        if (liveKey === null) {
          materializedKeyRef.current = resolvedKey;
        }

        for (const { tempId, item } of fresh) {
          if (generation !== uploadGenerationRef.current) return;
          await uploadOne(item, activeScope, tempId, generation);
        }
      } finally {
        activeUploadsRef.current.delete(uploadToken);
        if (generation === uploadGenerationRef.current) {
          uploadInFlightRef.current = activeUploadsRef.current.size > 0;
          setUploading(uploadInFlightRef.current);
        }
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
      const prevSig = sigByDocId.current.get(documentId);
      sigByDocId.current.delete(documentId);
      // The source remains until the DELETE returns, so a send in between can still retrieve it.
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
