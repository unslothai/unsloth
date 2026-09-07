// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  pickNativeDocumentFolder,
  useNativePathLeasesSupported,
} from "@/features/native-intents";
import { isTauri } from "@/lib/api-base";
import { createScopedSingleFlightRequest } from "@/lib/single-flight-request";
import { toast } from "@/lib/toast";
import { useCallback, useEffect, useRef, useState } from "react";
import {
  announceProjectSourcesUpdated,
  createLinkedFolder,
  deleteLinkedFolder,
  getFolderSyncJob,
  listLinkedFolders,
  noteProjectWork,
  rebuildLinkedFolder,
  streamFolderSyncJobEvents,
  syncLinkedFolder,
  watchProjectFolderJob,
} from "../api/rag-api";
import type {
  FolderSyncJob,
  LinkedFolder,
  LinkedFolderScope,
} from "../types/rag";
import {
  linkedFolderSourcesChanged,
  retainActiveFolderJobs,
} from "../types/rag";
export function useLinkedFolders(
  scope?: LinkedFolderScope,
  onSourcesChanged?: () => void,
) {
  const scopeType = scope?.type;
  const scopeId = scope?.id;
  const scopeKey = scope ? `${scope.type}:${scope.id}` : "global";
  const [folders, setFolders] = useState<LinkedFolder[]>([]);
  const [jobs, setJobs] = useState<Record<string, FolderSyncJob>>({});
  const [stateScopeKey, setStateScopeKey] = useState(scopeKey);
  const [loading, setLoading] = useState(true);
  const [mutating, setMutating] = useState(false);
  const nativePathLeasesSupported = useNativePathLeasesSupported();
  const controllers = useRef(new Map<string, AbortController>());
  const refreshGate = useRef(
    createScopedSingleFlightRequest<
      (signal: AbortSignal) => Promise<boolean>,
      boolean
    >((_scope, execute, signal) => execute(signal)),
  );
  const currentScopeKey = useRef(scopeKey);
  const folderSnapshot = useRef<LinkedFolder[] | null>(null);
  const notifiedJobs = useRef(new Set<string>());

  // The backend starts the job before it answers, so the window before trackJob
  // registers it is the project changing with nothing gating on it. trackJob
  // takes its own lease inside this one, so the two overlap.
  const projectWorkScopeId = scopeType === "project" ? scopeId : null;
  const withProjectWork = useCallback(
    async <T>(run: () => Promise<T>): Promise<T> => {
      if (!projectWorkScopeId) return run();
      noteProjectWork(projectWorkScopeId, 1);
      try {
        return await run();
      } finally {
        noteProjectWork(projectWorkScopeId, -1);
      }
    },
    [projectWorkScopeId],
  );

  /** Count a job against the project it was started for, whatever this hook is
   * showing by the time the response lands. */
  const watchStartedJob = useCallback(
    (jobId: string) => {
      if (projectWorkScopeId) watchProjectFolderJob(projectWorkScopeId, jobId);
    },
    [projectWorkScopeId],
  );

  const notifySourcesChanged = useCallback(
    (job: FolderSyncJob) => {
      if (notifiedJobs.current.has(job.id)) return;
      notifiedJobs.current.add(job.id);
      onSourcesChanged?.();
    },
    [onSourcesChanged],
  );

  const trackJob = useCallback(
    (initial: FolderSyncJob) => {
      if (controllers.current.has(initial.id)) return;
      // A sync reports at start and completion only, so the composer has nothing
      // to gate on in between. The count follows the job, not this component:
      // leaving the Sources tab aborts the stream below but not the sync.
      if (scopeType === "project" && scopeId) {
        watchProjectFolderJob(scopeId, initial.id);
      }
      setJobs((current) => ({ ...current, [initial.linkedFolderId]: initial }));
      setFolders((current) =>
        current.map((folder) =>
          folder.id === initial.linkedFolderId
            ? { ...folder, status: "syncing", activeJobId: initial.id }
            : folder,
        ),
      );
      const controller = new AbortController();
      controllers.current.set(initial.id, controller);

      const releaseController = () => {
        if (controllers.current.get(initial.id) === controller) {
          controllers.current.delete(initial.id);
        }
      };
      const apply = (job: FolderSyncJob) => {
        if (controller.signal.aborted) return;
        setJobs((current) => ({ ...current, [job.linkedFolderId]: job }));
      };
      const finish = (job: FolderSyncJob) => {
        if (controller.signal.aborted) return;
        apply(job);
        releaseController();
        notifySourcesChanged(job);
        if (job.status === "failed") {
          toast.error("Folder sync failed", {
            description: job.error ?? "The folder could not be indexed.",
          });
        }
      };

      void (async () => {
        let latest = initial;
        try {
          for await (const event of streamFolderSyncJobEvents(
            initial.id,
            controller.signal,
          )) {
            if (controller.signal.aborted) return;
            latest = {
              ...latest,
              ...event,
              status:
                event.type === "complete"
                  ? "completed"
                  : event.type === "error"
                    ? "failed"
                    : (event.status ?? "running"),
            };
            apply(latest);
            if (latest.status === "completed" || latest.status === "failed") {
              finish(latest);
              return;
            }
          }
        } catch {
          if (controller.signal.aborted) return;
        }

        try {
          for (let attempt = 0; attempt < 600; attempt += 1) {
            if (controller.signal.aborted) return;
            latest = await getFolderSyncJob(initial.id);
            apply(latest);
            if (latest.status === "completed" || latest.status === "failed") {
              finish(latest);
              return;
            }
            await new Promise((resolve) => setTimeout(resolve, 1500));
          }
          // Exhausted: release too, or refresh keeps seeing the job as tracked
          // and never restarts polling, freezing progress until unmount.
          releaseController();
        } catch {
          releaseController();
        }
      })();
    },
    [notifySourcesChanged, scopeType, scopeId],
  );

  const refresh = useCallback(
    (options?: { quiet?: boolean }): Promise<boolean> => {
      if (!options?.quiet) setLoading(true);
      return refreshGate.current.run(scopeKey, async (signal) => {
        const isCurrent = () =>
          !signal.aborted && currentScopeKey.current === scopeKey;
        try {
          const rows = await listLinkedFolders(
            scopeType && scopeId ? { type: scopeType, id: scopeId } : undefined,
          );
          if (!isCurrent()) return false;
          const sourcesChanged = linkedFolderSourcesChanged(
            folderSnapshot.current,
            rows,
          );
          folderSnapshot.current = rows;
          setStateScopeKey(scopeKey);
          setFolders(rows);
          setJobs((current) => retainActiveFolderJobs(rows, current));
          if (sourcesChanged) onSourcesChanged?.();
          for (const folder of rows) {
            if (
              !folder.activeJobId ||
              controllers.current.has(folder.activeJobId)
            ) {
              continue;
            }
            try {
              const job = await getFolderSyncJob(folder.activeJobId);
              if (!isCurrent()) return false;
              if (job.status === "pending" || job.status === "running") {
                trackJob(job);
              } else {
                setJobs((current) => ({ ...current, [folder.id]: job }));
                if (sourcesChanged) notifiedJobs.current.add(job.id);
                else notifySourcesChanged(job);
              }
            } catch {
              // The next refresh can reconcile a job that disappeared mid-request.
            }
          }
          return sourcesChanged;
        } catch (error) {
          if (isCurrent() && !options?.quiet) {
            toast.error("Failed to load linked folders", {
              description:
                error instanceof Error ? error.message : String(error),
            });
          }
          return false;
        } finally {
          if (isCurrent() && !options?.quiet) setLoading(false);
        }
      });
    },
    [
      scopeKey,
      scopeType,
      scopeId,
      trackJob,
      notifySourcesChanged,
      onSourcesChanged,
    ],
  );

  useEffect(() => {
    currentScopeKey.current = scopeKey;
    folderSnapshot.current = null;
    notifiedJobs.current.clear();
    const initialRefresh = window.setTimeout(() => void refresh(), 0);
    const interval = window.setInterval(
      () => void refresh({ quiet: true }),
      4000,
    );
    const activeControllers = controllers.current;
    return () => {
      window.clearTimeout(initialRefresh);
      window.clearInterval(interval);
      for (const controller of activeControllers.values()) controller.abort();
      activeControllers.clear();
    };
  }, [refresh, scopeKey]);

  const link = useCallback(async () => {
    if (!scopeType || !scopeId || !isTauri || !nativePathLeasesSupported) return;
    const operationScopeKey = scopeKey;
    setMutating(true);
    try {
      const selected = await pickNativeDocumentFolder();
      if (!selected || currentScopeKey.current !== operationScopeKey) return;
      const result = await withProjectWork(async () => {
        const created = await createLinkedFolder(
          { type: scopeType, id: scopeId },
          selected.token,
          selected.displayName,
        );
        watchStartedJob(created.job.id);
        return created;
      });
      if (currentScopeKey.current !== operationScopeKey) return;
      setStateScopeKey(operationScopeKey);
      setFolders((current) => [
        ...current.filter((folder) => folder.id !== result.linkedFolder.id),
        result.linkedFolder,
      ]);
      trackJob(result.job);
      onSourcesChanged?.();
    } catch (error) {
      toast.error("Could not link folder", {
        description: error instanceof Error ? error.message : String(error),
      });
    } finally {
      setMutating(false);
    }
  }, [
    scopeKey,
    scopeType,
    scopeId,
    nativePathLeasesSupported,
    trackJob,
    watchStartedJob,
    withProjectWork,
    onSourcesChanged,
  ]);

  const run = useCallback(
    async (folderId: string, mode: "sync" | "rebuild") => {
      const operationScopeKey = scopeKey;
      try {
        // Inside the request's lease and before the scope guard: the job runs on
        // the project it started for whatever this hook shows, and returning
        // without a watcher drops that count to zero mid-sync. Deduped, so
        // trackJob's own call is a no-op.
        const { job } = await withProjectWork(async () => {
          const started =
            mode === "rebuild"
              ? await rebuildLinkedFolder(folderId)
              : await syncLinkedFolder(folderId);
          watchStartedJob(started.job.id);
          return started;
        });
        if (currentScopeKey.current !== operationScopeKey) return;
        trackJob(job);
        onSourcesChanged?.();
      } catch (error) {
        toast.error(`Could not ${mode} folder`, {
          description: error instanceof Error ? error.message : String(error),
        });
      }
    },
    [scopeKey, trackJob, watchStartedJob, withProjectWork, onSourcesChanged],
  );

  const remove = useCallback(
    async (folderId: string, removeIndex: boolean) => {
      const operationScopeKey = scopeKey;
      const previous = folders;
      const activeJobId =
        folders.find((folder) => folder.id === folderId)?.activeJobId ??
        jobs[folderId]?.id;
      const activeJob =
        jobs[folderId]?.id === activeJobId ? jobs[folderId] : undefined;
      if (activeJobId) {
        controllers.current.get(activeJobId)?.abort();
        controllers.current.delete(activeJobId);
      }
      setFolders((current) =>
        current.filter((folder) => folder.id !== folderId),
      );
      const unlinkedProjectId = projectWorkScopeId;
      try {
        await withProjectWork(() => deleteLinkedFolder(folderId, removeIndex));
        // The rows are gone whatever this hook shows by now, and every other
        // composer on that project still lists them. Announce for the project
        // the unlink was for, not for the scope on screen.
        if (unlinkedProjectId) announceProjectSourcesUpdated(unlinkedProjectId);
        if (currentScopeKey.current !== operationScopeKey) return;
        onSourcesChanged?.();
        setJobs((current) => {
          const next = { ...current };
          delete next[folderId];
          return next;
        });
      } catch (error) {
        if (currentScopeKey.current !== operationScopeKey) return;
        setFolders(previous);
        if (
          activeJob &&
          (activeJob.status === "pending" || activeJob.status === "running")
        ) {
          trackJob(activeJob);
        } else {
          void refresh({ quiet: true });
        }
        toast.error("Could not unlink folder", {
          description: error instanceof Error ? error.message : String(error),
        });
      }
    },
    [
      scopeKey,
      folders,
      jobs,
      trackJob,
      refresh,
      withProjectWork,
      onSourcesChanged,
      projectWorkScopeId,
    ],
  );

  return {
    folders: stateScopeKey === scopeKey ? folders : [],
    jobs: stateScopeKey === scopeKey ? jobs : {},
    loading: stateScopeKey === scopeKey ? loading : true,
    mutating,
    desktopSupported: isTauri && nativePathLeasesSupported,
    link,
    sync: (folderId: string) => run(folderId, "sync"),
    rebuild: (folderId: string) => run(folderId, "rebuild"),
    remove,
    refresh,
  };
}
