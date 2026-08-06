// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { pickNativeDocumentFolder } from "@/features/native-intents";
import { isTauri } from "@/lib/api-base";
import { toast } from "@/lib/toast";
import { useCallback, useEffect, useRef, useState } from "react";
import {
  createLinkedFolder,
  deleteLinkedFolder,
  getFolderSyncJob,
  listLinkedFolders,
  rebuildLinkedFolder,
  streamFolderSyncJobEvents,
  syncLinkedFolder,
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
import {
  createScopedRefreshGate,
  runScopedRefresh,
  setScopedRefreshScope,
} from "./scoped-refresh";

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
  const controllers = useRef(new Map<string, AbortController>());
  const refreshGate = useRef(createScopedRefreshGate(scopeKey));
  const currentScopeKey = useRef(scopeKey);
  const folderSnapshot = useRef<LinkedFolder[] | null>(null);
  const notifiedJobs = useRef(new Set<string>());

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

      const apply = (job: FolderSyncJob) => {
        setJobs((current) => ({ ...current, [job.linkedFolderId]: job }));
      };
      const finish = (job: FolderSyncJob) => {
        apply(job);
        controllers.current.delete(job.id);
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
        } catch {
          controllers.current.delete(initial.id);
        }
      })();
    },
    [notifySourcesChanged],
  );

  const refresh = useCallback(
    (options?: { quiet?: boolean }): Promise<boolean> => {
      if (!options?.quiet) setLoading(true);
      return runScopedRefresh(
        refreshGate.current,
        scopeKey,
        async (isCurrent) => {
          try {
            const rows = await listLinkedFolders(
              scopeType && scopeId
                ? { type: scopeType, id: scopeId }
                : undefined,
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
        },
      ).then((result) => result ?? false);
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
    setScopedRefreshScope(refreshGate.current, scopeKey);
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
    if (!scopeType || !scopeId || !isTauri) return;
    const operationScopeKey = scopeKey;
    setMutating(true);
    try {
      const selected = await pickNativeDocumentFolder();
      if (!selected || currentScopeKey.current !== operationScopeKey) return;
      const result = await createLinkedFolder(
        { type: scopeType, id: scopeId },
        selected.token,
        selected.displayName,
      );
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
  }, [scopeKey, scopeType, scopeId, trackJob, onSourcesChanged]);

  const run = useCallback(
    async (folderId: string, mode: "sync" | "rebuild") => {
      const operationScopeKey = scopeKey;
      try {
        const { job } =
          mode === "rebuild"
            ? await rebuildLinkedFolder(folderId)
            : await syncLinkedFolder(folderId);
        if (currentScopeKey.current !== operationScopeKey) return;
        trackJob(job);
        onSourcesChanged?.();
      } catch (error) {
        toast.error(`Could not ${mode} folder`, {
          description: error instanceof Error ? error.message : String(error),
        });
      }
    },
    [scopeKey, trackJob, onSourcesChanged],
  );

  const remove = useCallback(
    async (folderId: string, removeIndex: boolean) => {
      const operationScopeKey = scopeKey;
      const previous = folders;
      setFolders((current) =>
        current.filter((folder) => folder.id !== folderId),
      );
      try {
        await deleteLinkedFolder(folderId, removeIndex);
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
        toast.error("Could not unlink folder", {
          description: error instanceof Error ? error.message : String(error),
        });
      }
    },
    [scopeKey, folders, onSourcesChanged],
  );

  return {
    folders: stateScopeKey === scopeKey ? folders : [],
    jobs: stateScopeKey === scopeKey ? jobs : {},
    loading: stateScopeKey === scopeKey ? loading : true,
    mutating,
    desktopSupported: isTauri,
    link,
    sync: (folderId: string) => run(folderId, "sync"),
    rebuild: (folderId: string) => run(folderId, "rebuild"),
    remove,
    refresh,
  };
}
