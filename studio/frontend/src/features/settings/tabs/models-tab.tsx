// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { Spinner } from "@/components/ui/spinner";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  type CachedGgufRepo,
  type CachedModelRepo,
  type ModelUpdateInfo,
  checkModelUpdates,
  deleteCachedModel,
  listCachedGguf,
  listCachedModels,
  updateCachedModel,
} from "@/features/chat/api/chat-api";
import { toast } from "@/lib/toast";
import { cn } from "@/lib/utils";
import {
  Delete02Icon,
  Download04Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { RefreshCwIcon } from "lucide-react";
import { useCallback, useEffect, useMemo, useState } from "react";
import { SettingsSection } from "../components/settings-section";

/** Format bytes to a human-readable size string. */
function formatBytes(bytes: number): string {
  if (bytes === 0) return "0 B";
  const units = ["B", "KB", "MB", "GB", "TB"];
  const i = Math.floor(Math.log(bytes) / Math.log(1024));
  const value = bytes / 1024 ** i;
  return `${value.toFixed(value < 10 ? 1 : 0)} ${units[i]}`;
}

interface CachedModel {
  repo_id: string;
  size_bytes: number;
  type: "gguf" | "model";
  cache_path?: string;
}

export function ModelsTab() {
  const [cachedGguf, setCachedGguf] = useState<CachedGgufRepo[]>([]);
  const [cachedModels, setCachedModels] = useState<CachedModelRepo[]>([]);
  const [updates, setUpdates] = useState<Map<string, ModelUpdateInfo>>(
    new Map(),
  );
  const [loading, setLoading] = useState(true);
  const [checkingUpdates, setCheckingUpdates] = useState(false);
  const [deletingModel, setDeletingModel] = useState<string | null>(null);
  const [updatingModel, setUpdatingModel] = useState<string | null>(null);
  const [confirmDelete, setConfirmDelete] = useState<CachedModel | null>(null);

  const allModels = useMemo<CachedModel[]>(() => {
    const models: CachedModel[] = [];
    for (const g of cachedGguf) {
      models.push({
        repo_id: g.repo_id,
        size_bytes: g.size_bytes,
        type: "gguf",
        cache_path: g.cache_path,
      });
    }
    for (const m of cachedModels) {
      models.push({
        repo_id: m.repo_id,
        size_bytes: m.size_bytes,
        type: "model",
      });
    }
    return models.sort((a, b) => a.repo_id.localeCompare(b.repo_id));
  }, [cachedGguf, cachedModels]);

  const totalSize = useMemo(
    () => allModels.reduce((sum, m) => sum + m.size_bytes, 0),
    [allModels],
  );

  const refreshModels = useCallback(async () => {
    setLoading(true);
    try {
      const [gguf, models] = await Promise.all([
        listCachedGguf(),
        listCachedModels(),
      ]);
      setCachedGguf(gguf);
      setCachedModels(models);
    } catch (err) {
      toast.error(
        err instanceof Error ? err.message : "Failed to load cached models",
      );
    } finally {
      setLoading(false);
    }
  }, []);

  const checkForUpdates = useCallback(async () => {
    if (allModels.length === 0) return;
    setCheckingUpdates(true);
    try {
      const repoIds = allModels.map((m) => m.repo_id);
      const updateInfos = await checkModelUpdates(repoIds);
      const updateMap = new Map<string, ModelUpdateInfo>();
      for (const info of updateInfos) {
        updateMap.set(info.repo_id.toLowerCase(), info);
      }
      setUpdates(updateMap);
      if (updateInfos.length > 0) {
        toast.success(`${updateInfos.length} model(s) have updates available`);
      } else {
        toast.success("All models are up to date");
      }
    } catch (err) {
      toast.error(
        err instanceof Error ? err.message : "Failed to check for updates",
      );
    } finally {
      setCheckingUpdates(false);
    }
  }, [allModels]);

  const handleDelete = useCallback(
    async (model: CachedModel) => {
      setDeletingModel(model.repo_id);
      try {
        await deleteCachedModel(model.repo_id);
        toast.success(`Deleted ${model.repo_id}`);
        await refreshModels();
      } catch (err) {
        toast.error(
          err instanceof Error ? err.message : "Failed to delete model",
        );
      } finally {
        setDeletingModel(null);
        setConfirmDelete(null);
      }
    },
    [refreshModels],
  );

  const handleUpdate = useCallback(
    async (model: CachedModel) => {
      setUpdatingModel(model.repo_id);
      try {
        const result = await updateCachedModel(model.repo_id);
        toast.success(result.message);
        // Remove from updates map since we've processed it
        setUpdates((prev) => {
          const next = new Map(prev);
          next.delete(model.repo_id.toLowerCase());
          return next;
        });
        await refreshModels();
      } catch (err) {
        toast.error(
          err instanceof Error ? err.message : "Failed to update model",
        );
      } finally {
        setUpdatingModel(null);
      }
    },
    [refreshModels],
  );

  useEffect(() => {
    refreshModels();
  }, [refreshModels]);

  // Auto-check for updates after models are loaded
  useEffect(() => {
    if (!loading && allModels.length > 0 && updates.size === 0) {
      checkForUpdates();
    }
  }, [loading, allModels.length]);

  return (
    <div className="flex flex-col gap-6">
      <header className="flex flex-col gap-1">
        <h1 className="text-lg font-semibold font-heading">Model Manager</h1>
        <p className="text-xs text-muted-foreground">
          Manage your downloaded models. Delete unused models to free up disk
          space, or check for updates to get the latest versions.
        </p>
      </header>

      <SettingsSection
        title="Downloaded Models"
        description={
          loading
            ? "Loading..."
            : `${allModels.length} model(s), ${formatBytes(totalSize)} total`
        }
      >
        <div className="flex items-center gap-2 py-3">
          <Button
            variant="outline"
            size="sm"
            onClick={refreshModels}
            disabled={loading}
          >
            <RefreshCwIcon
              className={cn("size-3.5 mr-1.5", loading && "animate-spin")}
            />
            Refresh
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={checkForUpdates}
            disabled={checkingUpdates || loading || allModels.length === 0}
          >
            <RefreshCwIcon
              className={cn("size-3.5 mr-1.5", checkingUpdates && "animate-spin")}
            />
            Check for Updates
          </Button>
        </div>

        {loading ? (
          <div className="flex items-center justify-center py-8">
            <Spinner className="size-5" />
          </div>
        ) : allModels.length === 0 ? (
          <div className="py-8 text-center text-sm text-muted-foreground">
            No downloaded models found.
          </div>
        ) : (
          <div className="flex flex-col divide-y divide-border/40">
            {allModels.map((model) => {
              const updateInfo = updates.get(model.repo_id.toLowerCase());
              const hasUpdate = updateInfo?.has_update ?? false;
              const isDeleting = deletingModel === model.repo_id;
              const isUpdating = updatingModel === model.repo_id;

              return (
                <div
                  key={model.repo_id}
                  className="flex items-center gap-3 py-3"
                >
                  <div className="flex min-w-0 flex-1 flex-col gap-0.5">
                    <div className="flex items-center gap-2">
                      <span className="min-w-0 truncate text-sm font-medium">
                        {model.repo_id}
                      </span>
                      {hasUpdate && (
                        <Tooltip>
                          <TooltipTrigger asChild>
                            <span className="flex shrink-0 items-center gap-1 rounded-full bg-amber-500/10 px-2 py-0.5 text-[10px] font-medium text-amber-600 dark:text-amber-400">
                              <HugeiconsIcon
                                icon={Download04Icon}
                                className="size-3"
                              />
                              Update available
                            </span>
                          </TooltipTrigger>
                          <TooltipContent side="top" className="max-w-xs">
                            <p>
                              A newer version is available on Hugging Face.
                            </p>
                            <p className="mt-1 text-[10px] text-muted-foreground">
                              Latest: {updateInfo?.latest_commit?.slice(0, 7)}
                            </p>
                          </TooltipContent>
                        </Tooltip>
                      )}
                    </div>
                    <div className="flex items-center gap-2 text-xs text-muted-foreground">
                      <span
                        className={cn(
                          "rounded px-1.5 py-0.5 text-[10px] font-medium uppercase",
                          model.type === "gguf"
                            ? "bg-emerald-500/10 text-emerald-600 dark:text-emerald-400"
                            : "bg-blue-500/10 text-blue-600 dark:text-blue-400",
                        )}
                      >
                        {model.type}
                      </span>
                      <span>{formatBytes(model.size_bytes)}</span>
                    </div>
                  </div>

                  <div className="flex shrink-0 items-center gap-1.5">
                    {hasUpdate && (
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => handleUpdate(model)}
                        disabled={isUpdating || isDeleting}
                        className="h-7 px-2 text-xs"
                      >
                        {isUpdating ? (
                          <Spinner className="size-3 mr-1" />
                        ) : (
                          <HugeiconsIcon
                            icon={Download04Icon}
                            className="size-3 mr-1"
                          />
                        )}
                        Update
                      </Button>
                    )}
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => setConfirmDelete(model)}
                      disabled={isDeleting || isUpdating}
                      className="h-7 px-2 text-xs text-muted-foreground hover:text-destructive"
                    >
                      {isDeleting ? (
                        <Spinner className="size-3" />
                      ) : (
                        <HugeiconsIcon icon={Delete02Icon} className="size-3" />
                      )}
                    </Button>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </SettingsSection>

      <SettingsSection title="Tips">
        <div className="py-3 text-xs text-muted-foreground leading-relaxed">
          <ul className="list-disc pl-4 space-y-1.5">
            <li>
              <strong>GGUF models</strong> are quantized models optimized for
              CPU/GPU inference with llama.cpp.
            </li>
            <li>
              <strong>Regular models</strong> are full-precision Hugging Face
              models used for training.
            </li>
            <li>
              Deleting a model frees up disk space. You can re-download it later
              when needed.
            </li>
            <li>
              Updating a model will delete the old version and prepare for
              downloading the latest version on next load.
            </li>
          </ul>
        </div>
      </SettingsSection>

      <AlertDialog
        open={confirmDelete !== null}
        onOpenChange={(open) => !open && setConfirmDelete(null)}
      >
        <AlertDialogContent size="sm">
          <AlertDialogHeader>
            <AlertDialogTitle>Delete cached model?</AlertDialogTitle>
            <AlertDialogDescription>
              This will remove{" "}
              <span className="font-medium text-foreground">
                {confirmDelete?.repo_id}
              </span>{" "}
              ({formatBytes(confirmDelete?.size_bytes ?? 0)}) from disk. You can
              re-download it later.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction
              variant="destructive"
              onClick={() => confirmDelete && handleDelete(confirmDelete)}
            >
              Delete
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
}
