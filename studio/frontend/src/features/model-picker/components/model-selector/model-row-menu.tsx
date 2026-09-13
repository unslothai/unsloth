// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Condensed row actions for model rows so pin, update, and delete do not grow into an icon strip.
// Mirrors the sidebar chat rows' MoreVertical menu pattern.

import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { usePlatformStore } from "@/config/env";
import { revealCachedModel, revealLocalPath, getLocalDeletePreview, deleteLocalPath } from "@/features/chat";
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
  DeleteConfirmDialog,
  DeleteImpactSummary,
  UpdateConfirmDialog,
  ggufVariantsMatch,
  subscribeJobListeners,
  useDeleteImpact,
} from "@/features/hub";
import { toast } from "@/lib/toast";
import { cn } from "@/lib/utils";
import {
  Delete02Icon,
  Folder01Icon,
  MoreVerticalIcon,
  PinIcon,
  PinOffIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { RefreshCw } from "lucide-react";
import {
  type ReactNode,
  useCallback,
  useEffect,
  useRef,
  useState,
} from "react";

interface ModelRowMenuPin {
  pinned: boolean;
  /** Menu item labels, e.g. "Pin quant to the top" / "Unpin quant". */
  pinLabel: string;
  unpinLabel: string;
  onToggle: () => void;
}

interface ModelRowMenuUpdate {
  title: string;
  description: ReactNode;
  /** Repo + variant the update targets. */
  repoId: string;
  variant?: string | null;
  disabled?: boolean;
  onConfirm: () => Promise<void> | void;
  onUpdated?: () => void;
}

interface ModelRowMenuDelete {
  title: string;
  description: ReactNode;
  /** Repo (and quant) to preview the delete for, so the dialog can state what it actually reclaims
   *  and what shared assets it leaves behind. Omit to keep the plain wording. */
  impact?: { repoId: string; variant?: string | null };
  successMessage: string;
  disabled?: boolean;
  onConfirm: () => Promise<void> | void;
  onDeleted?: () => void;
}

/** Managed-cache location for "Reveal in Finder" (resolved server-side). */
interface ModelRowMenuCachePath {
  repoId: string;
  variant?: string;
}

/** Direct on-disk path for "Reveal in Finder" (custom folders, LM Studio, local models). */
interface ModelRowMenuLocalPath {
  path: string;
  displayName?: string;
}

export function ModelRowMenu({
  ariaLabel,
  buttonClassName,
  iconClassName,
  cachePath,
  localPath,
  pin,
  update,
  del,
}: {
  ariaLabel: string;
  buttonClassName?: string;
  iconClassName?: string;
  /** Enables "Reveal in Finder" for cached repos. */
  cachePath?: ModelRowMenuCachePath;
  /** Enables "Reveal in Finder" for local files/dirs. Takes precedence over cachePath when both are set. */
  localPath?: ModelRowMenuLocalPath;
  pin?: ModelRowMenuPin;
  update?: ModelRowMenuUpdate;
  del?: ModelRowMenuDelete;
}) {
  const deviceType = usePlatformStore((s) => s.deviceType);
  const revealLabel =
    deviceType === "mac" ? "Reveal in Finder" : "Reveal in Folder";
  const [deleteOpen, setDeleteOpen] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const deleteImpact = useDeleteImpact(
    deleteOpen && Boolean(del?.impact),
    del?.impact?.repoId ?? "",
    del?.impact?.variant,
  );
  const [localPreview, setLocalPreview] = useState<{
    model_files: number;
    model_bytes: number;
    other_files: number;
    other_bytes: number;
    is_dir: boolean;
  } | null>(null);
  const [localMode, setLocalMode] = useState<"model_only" | "all">("model_only");
  const [updateOpen, setUpdateOpen] = useState(false);

  useEffect(() => {
    if (!deleteOpen || !localPath?.path) {
      setLocalPreview(null);
      return;
    }
    let cancelled = false;
    getLocalDeletePreview(localPath.path, localPath.displayName)
      .then((data: { model_files: number; model_bytes: number; other_files: number; other_bytes: number; is_dir: boolean }) => {
        if (!cancelled) setLocalPreview(data);
      })
      .catch(() => {
        if (!cancelled) setLocalPreview(null);
      });
    return () => {
      cancelled = true;
    };
  }, [deleteOpen, localPath?.path, localPath?.displayName]);

  useEffect(() => {
    if (deleteOpen) setLocalMode("model_only");
  }, [deleteOpen]);

  // Refresh the caller when this repo+variant's managed update completes.
  const onUpdatedRef = useRef(update?.onUpdated);
  useEffect(() => {
    onUpdatedRef.current = update?.onUpdated;
  }, [update?.onUpdated]);
  const updateRepoId = update?.repoId;
  const updateVariant = update?.variant ?? null;
  useEffect(() => {
    if (!updateRepoId) return;
    return subscribeJobListeners("model", updateRepoId, {
      onComplete: (completedVariant) => {
        const matches = updateVariant
          ? ggufVariantsMatch(completedVariant, updateVariant)
          : !completedVariant;
        if (matches) onUpdatedRef.current?.();
      },
    });
  }, [updateRepoId, updateVariant]);

  const onDeleteConfirm = del?.onConfirm;
  const onDeleted = del?.onDeleted;
  const deleteSuccessMessage = del?.successMessage;
  const handleDeleteConfirm = useCallback(async () => {
    setDeleting(true);
    try {
      if (localPath?.path) {
        await deleteLocalPath(localPath.path, localMode, localPath.displayName);
        if (deleteSuccessMessage) toast.success(deleteSuccessMessage);
      } else {
        if (!onDeleteConfirm) return;
        await onDeleteConfirm();
        if (deleteSuccessMessage) toast.success(deleteSuccessMessage);
      }
      onDeleted?.();
      setDeleteOpen(false);
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Failed to delete model");
    } finally {
      setDeleting(false);
    }
  }, [localMode, localPath?.path, localPath?.displayName, onDeleteConfirm, onDeleted, deleteSuccessMessage]);

  const onUpdateConfirm = update?.onConfirm;
  const handleUpdateConfirm = useCallback(() => {
    // Start the re-download and close the dialog; the Downloads panel owns progress and cancel. Only
    // a failure to START toasts.
    void Promise.resolve()
      .then(onUpdateConfirm)
      .catch((err) => {
        toast.error(
          err instanceof Error ? err.message : "Failed to start update",
        );
      });
    setUpdateOpen(false);
  }, [onUpdateConfirm]);

  const cachePathRepoId = cachePath?.repoId;
  const cachePathVariant = cachePath?.variant;
  const localRevealPath = localPath?.path?.trim() || null;
  const handleReveal = useCallback(() => {
    if (localRevealPath) {
      revealLocalPath(localRevealPath).catch((err) => {
        toast.error(
          err instanceof Error ? err.message : "Failed to open file manager",
        );
      });
      return;
    }
    if (!cachePathRepoId) return;
    revealCachedModel(cachePathRepoId, cachePathVariant).catch((err) => {
      toast.error(
        err instanceof Error ? err.message : "Failed to open file manager",
      );
    });
  }, [localRevealPath, cachePathRepoId, cachePathVariant]);

  const canReveal = Boolean(localRevealPath || cachePathRepoId);
  if (!pin && !update && !del && !canReveal) return null;

  return (
    <>
      <DropdownMenu>
        <DropdownMenuTrigger asChild={true}>
          <button
            type="button"
            onClick={(e) => e.stopPropagation()}
            aria-label={ariaLabel}
            className={cn(
              // Fixed box, matching ModelLoadSettingsAction beside it.
              "flex size-5 shrink-0 items-center justify-center rounded-md text-muted-foreground/60 transition-colors hover:bg-black/5 hover:text-foreground dark:hover:bg-white/10",
              buttonClassName,
            )}
          >
            <HugeiconsIcon
              icon={MoreVerticalIcon}
              strokeWidth={1.75}
              className={cn("size-3.5", iconClassName)}
            />
          </button>
        </DropdownMenuTrigger>
        <DropdownMenuContent
          side="bottom"
          align="end"
          sideOffset={2}
          className="unsloth-plus-menu menu-flat-destructive w-48"
        >
          {pin && (
            <DropdownMenuItem
              onSelect={(e) => {
                e.stopPropagation();
                pin.onToggle();
              }}
            >
              <HugeiconsIcon
                icon={pin.pinned ? PinOffIcon : PinIcon}
                strokeWidth={1.75}
                className="size-icon"
              />
              <span>{pin.pinned ? pin.unpinLabel : pin.pinLabel}</span>
            </DropdownMenuItem>
          )}
          {canReveal && (
            <DropdownMenuItem
              onSelect={(e) => {
                e.stopPropagation();
                handleReveal();
              }}
            >
              <HugeiconsIcon
                icon={Folder01Icon}
                strokeWidth={1.75}
                className="size-icon"
              />
              <span>{revealLabel}</span>
            </DropdownMenuItem>
          )}
          {update && (
            <DropdownMenuItem
              disabled={update.disabled}
              onSelect={(e) => {
                e.stopPropagation();
                setUpdateOpen(true);
              }}
            >
              <RefreshCw className="size-icon" />
              <span>Update</span>
            </DropdownMenuItem>
          )}
          {del && (
            <>
              {(canReveal || pin || update) && <DropdownMenuSeparator />}
              <DropdownMenuItem
                variant="destructive"
                disabled={del.disabled}
                onSelect={(e) => {
                  e.stopPropagation();
                  setDeleteOpen(true);
                }}
              >
                <HugeiconsIcon
                  icon={Delete02Icon}
                  strokeWidth={1.75}
                  className="size-icon"
                />
                <span>Delete</span>
              </DropdownMenuItem>
            </>
          )}
        </DropdownMenuContent>
      </DropdownMenu>

      {del && localPath ? (
        (() => {
          const fmt = (b: number) => {
            if (!Number.isFinite(b) || b <= 0) return "0 B";
            const u = ["B", "KB", "MB", "GB", "TB"];
            let i = 0;
            let v = b;
            while (v >= 1024 && i < u.length - 1) {
              v /= 1024;
              i += 1;
            }
            return `${v.toFixed(v < 10 ? 1 : 0)} ${u[i]}`;
          };
          const showChoice = Boolean(
            localPreview?.is_dir && (localPreview.other_files > 0 || localPreview.model_files > 1),
          );
          return (
            <AlertDialog
              open={deleteOpen}
              onOpenChange={(nextOpen) => {
                if (!nextOpen && deleting) return;
                setDeleteOpen(nextOpen);
              }}
            >
              <AlertDialogContent>
                <AlertDialogHeader>
                  <AlertDialogTitle>{del.title}</AlertDialogTitle>
                  <AlertDialogDescription asChild>
                    <div className="space-y-3">
                      <div>{del.description}</div>
                      {showChoice && localPreview ? (
                        <div className="rounded-md border border-border/60 bg-muted/20 p-3 text-sm">
                          <p className="text-muted-foreground">
                            This folder contains {localPreview.model_files} model file
                            {localPreview.model_files === 1 ? "" : "s"} ({fmt(localPreview.model_bytes)}) and{" "}
                            {localPreview.other_files} other file{localPreview.other_files === 1 ? "" : "s"} (
                            {fmt(localPreview.other_bytes)}).
                          </p>
                          <div className="mt-3 space-y-2">
                            <label className="flex items-start gap-2">
                              <input
                                type="radio"
                                name="local-delete-mode"
                                checked={localMode === "model_only"}
                                onChange={() => setLocalMode("model_only")}
                                className="mt-1"
                              />
                              <span>
                                <span className="font-medium">Delete model files only</span>
                                <span className="block text-xs text-muted-foreground">Other files are kept.</span>
                              </span>
                            </label>
                            <label className="flex items-start gap-2">
                              <input
                                type="radio"
                                name="local-delete-mode"
                                checked={localMode === "all"}
                                onChange={() => setLocalMode("all")}
                                className="mt-1"
                              />
                              <span>
                                <span className="font-medium">Delete everything in this folder</span>
                                <span className="block text-xs text-destructive">This cannot be undone.</span>
                              </span>
                            </label>
                          </div>
                          <p className="mt-2 text-xs text-muted-foreground">
                            The folder also contains documents, images, or other content that is not part of this model.
                          </p>
                        </div>
                      ) : null}
                    </div>
                  </AlertDialogDescription>
                </AlertDialogHeader>
                <AlertDialogFooter>
                  <AlertDialogCancel disabled={deleting}>Cancel</AlertDialogCancel>
                  <AlertDialogAction
                    onClick={() => void handleDeleteConfirm()}
                    disabled={deleting}
                    className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
                  >
                    {deleting ? <Spinner className="size-4" /> : "Delete"}
                  </AlertDialogAction>
                </AlertDialogFooter>
              </AlertDialogContent>
            </AlertDialog>
          );
        })()
      ) : del ? (
        <DeleteConfirmDialog
          open={deleteOpen}
          onOpenChange={(nextOpen) => {
            if (!nextOpen && deleting) return;
            setDeleteOpen(nextOpen);
          }}
          title={del.title}
          description={
            <>
              {del.description}
              <DeleteImpactSummary impact={deleteImpact} />
            </>
          }
          deleting={deleting}
          blocked={(deleteImpact?.blocked_by.length ?? 0) > 0}
          onConfirm={() => void handleDeleteConfirm()}
        />
      ) : null}

      {update && (
        <UpdateConfirmDialog
          open={updateOpen}
          onOpenChange={setUpdateOpen}
          title={update.title}
          description={update.description}
          updating={false}
          onConfirm={handleUpdateConfirm}
        />
      )}
    </>
  );
}
