// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Condensed row actions for model rows: everything except the run-settings
// gear collapses into one dots menu (pin, update, delete) so rows don't grow
// an icon strip. Mirrors the sidebar chat rows' MoreVertical menu pattern.

import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { usePlatformStore } from "@/config/env";
import { revealCachedModel } from "@/features/chat";
import {
  DeleteConfirmDialog,
  UpdateConfirmDialog,
  ggufVariantsMatch,
  subscribeJobListeners,
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
  /** Repo + variant the update targets (see ModelUpdateAction). */
  repoId: string;
  variant?: string | null;
  disabled?: boolean;
  onConfirm: () => Promise<void> | void;
  onUpdated?: () => void;
}

interface ModelRowMenuDelete {
  title: string;
  description: ReactNode;
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

export function ModelRowMenu({
  ariaLabel,
  buttonClassName,
  iconClassName,
  cachePath,
  pin,
  update,
  del,
}: {
  ariaLabel: string;
  buttonClassName?: string;
  iconClassName?: string;
  /** Enables "Reveal in Finder" for cached repos. */
  cachePath?: ModelRowMenuCachePath;
  pin?: ModelRowMenuPin;
  update?: ModelRowMenuUpdate;
  del?: ModelRowMenuDelete;
}) {
  const deviceType = usePlatformStore((s) => s.deviceType);
  const revealLabel =
    deviceType === "mac"
      ? "Reveal in Finder"
      : deviceType === "windows"
        ? "Reveal in File Explorer"
        : "Reveal in File Manager";
  const [deleteOpen, setDeleteOpen] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [updateOpen, setUpdateOpen] = useState(false);

  // Refresh the caller when this repo+variant's managed update completes
  // (mirrors ModelUpdateAction).
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
    if (!onDeleteConfirm) return;
    setDeleting(true);
    try {
      await onDeleteConfirm();
      if (deleteSuccessMessage) toast.success(deleteSuccessMessage);
      onDeleted?.();
      setDeleteOpen(false);
    } catch (err) {
      toast.error(
        err instanceof Error ? err.message : "Failed to delete model",
      );
    } finally {
      setDeleting(false);
    }
  }, [onDeleteConfirm, onDeleted, deleteSuccessMessage]);

  const onUpdateConfirm = update?.onConfirm;
  const handleUpdateConfirm = useCallback(() => {
    // Start the re-download and close the dialog; the Downloads panel owns
    // progress + cancel. Only a failure to START toasts.
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
  const handleReveal = useCallback(() => {
    if (!cachePathRepoId) return;
    revealCachedModel(cachePathRepoId, cachePathVariant).catch((err) => {
      toast.error(
        err instanceof Error ? err.message : "Failed to open file manager",
      );
    });
  }, [cachePathRepoId, cachePathVariant]);

  if (!pin && !update && !del && !cachePath) return null;

  return (
    <>
      <DropdownMenu>
        <DropdownMenuTrigger asChild={true}>
          <button
            type="button"
            onClick={(e) => e.stopPropagation()}
            aria-label={ariaLabel}
            className={cn(
              "shrink-0 rounded-md p-1 text-muted-foreground/60 transition-colors hover:bg-black/5 hover:text-foreground dark:hover:bg-white/10",
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
          {cachePath && (
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
              {(cachePath || pin || update) && <DropdownMenuSeparator />}
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

      {del && (
        <DeleteConfirmDialog
          open={deleteOpen}
          onOpenChange={(nextOpen) => {
            if (!nextOpen && deleting) return;
            setDeleteOpen(nextOpen);
          }}
          title={del.title}
          description={del.description}
          deleting={deleting}
          onConfirm={() => void handleDeleteConfirm()}
        />
      )}

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
