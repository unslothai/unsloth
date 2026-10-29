// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Condensed row actions so pin, update and delete do not grow into an icon strip.

import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { revealCachedModel } from "@/features/chat";
import {
  DeleteConfirmDialog,
  DeleteImpactSummary,
  UpdateConfirmDialog,
  ggufVariantsMatch,
  subscribeJobListeners,
  useDeleteImpact,
} from "@/features/hub";
import { useRevealLabel } from "@/features/library";
import { toast } from "@/lib/toast";
import { cn } from "@/lib/utils";
import { RefreshGlyph } from "@/lib/refresh-icon";
import {
  Delete02Icon,
  Folder01Icon,
  MoreVerticalIcon,
  PinIcon,
  PinOffIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import {
  type ReactNode,
  useCallback,
  useEffect,
  useRef,
  useState,
} from "react";

/** Rendered under the pin and above cache/update, so delete stays last. */
export interface ModelRowMenuItem {
  key: string;
  label: string;
  icon: ReactNode;
  onSelect: () => void;
  disabled?: boolean;
}

interface ModelRowMenuPin {
  pinned: boolean;
  pinLabel: string;
  unpinLabel: string;
  onToggle: () => void;
}

interface ModelRowMenuUpdate {
  title: string;
  description: ReactNode;
  repoId: string;
  variant?: string | null;
  disabled?: boolean;
  onConfirm: () => Promise<void> | void;
  onUpdated?: () => void;
}

interface ModelRowMenuDelete {
  title: string;
  description: ReactNode;
  /** Lets the delete dialog state what it reclaims and what shared assets remain. */
  impact?: { repoId: string; variant?: string | null; cachePath?: string | null };
  successMessage: string;
  disabled?: boolean;
  onConfirm: () => Promise<void> | void;
  onDeleted?: () => void;
}

interface ModelRowMenuCachePath {
  repoId: string;
  variant?: string;
}

export function ModelRowMenu({
  ariaLabel,
  buttonClassName,
  iconClassName,
  cachePath,
  onReveal,
  pin,
  items,
  update,
  del,
}: {
  ariaLabel: string;
  buttonClassName?: string;
  iconClassName?: string;
  cachePath?: ModelRowMenuCachePath;
  onReveal?: () => Promise<void>;
  pin?: ModelRowMenuPin;
  items?: readonly ModelRowMenuItem[];
  update?: ModelRowMenuUpdate;
  del?: ModelRowMenuDelete;
}) {
  // Null unless the owner is on the backend's own machine with a file manager.
  const revealLabel = useRevealLabel();
  const [deleteOpen, setDeleteOpen] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const deleteImpact = useDeleteImpact(
    deleteOpen && Boolean(del?.impact),
    del?.impact?.repoId ?? "",
    del?.impact?.variant,
    del?.impact?.cachePath,
  );
  const [updateOpen, setUpdateOpen] = useState(false);

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
    // The Downloads panel owns progress and cancel; only a failure to start toasts.
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
    const reveal = onReveal
      ? onReveal()
      : cachePathRepoId
        ? revealCachedModel(cachePathRepoId, cachePathVariant)
        : null;
    reveal?.catch((err) => {
      toast.error(
        err instanceof Error ? err.message : "Failed to open file manager",
      );
    });
  }, [onReveal, cachePathRepoId, cachePathVariant]);

  const canReveal = Boolean(revealLabel && (cachePath || onReveal));
  if (!pin && !update && !del && !canReveal && !items?.length) return null;

  return (
    <>
      <DropdownMenu>
        <DropdownMenuTrigger asChild={true}>
          <button
            type="button"
            onClick={(e) => e.stopPropagation()}
            aria-label={ariaLabel}
            className={cn(
              "flex size-5 shrink-0 items-center justify-center rounded-md text-muted-foreground/80 transition-colors hover:bg-[rgb(0_0_0_/_calc(0.05*var(--contrast-wash-gain,1)))] hover:text-foreground dark:hover:bg-[rgb(255_255_255_/_calc(0.1*var(--contrast-wash-gain,1)))]",
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
          {items?.map((item) => (
            <DropdownMenuItem
              key={item.key}
              disabled={item.disabled}
              onSelect={(e) => {
                e.stopPropagation();
                item.onSelect();
              }}
            >
              {item.icon}
              <span>{item.label}</span>
            </DropdownMenuItem>
          ))}
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
              <RefreshGlyph className="size-icon" />
              <span>Update</span>
            </DropdownMenuItem>
          )}
          {del && (
            <>
              {(canReveal || pin || update || items?.length) && (
                <DropdownMenuSeparator />
              )}
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
