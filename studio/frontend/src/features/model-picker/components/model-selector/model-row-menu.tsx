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
  Settings02Icon,
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

import { useRowActive } from "./row-activation";

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
  /** Repo (and quant) to preview the delete for, so the dialog can state what it actually
   * reclaims and what shared assets it leaves behind. Omit to keep the plain wording. */
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

/** The model's settings page: load config plus what the API will apply. */
interface ModelRowMenuSettings {
  onOpen: () => void;
}

type ModelRowMenuProps = {
  ariaLabel: string;
  buttonClassName?: string;
  iconClassName?: string;
  /** Enables "Reveal in Finder" for cached repos. */
  cachePath?: ModelRowMenuCachePath;
  settings?: ModelRowMenuSettings;
  pin?: ModelRowMenuPin;
  update?: ModelRowMenuUpdate;
  del?: ModelRowMenuDelete;
};

/** The dots button exactly as Radix renders it, for a row nothing has reached yet.
 *
 *  It sits under `opacity-0` until the row is hovered or focused, so there is nothing on screen to
 *  keep in sync; what has to match is the a11y tree and the tab order, hence the same label, the
 *  same `aria-haspopup` / `aria-expanded` / `data-state` a closed Radix trigger carries, and a real
 *  focusable <button> in the same slot. */
function ModelRowMenuPlaceholder({
  ariaLabel,
  buttonClassName,
  iconClassName,
  onActivate,
}: {
  ariaLabel: string;
  buttonClassName?: string;
  iconClassName?: string;
  onActivate: () => void;
}) {
  return (
    <button
      type="button"
      aria-haspopup="menu"
      aria-expanded={false}
      data-state="closed"
      data-slot="dropdown-menu-trigger"
      onPointerDown={(e) => {
        e.stopPropagation();
        onActivate();
      }}
      onClick={(e) => {
        e.stopPropagation();
        onActivate();
      }}
      aria-label={ariaLabel}
      className={cn(
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
  );
}

/**
 * The row's dots menu, mounted only once something has reached the row.
 *
 * A menu that is closed shows nothing, and on an On Device row the trigger itself is invisible
 * until `group-hover` / `focus-within` lifts it, yet the merge base paid for a Radix root, a
 * portal, a Presence state machine, two AlertDialogs and a platform-store subscription for every
 * cached repo the user owns. `useRowActive()` is false only inside a `ModelRowShell` that no
 * pointer or focus has reached; everywhere else (the Hub catalog, and any row outside a shell) it
 * is true and this renders exactly what it always did.
 */
export function ModelRowMenu(props: ModelRowMenuProps) {
  const { ariaLabel, buttonClassName, iconClassName, pin, update, cachePath, settings, del } =
    props;
  const rowActive = useRowActive();
  const [open, setOpen] = useState(false);
  const [activated, setActivated] = useState(false);

  // Kept in the shell, not in the body: this is how a row learns that a managed update it started
  // has finished, and a subscription that only exists while the menu is mounted would miss the
  // completion of the very download it kicked off.
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

  if (!pin && !update && !del && !cachePath && !settings) return null;

  if (!rowActive && !activated) {
    return (
      <ModelRowMenuPlaceholder
        ariaLabel={ariaLabel}
        buttonClassName={buttonClassName}
        iconClassName={iconClassName}
        onActivate={() => {
          // A press that lands on the placeholder is a press that wanted the menu: Radix's own
          // trigger opens on pointerdown, so open on mount rather than swallowing the gesture.
          setActivated(true);
          setOpen(true);
        }}
      />
    );
  }

  return (
    <ModelRowMenuLive
      {...props}
      open={open}
      onOpenChange={setOpen}
      onUpdatedRef={onUpdatedRef}
    />
  );
}

function ModelRowMenuLive({
  ariaLabel,
  buttonClassName,
  iconClassName,
  cachePath,
  settings,
  pin,
  update,
  del,
  open,
  onOpenChange,
}: ModelRowMenuProps & {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  /** Owned by the shell so the update subscription outlives this component. */
  onUpdatedRef: { current: (() => void) | undefined };
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
  const [updateOpen, setUpdateOpen] = useState(false);

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

  return (
    <>
      <DropdownMenu open={open} onOpenChange={onOpenChange}>
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
          {settings && (
            <DropdownMenuItem
              onSelect={(e) => {
                e.stopPropagation();
                settings.onOpen();
              }}
            >
              <HugeiconsIcon
                icon={Settings02Icon}
                strokeWidth={1.75}
                className="size-icon"
              />
              <span>Settings</span>
            </DropdownMenuItem>
          )}
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
