// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

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
import { subscribeJobListeners } from "@/features/hub/download-manager";
import { ggufVariantsMatch } from "@/features/hub/lib/model-identity";
import { cn } from "@/lib/utils";
import { RefreshCw } from "lucide-react";
import { useCallback, useEffect, useRef, useState, type ReactNode } from "react";
import { toast } from "sonner";

interface ModelUpdateActionProps {
  ariaLabel: string;
  title: string;
  description: ReactNode;
  /** Repo + variant the update targets, so this action can refresh its caller
   *  (clearing the "update available" cue) when the matching managed download
   *  completes. `variant` is null for full-model (safetensors / MLX) rows. */
  repoId: string;
  variant?: string | null;
  buttonClassName?: string;
  iconClassName?: string;
  disabled?: boolean;
  /** Starts the update — which now runs as a managed download. Resolves once the
   *  download has been handed to the download manager, NOT when it finishes. */
  onConfirm: () => Promise<void> | void;
  /** Fired when THIS repo+variant's managed update actually completes. */
  onUpdated?: () => void;
}

export function ModelUpdateAction({
  ariaLabel,
  title,
  description,
  repoId,
  variant = null,
  buttonClassName,
  iconClassName,
  disabled = false,
  onConfirm,
  onUpdated,
}: ModelUpdateActionProps) {
  const [open, setOpen] = useState(false);

  // The update is a managed download (it surfaces in the global Downloads panel
  // with progress + cancel). When this exact repo+variant finishes, refresh the
  // caller so the "update available" cue clears once the new revision is on
  // disk. A ref keeps the subscription stable across renders without resubscribing.
  const onUpdatedRef = useRef(onUpdated);
  onUpdatedRef.current = onUpdated;
  useEffect(() => {
    return subscribeJobListeners("model", repoId, {
      onComplete: (completedVariant) => {
        const matches = variant
          ? ggufVariantsMatch(completedVariant, variant)
          : !completedVariant;
        if (matches) onUpdatedRef.current?.();
      },
    });
  }, [repoId, variant]);

  const handleConfirm = useCallback(() => {
    // Start the background re-download and close the dialog immediately; the
    // Downloads panel owns progress + cancel from here. Only a failure to START
    // surfaces a toast — a failed download reports itself in the panel.
    void Promise.resolve()
      .then(onConfirm)
      .catch((err) => {
        toast.error(
          err instanceof Error ? err.message : "Failed to start update",
        );
      });
    setOpen(false);
  }, [onConfirm]);

  return (
    <>
      <button
        type="button"
        onClick={(e) => {
          e.stopPropagation();
          if (disabled) return;
          setOpen(true);
        }}
        aria-label={ariaLabel}
        disabled={disabled}
        className={cn(
          "shrink-0 rounded-md p-1.5 text-muted-foreground/60 transition-colors hover:bg-amber-500/10 hover:text-amber-700 dark:hover:bg-amber-500/15 dark:hover:text-amber-300",
          disabled && "cursor-not-allowed opacity-40 hover:bg-transparent hover:text-muted-foreground/60",
          buttonClassName,
        )}
      >
        <RefreshCw className={cn("size-3.5", iconClassName)} />
      </button>

      <AlertDialog open={open} onOpenChange={setOpen}>
        <AlertDialogContent size="sm">
          <AlertDialogHeader>
            <AlertDialogTitle>{title}</AlertDialogTitle>
            <AlertDialogDescription>{description}</AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>No</AlertDialogCancel>
            <AlertDialogAction
              variant="default"
              onClick={(e) => {
                e.preventDefault();
                handleConfirm();
              }}
            >
              Yes
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </>
  );
}
