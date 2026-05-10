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
import { cn } from "@/lib/utils";
import { RefreshCw } from "lucide-react";
import { useCallback, useState, type ReactNode } from "react";
import { toast } from "sonner";

interface ModelUpdateActionProps {
  ariaLabel: string;
  title: string;
  description: ReactNode;
  successMessage: string;
  loadingLabel?: string;
  buttonClassName?: string;
  iconClassName?: string;
  disabled?: boolean;
  onConfirm: () => Promise<void> | void;
  onUpdated?: () => void;
}

export function ModelUpdateAction({
  ariaLabel,
  title,
  description,
  successMessage,
  loadingLabel = "Updating...",
  buttonClassName,
  iconClassName,
  disabled = false,
  onConfirm,
  onUpdated,
}: ModelUpdateActionProps) {
  const [open, setOpen] = useState(false);
  const [updating, setUpdating] = useState(false);

  const handleConfirm = useCallback(async () => {
    setUpdating(true);
    try {
      setOpen(false);
      await onConfirm();
      toast.success(successMessage);
      onUpdated?.();
    } catch (err) {
      toast.error(
        err instanceof Error ? err.message : "Failed to update model",
      );
    } finally {
      setUpdating(false);
    }
  }, [onConfirm, onUpdated, successMessage]);

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
          "shrink-0 rounded-md p-1.5 text-muted-foreground/60 transition-colors hover:bg-yellow-400/10 hover:text-yellow-400",
          disabled && "cursor-not-allowed opacity-40 hover:bg-transparent hover:text-muted-foreground/60",
          buttonClassName,
        )}
      >
        <RefreshCw className={cn("size-3.5", iconClassName)} />
      </button>

      <AlertDialog
        open={open}
        onOpenChange={(nextOpen) => {
          if (!nextOpen && updating) return;
          setOpen(nextOpen);
        }}
      >
        <AlertDialogContent size="sm">
          <AlertDialogHeader>
            <AlertDialogTitle>{title}</AlertDialogTitle>
            <AlertDialogDescription>{description}</AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel disabled={updating}>No</AlertDialogCancel>
            <AlertDialogAction
              variant="default"
              disabled={updating}
              onClick={(e) => {
                e.preventDefault();
                handleConfirm();
              }}
            >
              {updating ? loadingLabel : "Yes"}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </>
  );
}
