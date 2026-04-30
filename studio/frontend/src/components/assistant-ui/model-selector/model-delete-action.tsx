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
import { Trash2Icon } from "lucide-react";
import { useCallback, useState, type ReactNode } from "react";
import { toast } from "sonner";

interface ModelDeleteActionProps {
  ariaLabel: string;
  title: string;
  description: ReactNode;
  successMessage: string;
  loadingLabel?: string;
  buttonClassName?: string;
  iconClassName?: string;
  onConfirm: () => Promise<void> | void;
  onDeleted?: () => void;
}

export function ModelDeleteAction({
  ariaLabel,
  title,
  description,
  successMessage,
  loadingLabel = "Deleting...",
  buttonClassName,
  iconClassName,
  onConfirm,
  onDeleted,
}: ModelDeleteActionProps) {
  const [open, setOpen] = useState(false);
  const [deleting, setDeleting] = useState(false);

  const handleConfirm = useCallback(async () => {
    setDeleting(true);
    try {
      await onConfirm();
      toast.success(successMessage);
      onDeleted?.();
      setOpen(false);
    } catch (err) {
      toast.error(
        err instanceof Error ? err.message : "Failed to delete model",
      );
    } finally {
      setDeleting(false);
    }
  }, [onConfirm, onDeleted, successMessage]);

  return (
    <>
      <button
        type="button"
        onClick={(e) => {
          e.stopPropagation();
          setOpen(true);
        }}
        aria-label={ariaLabel}
        className={cn(
          "shrink-0 rounded-md p-1.5 text-muted-foreground/60 transition-colors hover:bg-destructive/10 hover:text-destructive",
          buttonClassName,
        )}
      >
        <Trash2Icon className={cn("size-3.5", iconClassName)} />
      </button>

      <AlertDialog
        open={open}
        onOpenChange={(nextOpen) => {
          if (!nextOpen && deleting) return;
          setOpen(nextOpen);
        }}
      >
        <AlertDialogContent size="sm">
          <AlertDialogHeader>
            <AlertDialogTitle>{title}</AlertDialogTitle>
            <AlertDialogDescription>{description}</AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel disabled={deleting}>No</AlertDialogCancel>
            <AlertDialogAction
              variant="destructive"
              disabled={deleting}
              onClick={(e) => {
                e.preventDefault();
                handleConfirm();
              }}
            >
              {deleting ? loadingLabel : "Yes"}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </>
  );
}
