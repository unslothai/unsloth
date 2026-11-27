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
import { toast } from "@/lib/toast";
import {
  LOAD_ANYWAY_CONFIRM_TITLE,
  LOAD_ANYWAY_LABEL,
  LOAD_ANYWAY_WARNING,
  UNIFIED_MEMORY_LOAD_TITLE,
} from "@/lib/unified-memory-load-refusal";
import { type ReactNode, useCallback, useRef, useState } from "react";

interface PendingRetry {
  message: string;
  retry: () => void;
}

/**
 * The toast action only opens a confirmation; only Confirm retries, and dismiss retires the prompt.
 */
export function useLoadAnywayPrompt(): {
  offer: (message: string, retry: () => void) => void;
  dismiss: () => void;
  dialog: ReactNode;
} {
  const [pending, setPending] = useState<PendingRetry | null>(null);
  const toastId = useRef<string | number | null>(null);
  const dismiss = useCallback(() => {
    if (toastId.current !== null) {
      toast.dismiss(toastId.current);
    }
    toastId.current = null;
    setPending(null);
  }, []);
  const offer = useCallback((message: string, retry: () => void) => {
    toastId.current = toast.error(UNIFIED_MEMORY_LOAD_TITLE, {
      description: message,
      duration: 20_000,
      action: {
        label: LOAD_ANYWAY_LABEL,
        onClick: () => setPending({ message, retry }),
      },
    });
  }, []);
  const confirm = () => {
    const current = pending;
    setPending(null);
    current?.retry();
  };
  const dialog = (
    <AlertDialog
      open={pending !== null}
      onOpenChange={(open) => !open && setPending(null)}
    >
      <AlertDialogContent onOverlayClick={() => setPending(null)}>
        <AlertDialogHeader>
          <AlertDialogTitle>{LOAD_ANYWAY_CONFIRM_TITLE}</AlertDialogTitle>
          <AlertDialogDescription className="break-words">
            {LOAD_ANYWAY_WARNING}
          </AlertDialogDescription>
          {pending ? (
            <p className="break-words text-xs text-muted-foreground">
              {pending.message}
            </p>
          ) : null}
        </AlertDialogHeader>
        <AlertDialogFooter>
          <AlertDialogCancel onClick={() => setPending(null)}>
            Cancel
          </AlertDialogCancel>
          <AlertDialogAction
            variant="destructive"
            onClick={confirm}
            data-testid="load-anyway-confirm"
          >
            {LOAD_ANYWAY_LABEL}
          </AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  );
  return { offer, dismiss, dialog };
}
