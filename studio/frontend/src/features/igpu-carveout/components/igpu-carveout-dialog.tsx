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
import { Gauge } from "lucide-react";
import { useIgpuCarveoutDialogStore } from "../stores/igpu-carveout-dialog-store";

/** Advisory shown at most once per GPU-memory allocation: this machine's
 *  integrated GPU has less memory dedicated to it than the model needs, so the
 *  weights run from shared system memory and generation is slower than it could
 *  be. The model has already loaded; nothing here blocks anything.
 *
 *  Deliberately not a warning triangle. The load succeeded and the user has done
 *  nothing wrong, so this is information about a setting, not an error. */
export function IgpuCarveoutDialog() {
  const open = useIgpuCarveoutDialogStore((state) => state.open);
  const advice = useIgpuCarveoutDialogStore((state) => state.advice);
  const close = useIgpuCarveoutDialogStore((state) => state.close);
  const dismissForever = useIgpuCarveoutDialogStore((state) => state.dismissForever);

  if (!advice) return null;

  return (
    <AlertDialog
      open={open}
      onOpenChange={(next) => {
        // Escape and the overlay are not a decision, so they close without
        // recording the dismissal and the notice may return on a later load.
        if (!next) close();
      }}
    >
      <AlertDialogContent className="max-w-lg" data-testid="igpu-carveout-dialog">
        <AlertDialogHeader>
          <div className="flex w-full min-w-0 items-start gap-3">
            <div className="flex size-9 shrink-0 items-center justify-center rounded-full bg-sky-500/10 text-sky-600 dark:text-sky-400">
              <Gauge className="size-5" />
            </div>
            {/* min-w-0 so a long unbreakable token in the backend's prose wraps
                instead of stretching the flex track past the dialog. */}
            <div className="min-w-0 flex-1 space-y-1 text-left">

              <AlertDialogTitle>This model could run faster</AlertDialogTitle>
              {/* break-words as well as min-w-0 above: constraining the flex
                  track stops the box growing, but only wrapping inside a word
                  stops a long unbreakable token overflowing it. */}
              <AlertDialogDescription className="break-words whitespace-pre-line">
                {advice.message}
              </AlertDialogDescription>
            </div>
          </div>
        </AlertDialogHeader>
        <AlertDialogFooter className="sm:justify-between">
          <AlertDialogCancel onClick={close}>Remind me later</AlertDialogCancel>
          <AlertDialogAction onClick={dismissForever}>
            Don&apos;t show this again
          </AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  );
}
