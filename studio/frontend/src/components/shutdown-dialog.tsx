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
import { authFetch } from "@/features/auth";
import { toastError } from "@/shared/toast";
import { useState } from "react";

interface ShutdownDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  /** Called after shutdown succeeds, before we replace document.body. Lets
   *  callers remove their beforeunload listener so the browser doesn't prompt
   *  "Leave site?" when closing the final tab. */
  onAfterShutdown?: () => void;
}

export function ShutdownDialog({
  open,
  onOpenChange,
  onAfterShutdown,
}: ShutdownDialogProps) {
  const [stopping, setStopping] = useState(false);

  const handleStop = async () => {
    setStopping(true);
    let accepted = false;
    try {
      const res = await authFetch("/api/shutdown", { method: "POST" });
      accepted = res.ok;
      if (!accepted) {
        toastError("Failed to shut down server");
        setStopping(false);
        return;
      }
    } catch {
      // Network error: request never reached the server
      toastError("Could not reach server");
      setStopping(false);
      return;
    }

    onAfterShutdown?.();
    document.body.innerHTML = `
      <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;height:100vh;font-family:sans-serif;gap:12px">
        <p style="font-size:calc(1.1rem * var(--ui-font-scale, 1));font-weight:600;margin:0">Unsloth Studio has stopped.</p>
        <p style="font-size:calc(0.9rem * var(--ui-font-scale, 1));color:#888;margin:0">You can now close this tab.</p>
      </div>`;
  };

  return (
    <AlertDialog open={open} onOpenChange={onOpenChange}>
      <AlertDialogContent>
        <AlertDialogHeader>
          <AlertDialogTitle>Stop Unsloth Studio?</AlertDialogTitle>
          <AlertDialogDescription>
            This will shut down the server. Any active training or inference
            jobs will be terminated. You can restart it any time from the
            desktop shortcut.
          </AlertDialogDescription>
        </AlertDialogHeader>
        <AlertDialogFooter>
          <AlertDialogCancel>Cancel</AlertDialogCancel>
          <AlertDialogAction
            onClick={handleStop}
            disabled={stopping}
            variant="destructive"
          >
            {stopping ? "Stopping…" : "Stop server"}
          </AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  );
}
