// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { toastError } from "@/shared/toast";
import { useState } from "react";
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

interface ShutdownDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  /** Called after the shutdown API returns success, right before we replace
   *  document.body with the "Server stopped" page. Callers use this to remove
   *  their beforeunload listener — otherwise the browser would prompt
   *  "Leave site?" when the user tries to close the final tab. */
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
      // Network error — shutdown request never reached the server
      toastError("Could not reach server");
      setStopping(false);
      return;
    }

    onAfterShutdown?.();
    document.body.innerHTML = `
      <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;height:100vh;font-family:sans-serif;gap:12px">
        <p style="font-size:1.1rem;font-weight:600;margin:0">Unsloth Studio has stopped.</p>
        <p style="font-size:0.9rem;color:#888;margin:0">You can now close this tab.</p>
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
