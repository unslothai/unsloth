// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

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
  /** Called right before the shutdown API request so callers can remove the
   *  beforeunload listener — otherwise the "Server stopped" page would still
   *  trigger a "Leave site?" prompt when the user tries to close it. */
  onBeforeShutdown?: () => void;
}

export function ShutdownDialog({
  open,
  onOpenChange,
  onBeforeShutdown,
}: ShutdownDialogProps) {
  const [stopping, setStopping] = useState(false);

  const handleStop = async () => {
    setStopping(true);
    onBeforeShutdown?.();
    try {
      await fetch("/api/shutdown", { method: "POST" });
    } catch {
      // Server may already be unreachable — that's fine
    }
    // Replace page content — the SPA is no longer functional after shutdown.
    // We avoid a router navigation since all API calls will fail.
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
            className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
          >
            {stopping ? "Stopping…" : "Stop server"}
          </AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  );
}
