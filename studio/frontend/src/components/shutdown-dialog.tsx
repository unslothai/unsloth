// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { useI18n } from "@/features/i18n";
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
  const { t } = useI18n();

  const handleStop = async () => {
    setStopping(true);
    let accepted = false;
    try {
      const res = await authFetch("/api/shutdown", { method: "POST" });
      accepted = res.ok;
      if (!accepted) {
        toastError(t("shutdown.error.failed"));
        setStopping(false);
        return;
      }
    } catch {
      // Network error — shutdown request never reached the server
      toastError(t("shutdown.error.unreachable"));
      setStopping(false);
      return;
    }

    onAfterShutdown?.();
    document.body.innerHTML = `
      <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;height:100vh;font-family:sans-serif;gap:12px">
        <p style="font-size:1.1rem;font-weight:600;margin:0">${t("shutdown.stopped.title")}</p>
        <p style="font-size:0.9rem;color:#888;margin:0">${t("shutdown.stopped.hint")}</p>
      </div>`;
  };

  return (
    <AlertDialog open={open} onOpenChange={onOpenChange}>
      <AlertDialogContent>
        <AlertDialogHeader>
          <AlertDialogTitle>{t("shutdown.title")}</AlertDialogTitle>
          <AlertDialogDescription>
            {t("shutdown.description")}
          </AlertDialogDescription>
        </AlertDialogHeader>
        <AlertDialogFooter>
          <AlertDialogCancel>{t("shutdown.cancel")}</AlertDialogCancel>
          <AlertDialogAction
            onClick={handleStop}
            disabled={stopping}
            variant="destructive"
          >
            {stopping ? t("shutdown.stopping") : t("shutdown.stop")}
          </AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  );
}
