// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { useTauriRepairController } from "@/hooks/tauri-repair-context";
import { useT } from "@/i18n";
import { type ReactElement, useState } from "react";
import { SettingsRow } from "./settings-row";

/**
 * Reruns the bundled installer: `studio update` reuses the environment, so only this re-selects
 * the PyTorch index (e.g. to fix a CPU-only wheel). Desktop-only, and hidden for an external
 * backend, where the refusal would arrive after the shell switched to the repair screen.
 */
export function DesktopRepairControl(): ReactElement | null {
  const t = useT();
  const repair = useTauriRepairController();
  const [confirmOpen, setConfirmOpen] = useState(false);
  if (!repair || repair.isExternalServer) return null;

  return (
    <>
      <SettingsRow
        destructive={true}
        label={t("settings.general.repairInstall.label")}
        description={t("settings.general.repairInstall.description")}
      >
        <Button
          variant="outline"
          size="sm"
          onClick={() => setConfirmOpen(true)}
          className="text-destructive hover:text-destructive hover:border-destructive/60"
        >
          {t("settings.general.repairInstall.action")}
        </Button>
      </SettingsRow>

      <Dialog open={confirmOpen} onOpenChange={setConfirmOpen}>
        <DialogContent className="max-w-md">
          <DialogHeader>
            <DialogTitle>
              {t("settings.general.repairInstall.confirmTitle")}
            </DialogTitle>
            <DialogDescription>
              {t("settings.general.repairInstall.confirmDescription")}
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="outline" onClick={() => setConfirmOpen(false)}>
              {t("common.cancel")}
            </Button>
            <Button
              onClick={() => {
                // Close first, or the dialog would sit on top of the repairing screen.
                setConfirmOpen(false);
                void repair.repairInstall();
              }}
              className="bg-destructive hover:bg-destructive/90 text-destructive-foreground"
            >
              {t("settings.general.repairInstall.confirmAction")}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
}
