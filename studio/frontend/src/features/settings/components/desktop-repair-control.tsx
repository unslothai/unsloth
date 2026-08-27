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
 * Reruns the bundled installer over the managed environment.
 *
 * Exists because an update cannot repair everything an update can break. `studio update`
 * reuses the environment it finds; only the installer re-selects the PyTorch index and
 * force-reinstalls the trio. A managed venv that ended up with a CPU-only PyTorch wheel
 * therefore survives any number of successful updates, and until now the only way out was
 * to paste the install one-liner into a terminal.
 *
 * Desktop-only: outside Tauri there is no controller in context and the row renders nothing.
 * Confirmed before it runs, because it stops the backend and rewrites the environment.
 */
export function DesktopRepairControl(): ReactElement | null {
  const t = useT();
  const repair = useTauriRepairController();
  const [confirmOpen, setConfirmOpen] = useState(false);
  if (!repair) return null;

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
                // Closed first: the repair stops the backend and swaps the app over to the
                // repairing screen, so a dialog still mounted would sit on top of it.
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
