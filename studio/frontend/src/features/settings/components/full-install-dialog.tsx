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
import { useFullInstallUpgrade } from "@/hooks/use-full-install-upgrade";
import { useT } from "@/i18n";
import { type ReactElement, useState } from "react";
import { SettingsRow } from "./settings-row";

/**
 * Confirms turning a Chat only (GGUF) install into the full one. The work is the manual repair:
 * it reruns the installer with PyTorch, which training, export, video, safetensors models and
 * the image generation features beyond GGUF need.
 */
function FullInstallDialog({
  open,
  onOpenChange,
  onConfirm,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onConfirm: () => void;
}): ReactElement {
  const t = useT();
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-md">
        <DialogHeader>
          <DialogTitle>{t("settings.general.fullInstall.confirmTitle")}</DialogTitle>
          <DialogDescription>
            {t("settings.general.fullInstall.confirmDescription")}
          </DialogDescription>
        </DialogHeader>
        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)}>
            {t("common.cancel")}
          </Button>
          <Button
            onClick={() => {
              // Closed first: the repair swaps the app over to the progress screen, so a dialog
              // still mounted would sit on top of it.
              onOpenChange(false);
              onConfirm();
            }}
          >
            {t("settings.general.fullInstall.confirmAction")}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

/** Settings > General "Install type" row, beside the version and update rows it belongs with. */
export function DesktopInstallTypeControl(): ReactElement | null {
  const t = useT();
  const controller = useFullInstallUpgrade();
  const [confirmOpen, setConfirmOpen] = useState(false);
  if (!controller) return null;

  return (
    <>
      <SettingsRow
        label={t("settings.general.fullInstall.label")}
        description={t("settings.general.fullInstall.description")}
      >
        <Button size="sm" onClick={() => setConfirmOpen(true)}>
          {t("settings.general.fullInstall.action")}
        </Button>
      </SettingsRow>
      <FullInstallDialog
        open={confirmOpen}
        onOpenChange={setConfirmOpen}
        onConfirm={() => void controller.repairInstall()}
      />
    </>
  );
}
