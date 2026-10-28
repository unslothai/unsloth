// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";
import { SettingsRow } from "@/features/settings/components/settings-row";
import { useT } from "@/i18n";
import { useState } from "react";
import {
  hideTrickOrTreatRow,
  isHalloweenSeason,
  showTrickOrTreatRow,
  useTrickOrTreatVisible,
} from "./halloween-store";
import { TrickOrTreatDialog } from "./trick-or-treat-dialog";

export function TrickOrTreatSettingsRows() {
  const t = useT();
  const inSidebar = useTrickOrTreatVisible();
  const [open, setOpen] = useState(false);
  if (!isHalloweenSeason()) return null;
  return (
    <>
      <SettingsRow
        label={t("settings.appearance.trickOrTreat.label")}
        description={t("settings.appearance.trickOrTreat.description")}
      >
        <Button
          size="sm"
          variant="outline"
          data-testid="settings-trick-or-treat-open"
          onClick={() => setOpen(true)}
        >
          {t("settings.appearance.trickOrTreat.open")}
        </Button>
      </SettingsRow>
      <SettingsRow label={t("settings.appearance.trickOrTreat.showInSidebar")}>
        <Switch
          aria-label={t("settings.appearance.trickOrTreat.showInSidebar")}
          data-testid="settings-trick-or-treat-sidebar"
          checked={inSidebar}
          onCheckedChange={(on) =>
            on ? showTrickOrTreatRow() : hideTrickOrTreatRow()
          }
        />
      </SettingsRow>
      <TrickOrTreatDialog open={open} onOpenChange={setOpen} />
    </>
  );
}
