// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Switch } from "@/components/ui/switch";
import { SettingsRow } from "@/features/settings/components/settings-row";
import { useT } from "@/i18n";
import {
  hideTrickOrTreatRow,
  isHalloweenSeason,
  showTrickOrTreatRow,
  useTrickOrTreatVisible,
} from "./halloween-store";

export function TrickOrTreatSettingsRow() {
  const t = useT();
  const inSidebar = useTrickOrTreatVisible();
  if (!isHalloweenSeason()) return null;
  return (
    <SettingsRow
      label={t("settings.appearance.trickOrTreat.label")}
      description={t("settings.appearance.trickOrTreat.description")}
    >
      <Switch
        aria-label={t("settings.appearance.trickOrTreat.label")}
        data-testid="settings-trick-or-treat-sidebar"
        checked={inSidebar}
        onCheckedChange={(on) =>
          on ? showTrickOrTreatRow() : hideTrickOrTreatRow()
        }
      />
    </SettingsRow>
  );
}
