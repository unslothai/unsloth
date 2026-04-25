// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Switch } from "@/components/ui/switch";
import { useI18n } from "@/features/i18n";
import { useSidebarPin } from "@/hooks/use-sidebar-pin";
import { SettingsRow } from "../components/settings-row";
import { SettingsSection } from "../components/settings-section";
import { ThemeSegmented } from "../components/theme-segmented";

export function AppearanceTab() {
  const { t } = useI18n();
  const { pinned, setPinned } = useSidebarPin();
  return (
    <div className="flex flex-col gap-6">
      <header className="flex flex-col gap-1">
        <h1 className="text-lg font-semibold font-heading">
          {t("settings.appearance.title")}
        </h1>
        <p className="text-xs text-muted-foreground">
          {t("settings.appearance.subtitle")}
        </p>
      </header>

      <SettingsSection title={t("settings.appearance.theme")}>
        <SettingsRow
          label={t("settings.appearance.colorScheme.label")}
          description={t("settings.appearance.colorScheme.description")}
        >
          <ThemeSegmented />
        </SettingsRow>
      </SettingsSection>

      <SettingsSection title={t("settings.appearance.layout")}>
        <SettingsRow
          label={t("settings.appearance.pinSidebar.label")}
          description={t("settings.appearance.pinSidebar.description")}
        >
          <Switch checked={pinned} onCheckedChange={setPinned} />
        </SettingsRow>
      </SettingsSection>
    </div>
  );
}
