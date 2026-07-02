// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Switch } from "@/components/ui/switch";
import { useSidebarPin } from "@/hooks/use-sidebar-pin";
import { useT } from "@/i18n";
import { LanguageSelect } from "../components/language-select";
import { SettingsRow } from "../components/settings-row";
import { SettingsSection } from "../components/settings-section";
import { ThemeSegmented } from "../components/theme-segmented";

export function AppearanceTab() {
  const t = useT();
  const { pinned, setPinned } = useSidebarPin();
  return (
    <div className="flex flex-col gap-6">
      <header className="flex flex-col gap-1">
        <h1 className="text-xl font-semibold font-heading">
          {t("settings.appearance.title")}
        </h1>
        <p className="text-xs text-muted-foreground">
          {t("settings.appearance.description")}
        </p>
      </header>

      <SettingsSection title={t("settings.appearance.theme.title")}>
        <SettingsRow
          label={t("settings.appearance.theme.label")}
          description={t("settings.appearance.theme.description")}
        >
          <ThemeSegmented />
        </SettingsRow>
      </SettingsSection>

      <SettingsSection title={t("settings.appearance.language.title")}>
        <SettingsRow
          label={t("settings.appearance.language.label")}
          description={t("settings.appearance.language.description")}
        >
          <LanguageSelect />
        </SettingsRow>
      </SettingsSection>

      <SettingsSection title={t("settings.appearance.layout.title")}>
        <SettingsRow
          label={t("settings.appearance.layout.compactSidebar")}
          description={t("settings.appearance.layout.compactSidebarDescription")}
        >
          <Switch checked={pinned} onCheckedChange={setPinned} />
        </SettingsRow>
      </SettingsSection>
    </div>
  );
}
