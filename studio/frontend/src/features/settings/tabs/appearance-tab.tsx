// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Switch } from "@/components/ui/switch";
import { useSidebarPin } from "@/hooks/use-sidebar-pin";
import { useT } from "@/i18n";
import {
  ActiveColorControl,
  ChatFontRow,
  CodeFontRow,
  CodeFontSizeRow,
  ContrastSliderRow,
  FontSmoothingSwitch,
  HeadingFontRow,
  PointerCursorsSwitch,
  ReduceMotionSegmented,
  ResetCustomizationButton,
  UiFontRow,
  UiFontSizeRow,
} from "../components/appearance-custom-controls";
import { PaletteCards } from "../components/palette-cards";
import { SettingsRow } from "../components/settings-row";
import { SidebarMenuCustomizer } from "../components/sidebar-menu-customizer";
import {
  SettingsGroupDivider,
  SettingsSection,
} from "../components/settings-section";
import { ThemeSegmented } from "../components/theme-segmented";
import { useTheme } from "../stores/theme-store";

export function AppearanceTab() {
  const t = useT();
  const { resolved } = useTheme();
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
        <SettingsRow
          label={t("settings.appearance.palette.label")}
          description={t("settings.appearance.palette.description")}
          className="flex-col items-stretch gap-3"
        >
          <PaletteCards />
        </SettingsRow>
      </SettingsSection>

      <SettingsSection
        title={t(
          resolved === "light"
            ? "settings.appearance.custom.colors.lightGroup"
            : "settings.appearance.custom.colors.darkGroup",
        )}
      >
        <SettingsRow label={t("settings.appearance.custom.colors.accent")}>
          <ActiveColorControl
            colorKey="accent"
            label={t("settings.appearance.custom.colors.accent")}
          />
        </SettingsRow>
        <SettingsRow label={t("settings.appearance.custom.colors.background")}>
          <ActiveColorControl
            colorKey="background"
            label={t("settings.appearance.custom.colors.background")}
          />
        </SettingsRow>
        <SettingsRow label={t("settings.appearance.custom.colors.foreground")}>
          <ActiveColorControl
            colorKey="foreground"
            label={t("settings.appearance.custom.colors.foreground")}
          />
        </SettingsRow>
        <SettingsGroupDivider />
        <SettingsRow label={t("settings.appearance.custom.uiFont.label")}>
          <UiFontRow />
        </SettingsRow>
        <SettingsRow label={t("settings.appearance.custom.headingFont.label")}>
          <HeadingFontRow />
        </SettingsRow>
        <SettingsRow label={t("settings.appearance.custom.chatFont.label")}>
          <ChatFontRow />
        </SettingsRow>
        <SettingsRow label={t("settings.appearance.custom.codeFont.label")}>
          <CodeFontRow />
        </SettingsRow>
        <SettingsGroupDivider />
        <SettingsRow
          label={t("settings.appearance.custom.contrast.label")}
          description={t("settings.appearance.custom.contrast.description")}
        >
          <ContrastSliderRow />
        </SettingsRow>
      </SettingsSection>

      <SettingsSection title={t("settings.appearance.custom.preferencesTitle")}>
        <SettingsRow
          label={t("settings.appearance.custom.pointerCursors.label")}
          description={t(
            "settings.appearance.custom.pointerCursors.description",
          )}
        >
          <PointerCursorsSwitch />
        </SettingsRow>
        <SettingsRow
          label={t("settings.appearance.custom.reduceMotion.label")}
          description={t("settings.appearance.custom.reduceMotion.description")}
        >
          <ReduceMotionSegmented />
        </SettingsRow>
        <SettingsRow
          label={t("settings.appearance.custom.uiFontSize.label")}
          description={t("settings.appearance.custom.uiFontSize.description")}
        >
          <UiFontSizeRow />
        </SettingsRow>
        <SettingsRow
          label={t("settings.appearance.custom.codeFontSize.label")}
          description={t("settings.appearance.custom.codeFontSize.description")}
        >
          <CodeFontSizeRow />
        </SettingsRow>
        <SettingsRow
          label={t("settings.appearance.custom.fontSmoothing.label")}
          description={t(
            "settings.appearance.custom.fontSmoothing.description",
          )}
        >
          <FontSmoothingSwitch />
        </SettingsRow>
        <SettingsRow
          label={t("settings.appearance.layout.compactSidebar")}
          description={t(
            "settings.appearance.layout.compactSidebarDescription",
          )}
        >
          <Switch checked={pinned} onCheckedChange={setPinned} />
        </SettingsRow>
      </SettingsSection>

      <SettingsSection
        title={t("settings.appearance.sidebarMenu.title")}
        description={t("settings.appearance.sidebarMenu.description")}
      >
        <div className="pt-3">
          <SidebarMenuCustomizer />
        </div>
      </SettingsSection>

      <div className="flex justify-end border-t border-border/60 pt-4">
        <ResetCustomizationButton />
      </div>
    </div>
  );
}
