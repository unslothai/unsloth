// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect, useRef } from "react";
import { Switch } from "@/components/ui/switch";
import { useSidebarPin } from "@/hooks/use-sidebar-pin";
import { useT } from "@/i18n";
import {
  ActiveColorControl,
  ChatFontRow,
  ChatWidthSelect,
  CodeFontRow,
  CodeFontSizeRow,
  ContrastSliderRow,
  FontSmoothingSwitch,
  HeadingFontRow,
  InterfaceScaleRow,
  PointerCursorsSwitch,
  ReduceMotionSegmented,
  ResetCustomizationButton,
  UiFontRow,
  UiFontSizeRow,
} from "../components/appearance-custom-controls";
import { ColorThemeSelect } from "../components/color-theme-select";
import { SettingsRow } from "../components/settings-row";
import { SidebarMenuCustomizer } from "../components/sidebar-menu-customizer";
import { SidebarNavCustomizer } from "../components/sidebar-nav-customizer";
import { SettingsSection } from "../components/settings-section";
import { ThemeSegmented } from "../components/theme-segmented";
import { useSettingsDialogStore } from "../stores/settings-dialog-store";
import { useTheme } from "../stores/theme-store";

const FONT_CONTROL_CLASS = "flex w-76 max-w-full items-center gap-2";
const FONT_SELECT_CLASS = "w-auto min-w-0 flex-1";

export function AppearanceTab() {
  const t = useT();
  const { resolved } = useTheme();
  const { pinned, setPinned } = useSidebarPin();
  // The sidebar's "Customize sidebar" entry lands mid-page, so scroll its section into view.
  const sidebarNavSectionRef = useRef<HTMLDivElement | null>(null);
  const scrollTarget = useSettingsDialogStore((s) => s.scrollTarget);
  const consumeScrollTarget = useSettingsDialogStore(
    (s) => s.consumeScrollTarget,
  );
  useEffect(() => {
    if (scrollTarget !== "appearance-sidebar-nav") return;
    const frame = window.requestAnimationFrame(() => {
      sidebarNavSectionRef.current?.scrollIntoView({
        block: "start",
        behavior: "smooth",
      });
      consumeScrollTarget("appearance-sidebar-nav");
    });
    return () => window.cancelAnimationFrame(frame);
  }, [consumeScrollTarget, scrollTarget]);
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
        <SettingsRow label={t("settings.appearance.theme.label")}>
          <ThemeSegmented />
        </SettingsRow>
        <SettingsRow label={t("settings.appearance.palette.label")}>
          <ColorThemeSelect />
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
        <SettingsRow label={t("settings.appearance.custom.contrast.label")}>
          <ContrastSliderRow />
        </SettingsRow>
      </SettingsSection>

      <SettingsSection title={t("settings.appearance.custom.preferencesTitle")}>
        {/* Shared width so the font pickers line up. */}
        <SettingsRow label={t("settings.appearance.custom.uiFont.label")}>
          <div className={FONT_CONTROL_CLASS}>
            <UiFontRow className={FONT_SELECT_CLASS} />
            <UiFontSizeRow />
          </div>
        </SettingsRow>
        <SettingsRow label={t("settings.appearance.custom.headingFont.label")}>
          <div className={FONT_CONTROL_CLASS}>
            <HeadingFontRow className={FONT_SELECT_CLASS} />
          </div>
        </SettingsRow>
        <SettingsRow label={t("settings.appearance.custom.chatFont.label")}>
          <div className={FONT_CONTROL_CLASS}>
            <ChatFontRow className={FONT_SELECT_CLASS} />
          </div>
        </SettingsRow>
        <SettingsRow label={t("settings.appearance.custom.codeFont.label")}>
          <div className={FONT_CONTROL_CLASS}>
            <CodeFontRow className={FONT_SELECT_CLASS} />
            <CodeFontSizeRow />
          </div>
        </SettingsRow>
        <SettingsRow
          label={t("settings.appearance.custom.interfaceScale.label")}
          description={t(
            "settings.appearance.custom.interfaceScale.description",
          )}
        >
          <InterfaceScaleRow />
        </SettingsRow>
        <SettingsRow
          label={t("settings.appearance.custom.chatWidth.label")}
          description={t("settings.appearance.custom.chatWidth.description")}
        >
          <ChatWidthSelect />
        </SettingsRow>
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

      {/* Nav shape first, then the profile menu inside it. */}
      <div ref={sidebarNavSectionRef} className="scroll-mt-4">
        <SettingsSection
          title={t("settings.appearance.sidebarNav.title")}
          description={t("settings.appearance.sidebarNav.description")}
        >
          <div className="pt-3">
            <SidebarNavCustomizer />
          </div>
        </SettingsSection>
      </div>

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
