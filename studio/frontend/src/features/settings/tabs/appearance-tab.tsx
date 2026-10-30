// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Keep the background settings UI in this lazy tab, outside the chat startup barrel.
// eslint-disable-next-line no-restricted-imports
import { ChatBackgroundControls } from "@/features/chat-background/controls";
import { useEffect, useRef } from "react";
import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";
import { useSidebarPin } from "@/hooks/use-sidebar-pin";
import { TrickOrTreatSettingsRow } from "@/features/halloween/trick-or-treat-settings-row";
import { useT } from "@/i18n";
import { isTauri } from "@/lib/api-base";
import {
  ActiveColorControl,
  ChatFontRow,
  ChatWidthSelect,
  SentAttachmentsSelect,
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
import { GpuRenderingSegmented } from "../components/gpu-rendering-segmented";
import { SettingsRow } from "../components/settings-row";
import { SidebarMenuCustomizer } from "../components/sidebar-menu-customizer";
import { SidebarNavCustomizer } from "../components/sidebar-nav-customizer";
import { SettingsSection } from "../components/settings-section";
import { ThemeSegmented } from "../components/theme-segmented";
import { gpuRenderingNotes } from "../gpu-rendering-notes";
import { useGpuRenderingSetting } from "../hooks/use-gpu-rendering-setting";
import { useSettingsDialogStore } from "../stores/settings-dialog-store";

const FONT_CONTROL_CLASS = "flex w-76 max-w-full items-center gap-2";
const FONT_SELECT_CLASS = "w-auto min-w-0 flex-1";
const NOTE_CLASS =
  "max-w-[calc(280px*var(--ui-space-scale,1))] text-right text-xs";

function GpuRenderingRow() {
  const t = useT();
  const gpu = useGpuRenderingSetting({
    enabled: isTauri,
    loadError: t("settings.appearance.gpuRendering.loadError"),
    saveError: t("settings.appearance.gpuRendering.saveError"),
    restartError: t("settings.appearance.gpuRendering.restartError"),
  });
  const status = gpu.status;
  const notes = status ? gpuRenderingNotes(status, status.setting) : [];
  const busy = gpu.saving || gpu.restarting;
  const below =
    notes.length > 0 || gpu.error ? (
      <div className="flex flex-col items-end gap-1.5">
        {notes.map((note) => {
          switch (note.kind) {
            case "macNote":
              return (
                <span key={note.kind} className={`${NOTE_CLASS} text-muted-foreground`}>
                  {t("settings.appearance.gpuRendering.macNote")}
                </span>
              );
            case "envOverride":
              return (
                <span key={note.kind} className={`${NOTE_CLASS} text-muted-foreground`}>
                  {t("settings.appearance.gpuRendering.envOverride", {
                    name: note.name,
                  })}
                </span>
              );
            case "fallbackNote":
              return (
                <div key={note.kind} className="flex flex-col items-end gap-1.5">
                  <span className={`${NOTE_CLASS} text-muted-foreground`}>
                    {t("settings.appearance.gpuRendering.fallbackNote")}
                  </span>
                  <Button
                    type="button"
                    variant="outline"
                    size="sm"
                    disabled={busy || !status}
                    onClick={() => {
                      if (status) void gpu.update(status.setting);
                    }}
                  >
                    {t("settings.appearance.gpuRendering.tryAgain")}
                  </Button>
                </div>
              );
            case "oldWebkitWarning":
              return (
                <span key={note.kind} className={`${NOTE_CLASS} text-muted-foreground`}>
                  {t("settings.appearance.gpuRendering.oldWebkitWarning", {
                    version: note.version,
                  })}
                </span>
              );
            case "oldDriverWarning":
              return (
                <span key={note.kind} className={`${NOTE_CLASS} text-muted-foreground`}>
                  {t("settings.appearance.gpuRendering.oldDriverWarning", {
                    version: note.version,
                  })}
                </span>
              );
            case "restartPending":
              return (
                <div key={note.kind} className="flex items-center gap-2">
                  <span className={`${NOTE_CLASS} text-muted-foreground`}>
                    {t("settings.appearance.gpuRendering.restartPending")}
                  </span>
                  <Button
                    type="button"
                    variant="outline"
                    size="sm"
                    disabled={busy}
                    onClick={() => void gpu.relaunch()}
                  >
                    {t("settings.appearance.gpuRendering.restartNow")}
                  </Button>
                </div>
              );
          }
        })}
        {gpu.error ? (
          <span className={`${NOTE_CLASS} text-destructive`}>{gpu.error}</span>
        ) : null}
      </div>
    ) : null;
  return (
    <SettingsRow
      label={t("settings.appearance.gpuRendering.label")}
      description={t("settings.appearance.gpuRendering.description")}
      below={below}
    >
      <GpuRenderingSegmented
        value={status?.setting ?? null}
        onChange={(next) => void gpu.update(next)}
        disabled={!status || !status.configurable || busy}
      />
    </SettingsRow>
  );
}

export function AppearanceTab() {
  const t = useT();
  const { pinned, setPinned } = useSidebarPin();
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
    <div className="settings-page">
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
        <TrickOrTreatSettingsRow />
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

      <ChatBackgroundControls />
      <SettingsSection title={t("settings.appearance.custom.preferencesTitle")}>
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
          label={t("settings.appearance.custom.sentAttachments.label")}
          description={t(
            "settings.appearance.custom.sentAttachments.description",
          )}
        >
          <SentAttachmentsSelect />
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
      </SettingsSection>

      {isTauri ? (
        <SettingsSection title={t("settings.appearance.gpuRendering.sectionTitle")}>
          <GpuRenderingRow />
        </SettingsSection>
      ) : null}

      <div ref={sidebarNavSectionRef} className="scroll-mt-4">
        <SettingsSection
          title={t("settings.appearance.sidebarNav.title")}
          description={t("settings.appearance.sidebarNav.description")}
        >
          <SettingsRow
            label={t("settings.appearance.layout.compactSidebar")}
            description={t(
              "settings.appearance.layout.compactSidebarDescription",
            )}
          >
            <Switch checked={pinned} onCheckedChange={setPinned} />
          </SettingsRow>
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
