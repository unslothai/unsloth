// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import {
  type LibrarySettings,
  SUGGESTED_LIMITS,
  useLibrarySettingsStore,
} from "@/features/library";
import { type TranslationKey, useT } from "@/i18n";
import { toast } from "@/lib/toast";
import { SettingsRow } from "../components/settings-row";
import { SettingsSection } from "../components/settings-section";

type ChoiceKey = {
  [K in keyof LibrarySettings]: LibrarySettings[K] extends string ? K : never;
}[keyof LibrarySettings];

type ToggleKey = {
  [K in keyof LibrarySettings]: LibrarySettings[K] extends boolean ? K : never;
}[keyof LibrarySettings];

const CHOICES: { [K in ChoiceKey]: [LibrarySettings[K], TranslationKey][] } = {
  cardSize: [
    ["small", "settings.library.small"],
    ["medium", "settings.library.medium"],
    ["large", "settings.library.large"],
  ],
  imageLayout: [
    ["masonry", "settings.library.masonry"],
    ["square", "settings.library.square"],
  ],
  startTab: [
    ["suggested", "settings.library.suggested"],
    ["favorites", "settings.library.favorites"],
    ["folders", "settings.library.folders"],
    ["all", "settings.library.all"],
  ],
  sort: [
    ["recent", "settings.library.recent"],
    ["oldest", "settings.library.oldest"],
    ["name", "settings.library.name"],
    ["size", "settings.library.size"],
  ],
  mediaTabs: [
    ["auto", "settings.library.mediaTabsAuto"],
    ["always", "settings.library.mediaTabsAlways"],
  ],
};

function ChoiceSelect({
  label,
  value,
  options,
  onChange,
}: {
  label: string;
  value: string;
  options: [string, string][];
  onChange: (value: string) => void;
}) {
  return (
    <Select value={value} onValueChange={onChange}>
      <SelectTrigger size="sm" className="w-48" aria-label={label}>
        <SelectValue />
      </SelectTrigger>
      <SelectContent align="end">
        {options.map(([option, text]) => (
          <SelectItem key={option} value={option}>
            {text}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  );
}

export function LibraryTab() {
  const t = useT();
  const settings = useLibrarySettingsStore();

  const choice = <K extends ChoiceKey>(key: K, labelKey: TranslationKey, descriptionKey: TranslationKey) => (
    <SettingsRow label={t(labelKey)} description={t(descriptionKey)}>
      <ChoiceSelect
        label={t(labelKey)}
        value={settings[key]}
        options={CHOICES[key].map(([option, text]) => [option, t(text)])}
        onChange={(value) => settings.set({ [key]: value } as Partial<LibrarySettings>)}
      />
    </SettingsRow>
  );

  const toggle = (key: ToggleKey, labelKey: TranslationKey, descriptionKey: TranslationKey) => (
    <SettingsRow label={t(labelKey)} description={t(descriptionKey)}>
      <Switch
        aria-label={t(labelKey)}
        checked={settings[key]}
        onCheckedChange={(checked) => settings.set({ [key]: checked })}
      />
    </SettingsRow>
  );

  return (
    <div className="flex flex-col gap-6">
      <header className="flex flex-col gap-1">
        <h1 className="text-xl font-semibold font-heading">{t("settings.library.title")}</h1>
      </header>

      <SettingsSection title={t("settings.library.layoutSection")}>
        {choice("cardSize", "settings.library.cardSize", "settings.library.cardSizeDescription")}
        {choice("imageLayout", "settings.library.imageLayout", "settings.library.imageLayoutDescription")}
        {toggle("showCardDates", "settings.library.showCardDates", "settings.library.showCardDatesDescription")}
      </SettingsSection>

      <SettingsSection title={t("settings.library.browsingSection")}>
        {choice("startTab", "settings.library.startTab", "settings.library.startTabDescription")}
        {choice("sort", "settings.library.sort", "settings.library.sortDescription")}
        <SettingsRow
          label={t("settings.library.suggestedLimit")}
          description={t("settings.library.suggestedLimitDescription")}
        >
          <ChoiceSelect
            label={t("settings.library.suggestedLimit")}
            value={String(settings.suggestedLimit)}
            options={SUGGESTED_LIMITS.map((limit) => [String(limit), String(limit)])}
            onChange={(value) => settings.set({ suggestedLimit: Number(value) })}
          />
        </SettingsRow>
        {choice("mediaTabs", "settings.library.mediaTabs", "settings.library.mediaTabsDescription")}
      </SettingsSection>

      <SettingsSection
        title={t("settings.library.contentSection")}
        description={t("settings.library.contentDescription")}
      >
        {toggle("showChatAttachments", "settings.library.showChatAttachments", "settings.library.showChatAttachmentsDescription")}
        {toggle("showChatToolFiles", "settings.library.showChatToolFiles", "settings.library.showChatToolFilesDescription")}
        {toggle("showGeneratedMedia", "settings.library.showGeneratedMedia", "settings.library.showGeneratedMediaDescription")}
        {toggle("showFineTunes", "settings.library.showFineTunes", "settings.library.showFineTunesDescription")}
      </SettingsSection>

      <SettingsSection title={t("settings.library.deletingSection")}>
        {toggle("confirmDelete", "settings.library.confirmDelete", "settings.library.confirmDeleteDescription")}
        <SettingsRow
          label={t("settings.library.reset")}
          description={t("settings.library.resetDescription")}
          destructive={true}
        >
          <Button
            variant="outline"
            size="sm"
            onClick={() => {
              settings.reset();
              toast.success(t("settings.library.resetDone"));
            }}
          >
            {t("settings.library.resetButton")}
          </Button>
        </SettingsRow>
      </SettingsSection>
    </div>
  );
}
