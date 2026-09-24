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
import { Spinner } from "@/components/ui/spinner";
import {
  LIBRARY_TABS,
  type LibrarySettings,
  type LibraryTab as LibraryTabId,
  type LibraryTabVisibility,
  type StorageCategory,
  SUGGESTED_LIMITS,
  formatSize,
  useLibrarySettingsStore,
  useLibraryStorage,
  useLibraryViewStore,
} from "@/features/library";
import { useSystemInfo } from "@/hooks";
import { type TranslationKey, useT } from "@/i18n";
import { ChevronRightStandardIcon } from "@/lib/chevron-icons";
import { toast } from "@/lib/toast";
import { HugeiconsIcon } from "@hugeicons/react";
import { useNavigate } from "@tanstack/react-router";
import { SettingsRow } from "../components/settings-row";
import { SettingsSection } from "../components/settings-section";
import { useSettingsDialogStore } from "../stores/settings-dialog-store";

type ChoiceKey = {
  [K in keyof LibrarySettings]: LibrarySettings[K] extends string ? K : never;
}[keyof LibrarySettings];

type ToggleKey = {
  [K in keyof LibrarySettings]: LibrarySettings[K] extends boolean ? K : never;
}[keyof LibrarySettings];

const CHOICES: { [K in Exclude<ChoiceKey, "lastTab">]: [LibrarySettings[K], TranslationKey][] } = {
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
    ["last", "settings.library.lastVisited"],
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
};

const TAB_LABELS: Record<LibraryTabId, TranslationKey> = {
  suggested: "settings.library.suggested",
  favorites: "settings.library.favorites",
  folders: "settings.library.folders",
  images: "settings.library.categoryImages",
  videos: "settings.library.categoryVideos",
  audio: "settings.library.categoryAudio",
  models: "settings.library.categoryFineTunes",
  all: "settings.library.all",
};

// These tabs can wait until they have something in them; the rest are simply on or off.
const CONTENT_TABS = new Set<LibraryTabId>(["images", "videos", "audio", "models"]);

const TAB_VISIBILITY: [LibraryTabVisibility, TranslationKey][] = [
  ["auto", "settings.library.tabAuto"],
  ["always", "settings.library.tabAlways"],
  ["hidden", "settings.library.tabHidden"],
];

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

const CATEGORY_LABELS: Record<StorageCategory, TranslationKey> = {
  files: "settings.library.categoryFiles",
  images: "settings.library.categoryImages",
  videos: "settings.library.categoryVideos",
  audio: "settings.library.categoryAudio",
  fineTunes: "settings.library.categoryFineTunes",
};

const GB = 1e9;

/** Library usage against the disk it lives on, and a way into each category, largest first. */
function StorageSection() {
  const t = useT();
  const navigate = useNavigate();
  const closeDialog = useSettingsDialogStore((s) => s.closeDialog);
  const setView = useLibraryViewStore((s) => s.setView);
  const storage = useLibraryStorage();
  const { disk } = useSystemInfo();
  const diskBytes = disk.total_gb > 0 ? disk.total_gb * GB : 0;
  const otherBytes = Math.max(0, diskBytes - disk.free_gb * GB - storage.totalBytes);
  const share = (bytes: number) => (diskBytes ? `${(bytes / diskBytes) * 100}%` : "0%");

  const open = (tab: (typeof storage.categories)[number]["tab"]) => {
    closeDialog();
    setView("list");
    void navigate({ to: "/library", search: { show: tab, sort: "size" } });
  };

  let body;
  if (storage.status === "loading") {
    body = <Spinner className="my-4 size-5 text-muted-foreground" />;
  } else if (storage.status === "error") {
    body = <p className="py-3 text-sm text-muted-foreground">{t("settings.library.storageError")}</p>;
  } else {
    body = (
      <>
        <div className="flex flex-col gap-2 py-3">
          <p className="text-sm font-medium text-foreground">
            {t("settings.library.storageUsed", { size: formatSize(storage.totalBytes) ?? "0 B" })}
          </p>
          {diskBytes > 0 && (
            <>
              {/* The Library, then everything else on the disk, then free space. */}
              <div className="flex h-2 overflow-hidden rounded-full bg-muted">
                <div
                  className="min-w-2 rounded-full bg-foreground"
                  style={{ width: share(storage.totalBytes) }}
                />
                <div className="bg-muted-foreground/35" style={{ width: share(otherBytes) }} />
              </div>
              <p className="text-xs text-muted-foreground">
                {t("settings.library.storageDisk", {
                  free: formatSize(disk.free_gb * GB) ?? "",
                  total: formatSize(diskBytes) ?? "",
                })}
              </p>
            </>
          )}
        </div>
        {storage.categories.length === 0 ? (
          <p className="pb-3 text-sm text-muted-foreground">{t("settings.library.storageEmpty")}</p>
        ) : (
          <div className="mb-3 flex flex-col overflow-hidden rounded-xl border border-border/60">
            {storage.categories.map((entry) => (
              <button
                key={entry.category}
                type="button"
                onClick={() => open(entry.tab)}
                className="flex items-center gap-3 border-border/60 px-4 py-3 text-left transition-colors not-first:border-t hover:bg-muted/60"
              >
                <span className="flex min-w-0 flex-1 flex-col gap-0.5">
                  <span className="text-sm font-medium text-foreground">
                    {t(CATEGORY_LABELS[entry.category])}
                  </span>
                  <span className="text-xs text-muted-foreground">
                    {formatSize(entry.bytes)}
                    {" · "}
                    {entry.count === 1
                      ? t("settings.library.itemCountOne")
                      : t("settings.library.itemCount", { count: entry.count.toLocaleString() })}
                  </span>
                </span>
                <HugeiconsIcon icon={ChevronRightStandardIcon} className="size-4 shrink-0 text-muted-foreground" />
              </button>
            ))}
          </div>
        )}
      </>
    );
  }

  return (
    <SettingsSection
      title={t("settings.library.storageSection")}
      description={t("settings.library.storageDescription")}
    >
      {body}
    </SettingsSection>
  );
}

export function LibraryTab() {
  const t = useT();
  const settings = useLibrarySettingsStore();

  const choice = <K extends keyof typeof CHOICES>(key: K, labelKey: TranslationKey, descriptionKey: TranslationKey) => (
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

      <StorageSection />

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
      </SettingsSection>

      <SettingsSection
        title={t("settings.library.tabsSection")}
        description={t("settings.library.tabsDescription")}
      >
        {LIBRARY_TABS.map((tab) => {
          const label = t(TAB_LABELS[tab]);
          const visibility = settings.tabs[tab] ?? "always";
          const setVisibility = (next: LibraryTabVisibility) =>
            settings.set({ tabs: { ...settings.tabs, [tab]: next } });
          return (
            <SettingsRow key={tab} label={label}>
              {CONTENT_TABS.has(tab) ? (
                <ChoiceSelect
                  label={label}
                  value={visibility}
                  options={TAB_VISIBILITY.map(([option, text]) => [option, t(text)])}
                  onChange={(value) => setVisibility(value as LibraryTabVisibility)}
                />
              ) : (
                <Switch
                  aria-label={label}
                  checked={visibility !== "hidden"}
                  onCheckedChange={(checked) => setVisibility(checked ? "always" : "hidden")}
                />
              )}
            </SettingsRow>
          );
        })}
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
