// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
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
  type LibraryLocation,
  type LibrarySearch,
  LibraryStorageBar,
  type LibrarySettings,
  type LibraryTab as LibraryTabId,
  type LibraryTabVisibility,
  type StorageCategory,
  SUGGESTED_LIMITS,
  formatSize,
  getLibraryLocations,
  moveLibraryLocation,
  parentFolder,
  refreshLibraryStorage,
  revealLibraryLocation,
  useLibrarySettingsStore,
  useLibraryStorage,
  useLibraryViewStore,
  useLibraryVisitStore,
  useRevealPlatform,
} from "@/features/library";
import { useIsAccountOwner } from "@/features/auth";
import { FolderBrowser } from "@/features/model-picker";
import { type TranslationKey, useT } from "@/i18n";
import { ChevronRightStandardIcon } from "@/lib/chevron-icons";
import { toast } from "@/lib/toast";
import { HugeiconsIcon } from "@hugeicons/react";
import { useNavigate } from "@tanstack/react-router";
import { useEffect, useRef, useState } from "react";
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

/** Library usage against the disk it lives on, and a way into each category, largest first. */
function StorageSection() {
  const t = useT();
  const navigate = useNavigate();
  const closeDialog = useSettingsDialogStore((s) => s.closeDialog);
  const scrollTarget = useSettingsDialogStore((s) => s.scrollTarget);
  const consumeScrollTarget = useSettingsDialogStore((s) => s.consumeScrollTarget);
  const sectionRef = useRef<HTMLElement | null>(null);
  const setView = useLibraryViewStore((s) => s.setView);
  const restartLibrary = useLibraryVisitStore((s) => s.restart);
  const storage = useLibraryStorage();

  // Manage storage in the Data tab lands here.
  useEffect(() => {
    if (scrollTarget !== "library-storage") return;
    const frame = window.requestAnimationFrame(() => {
      sectionRef.current?.scrollIntoView({ block: "start", behavior: "smooth" });
      consumeScrollTarget("library-storage");
    });
    return () => window.cancelAnimationFrame(frame);
  }, [consumeScrollTarget, scrollTarget]);

  const open = (link: LibrarySearch) => {
    closeDialog();
    setView("list");
    // Already on that tab, the view would keep its search and filters and show less than counted.
    restartLibrary();
    void navigate({ to: "/library", search: link });
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
          <LibraryStorageBar libraryBytes={storage.diskBytes} disk={storage.disk} />
        </div>
        {storage.categories.length === 0 && storage.hiddenBytes === 0 ? (
          <p className="pb-3 text-sm text-muted-foreground">{t("settings.library.storageEmpty")}</p>
        ) : (
          <>
            {storage.categories.length > 0 && (
              <div className="mb-3 flex flex-col overflow-hidden rounded-xl border border-border/60">
                {storage.categories.map((entry) => (
                  <button
                    key={entry.category}
                    type="button"
                    onClick={() => open(entry.link)}
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
            {storage.hiddenBytes > 0 && (
              <p className="pb-3 text-xs text-muted-foreground">
                {t("settings.library.storageHidden", { size: formatSize(storage.hiddenBytes) ?? "0 B" })}
              </p>
            )}
          </>
        )}
      </>
    );
  }

  return (
    <SettingsSection
      ref={sectionRef}
      title={t("settings.library.storageSection")}
      description={t("settings.library.storageDescription")}
    >
      {body}
    </SettingsSection>
  );
}

const LOCATION_LABELS: Record<LibraryLocation["key"], TranslationKey> = {
  uploads: "settings.library.locationUploads",
  images: "settings.library.categoryImages",
  videos: "settings.library.categoryVideos",
  audio: "settings.library.categoryAudio",
  fineTunes: "settings.library.categoryFineTunes",
  exports: "settings.library.locationExports",
};

/** Where each kind of file lives, with Reveal where Studio runs on this machine. */
function LocationsSection() {
  const t = useT();
  const reveal = useRevealPlatform();
  const [locations, setLocations] = useState<LibraryLocation[] | null>(null);
  useEffect(() => {
    let cancelled = false;
    getLibraryLocations().then(
      (next) => !cancelled && setLocations(next),
      () => !cancelled && setLocations([]),
    );
    return () => {
      cancelled = true;
    };
  }, []);
  const open = (key: LibraryLocation["key"]) =>
    revealLibraryLocation(key).catch((error: unknown) =>
      toast.error(t("settings.library.revealFailed"), {
        description: error instanceof Error ? error.message : String(error),
      }),
    );
  // Moving is the owner's call, like the model download folder: other accounts keep their files
  // in their own workspace.
  const owner = useIsAccountOwner();
  const [moving, setMoving] = useState(false);
  const [picking, setPicking] = useState<LibraryLocation | null>(null);
  // A Reset while the folder's drive is away leaves its files there: said before it happens.
  const [resettingAway, setResettingAway] = useState<LibraryLocation | null>(null);
  const nameOf = (location: LibraryLocation) => t(LOCATION_LABELS[location.key]);
  // Free space per row only when the folders span more than one disk; one disk is the bar above.
  const devices = new Set(
    (locations ?? []).flatMap((location) => (location.device ? [location.device] : [])),
  );

  async function move(location: LibraryLocation, path: string | null) {
    const name = nameOf(location);
    setMoving(true);
    // A move across drives can take minutes: the toast stays up and Change and Reset stay off
    // until the server answers.
    const id = toast.loading(t("settings.library.locationMoving", { name }));
    try {
      const result = await moveLibraryLocation(location.key, path);
      setLocations(result.locations);
      // The files may now sit on another disk, which the storage bar measures.
      refreshLibraryStorage();
      toast.success(t("settings.library.locationMoved", { name }), {
        id,
        description: result.leftBehind
          ? t("settings.library.locationLeftBehind", { path: result.leftBehind })
          : undefined,
      });
    } catch (error) {
      toast.error(t("settings.library.locationMoveFailed", { name }), {
        id,
        description: error instanceof Error ? error.message : String(error),
      });
    } finally {
      setMoving(false);
    }
  }

  return (
    <SettingsSection
      title={t("settings.library.locationsSection")}
      description={t("settings.library.locationsDescription")}
    >
      {locations === null ? (
        <Spinner className="my-4 size-5 text-muted-foreground" />
      ) : (
        locations.map((location) => (
          <SettingsRow
            key={location.key}
            label={t(LOCATION_LABELS[location.key])}
            description={
              <>
                <span className="block truncate font-mono text-[11px]" title={location.path}>
                  {location.path}
                </span>
                {location.available === false ? (
                  <span className="block text-xs text-destructive">
                    {t("settings.library.locationUnavailable")}
                  </span>
                ) : (
                  devices.size > 1 &&
                  location.disk && (
                    <span className="block text-xs">
                      {t("settings.library.locationFree", {
                        free: formatSize(location.disk.freeBytes) ?? "",
                      })}
                    </span>
                  )
                )}
              </>
            }
          >
            <div className="flex shrink-0 items-center gap-2">
              {owner && location.movable && location.custom && (
                <Button
                  variant="outline"
                  size="sm"
                  disabled={moving}
                  onClick={() =>
                    location.available === false
                      ? setResettingAway(location)
                      : void move(location, null)
                  }
                >
                  {t("settings.library.locationReset")}
                </Button>
              )}
              {owner && location.movable && (
                <Button
                  variant="outline"
                  size="sm"
                  disabled={moving}
                  onClick={() => setPicking(location)}
                >
                  {t("settings.library.locationChange")}
                </Button>
              )}
              {reveal && (
                <Button variant="outline" size="sm" onClick={() => void open(location.key)}>
                  {t(reveal === "finder" ? "settings.library.revealInFinder" : "settings.library.revealInFolder")}
                </Button>
              )}
            </div>
          </SettingsRow>
        ))
      )}
      <FolderBrowser
        open={picking !== null}
        onOpenChange={(next) => !next && setPicking(null)}
        onSelect={(path) => {
          if (picking) void move(picking, path);
        }}
        // Start beside the current folder, where a new one usually goes.
        initialPath={picking ? parentFolder(picking.path) : undefined}
        title={picking ? t("settings.library.locationMoveTitle", { name: nameOf(picking) }) : ""}
        description={t("settings.library.locationMoveDescription")}
        confirmLabel={t("settings.library.locationMoveAction")}
        showModelHints={false}
      />
      <AlertDialog
        open={resettingAway !== null}
        onOpenChange={(open) => !open && setResettingAway(null)}
      >
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle className="break-words">
              {t("settings.library.locationResetUnavailableTitle", {
                name: resettingAway ? nameOf(resettingAway) : "",
              })}
            </AlertDialogTitle>
            <AlertDialogDescription className="break-words">
              {t("settings.library.locationResetUnavailableDescription", {
                path: resettingAway?.path ?? "",
              })}
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>{t("common.cancel")}</AlertDialogCancel>
            <AlertDialogAction
              onClick={() => {
                if (resettingAway) void move(resettingAway, null);
                setResettingAway(null);
              }}
            >
              {t("settings.library.locationReset")}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
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

      <LocationsSection />
    </div>
  );
}
