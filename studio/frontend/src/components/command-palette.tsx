// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useAnimatedThemeToggle } from "@/components/ui/animated-theme-toggler";
import {
  Command,
  CommandDialog,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
  CommandSeparator,
  CommandShortcut,
} from "@/components/ui/command";
import { useIsAccountOwner } from "@/features/auth";
import { useChatSearchStore } from "@/features/chat";
import {
  DIALOG_SETTINGS_SEARCH_INDEX,
  SETTINGS_TABS,
  type SettingsTab,
  type ShortcutId,
  settingsTabVisible,
  triggerShortcut,
  useSettingsDialogStore,
  useShortcut,
  useShortcutAvailable,
  useShortcutLabel,
} from "@/features/settings";
import { useT, type TranslationKey } from "@/i18n";
import { useCommandPaletteStore } from "@/stores/command-palette";
import {
  AudioWave01Icon,
  ChefHatIcon,
  DashboardCircleIcon,
  DownloadSquare01Icon,
  FlimSlateIcon,
  Folder01Icon,
  Globe02Icon,
  Image03Icon,
  LibrariesIcon,
  Message01Icon,
  PencilEdit02Icon,
  Search01Icon,
  Settings02Icon,
  Sun03Icon,
  TestTube01Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useNavigate } from "@tanstack/react-router";
import { Moon } from "lucide-react";
import { useEffect, useState } from "react";

// matches sidebar: drop interior bubble paths
const TestTubeOutlineIcon = TestTube01Icon.slice(0, 3) as typeof TestTube01Icon;

// Through the root's workspace shortcuts: gated (Train, Video) and landing (Chat keeps its thread) as the chords do.
const WORKSPACES: {
  id: ShortcutId;
  icon: typeof Message01Icon;
  labelKey: TranslationKey;
  aliases?: string[];
}[] = [
  {
    id: "switchToChat",
    icon: Message01Icon,
    labelKey: "shell.commandPalette.chat",
  },
  {
    id: "switchToProjects",
    icon: Folder01Icon,
    labelKey: "shell.navigation.projects",
  },
  {
    id: "switchToHub",
    icon: DashboardCircleIcon,
    labelKey: "shell.navigation.hub",
    aliases: ["models"],
  },
  {
    id: "switchToTrain",
    icon: TestTubeOutlineIcon,
    labelKey: "shell.navigation.train",
    aliases: ["fine-tune", "training"],
  },
  {
    id: "switchToRecipes",
    icon: ChefHatIcon,
    labelKey: "shell.navigation.recipes",
    aliases: ["data", "datasets"],
  },
  {
    id: "switchToImages",
    icon: Image03Icon,
    labelKey: "shell.navigation.images",
    aliases: ["generate"],
  },
  {
    id: "switchToVideo",
    icon: FlimSlateIcon,
    labelKey: "shell.navigation.video",
    aliases: ["generate"],
  },
  {
    id: "switchToAudio",
    icon: AudioWave01Icon,
    labelKey: "shell.navigation.audio",
  },
  {
    id: "switchToExport",
    icon: DownloadSquare01Icon,
    labelKey: "shell.navigation.export",
    aliases: ["gguf", "checkpoint"],
  },
];

const SETTINGS_TAB_LABELS: Record<SettingsTab, TranslationKey> = {
  general: "settings.tabs.general",
  profile: "settings.tabs.profile",
  accounts: "settings.tabs.accounts",
  appearance: "settings.tabs.appearance",
  resources: "settings.tabs.resources",
  chat: "settings.tabs.chat",
  voice: "settings.tabs.voice",
  connections: "settings.tabs.connections",
  library: "shell.navigation.library",
  data: "settings.tabs.data",
  "api-keys": "settings.tabs.apiKeys",
  "remote-lan": "settings.tabs.remoteLan",
  agents: "settings.tabs.agents",
  "keyboard-shortcuts": "settings.tabs.keyboardShortcuts",
  debugging: "settings.tabs.debugging",
  about: "settings.tabs.about",
};

export function CommandPalette() {
  const isOpen = useCommandPaletteStore((s) => s.isOpen);
  const setOpen = useCommandPaletteStore((s) => s.setOpen);

  useShortcut("openCommandPalette", () =>
    useCommandPaletteStore.getState().toggle(),
  );

  // Auth routes unmount the palette; close it so it does not come back open.
  useEffect(() => () => useCommandPaletteStore.getState().setOpen(false), []);

  // A chord (⌘, / ⌘K) can open Settings or chat search over the palette; it gives way to them.
  const settingsOpen = useSettingsDialogStore((s) => s.open);
  const chatSearchOpen = useChatSearchStore((s) => s.isOpen);
  useEffect(() => {
    if (settingsOpen || chatSearchOpen) {
      useCommandPaletteStore.getState().close();
    }
  }, [settingsOpen, chatSearchOpen]);

  return (
    <CommandDialog
      open={isOpen}
      onOpenChange={setOpen}
      className="w-140 max-w-[calc(100%-2rem)] sm:max-w-140"
    >
      <PaletteContent />
    </CommandDialog>
  );
}

function WorkspaceItem({
  workspace,
  onSelect,
}: {
  workspace: (typeof WORKSPACES)[number];
  onSelect: () => void;
}) {
  const t = useT();
  const available = useShortcutAvailable(workspace.id, false);
  const label = useShortcutLabel(workspace.id);
  if (!available) return null;
  return (
    <CommandItem onSelect={onSelect} keywords={workspace.aliases}>
      <HugeiconsIcon icon={workspace.icon} strokeWidth={1.75} />
      <span>{t(workspace.labelKey)}</span>
      {label && <CommandShortcut>{label}</CommandShortcut>}
    </CommandItem>
  );
}

function PaletteContent() {
  const t = useT();
  const navigate = useNavigate();
  const close = useCommandPaletteStore((s) => s.close);
  const isOwner = useIsAccountOwner();
  const { isDark, toggleTheme, anchorRef } = useAnimatedThemeToggle();
  const [query, setQuery] = useState("");
  // Content stays mounted through the exit animation, so a quick reopen would keep the old filter.
  const isOpen = useCommandPaletteStore((s) => s.isOpen);
  const [wasOpen, setWasOpen] = useState(isOpen);
  if (isOpen !== wasOpen) {
    setWasOpen(isOpen);
    if (isOpen) setQuery("");
  }
  const hasQuery = query.trim().length > 0;
  const newChatAvailable = useShortcutAvailable("newChat", false);
  const newChatLabel = useShortcutLabel("newChat");
  const searchLabel = useShortcutLabel("searchChats");
  const settingsLabel = useShortcutLabel("openSettings");

  const runAndClose = (action: () => void) => () => {
    close();
    action();
  };

  const openSettings = (tab?: SettingsTab) =>
    runAndClose(() => {
      useSettingsDialogStore.getState().openDialog(tab, {
        opener: useCommandPaletteStore.getState().opener,
      });
    });

  return (
    <Command>
      <CommandInput
        placeholder={t("shell.commandPalette.placeholder")}
        value={query}
        onValueChange={setQuery}
      />
      <CommandList className="max-h-105">
        <CommandEmpty className="text-muted-foreground text-xs">
          {t("shell.commandPalette.noResults")}
        </CommandEmpty>
        <CommandGroup heading={t("shell.commandPalette.navigation")}>
          {WORKSPACES.map((workspace) => (
            <WorkspaceItem
              key={workspace.id}
              workspace={workspace}
              onSelect={runAndClose(() => void triggerShortcut(workspace.id))}
            />
          ))}
          <CommandItem
            onSelect={runAndClose(() => void navigate({ to: "/library" }))}
          >
            <HugeiconsIcon icon={LibrariesIcon} strokeWidth={1.75} />
            <span>{t("shell.navigation.library")}</span>
          </CommandItem>
          <CommandItem
            onSelect={runAndClose(() => void navigate({ to: "/api-monitor" }))}
            keywords={["api", "monitor", "requests"]}
          >
            <HugeiconsIcon icon={Globe02Icon} strokeWidth={1.75} />
            <span>{t("shell.navigation.api")}</span>
          </CommandItem>
          <CommandItem onSelect={openSettings()} keywords={["preferences"]}>
            <HugeiconsIcon icon={Settings02Icon} strokeWidth={1.75} />
            <span>{t("shell.navigation.settings")}</span>
            {settingsLabel && (
              <CommandShortcut>{settingsLabel}</CommandShortcut>
            )}
          </CommandItem>
        </CommandGroup>
        <CommandSeparator />
        <CommandGroup heading={t("shell.commandPalette.actions")}>
          {newChatAvailable && (
            <CommandItem
              onSelect={runAndClose(() => void triggerShortcut("newChat"))}
            >
              <HugeiconsIcon icon={PencilEdit02Icon} strokeWidth={1.75} />
              <span>{t("shell.navigation.newChat")}</span>
              {newChatLabel && (
                <CommandShortcut>{newChatLabel}</CommandShortcut>
              )}
            </CommandItem>
          )}
          <CommandItem
            onSelect={runAndClose(() =>
              useChatSearchStore.getState().open({
                opener: useCommandPaletteStore.getState().opener,
              }),
            )}
          >
            <HugeiconsIcon icon={Search01Icon} strokeWidth={1.75} />
            <span>{t("shell.commandPalette.searchChats")}</span>
            {searchLabel && <CommandShortcut>{searchLabel}</CommandShortcut>}
          </CommandItem>
          <CommandItem
            ref={anchorRef as React.Ref<HTMLDivElement>}
            onSelect={runAndClose(() => void toggleTheme())}
            keywords={["theme"]}
          >
            {isDark ? (
              <HugeiconsIcon icon={Sun03Icon} strokeWidth={1.75} />
            ) : (
              <Moon strokeWidth={1.75} className="size-4" />
            )}
            <span>
              {isDark
                ? t("shell.navigation.lightMode")
                : t("shell.navigation.darkMode")}
            </span>
          </CommandItem>
        </CommandGroup>
        {/* Settings pages only once the user types. */}
        {hasQuery && (
          <>
            <CommandSeparator />
            <CommandGroup heading={t("shell.navigation.settings")}>
              {SETTINGS_TABS.filter((tab) =>
                settingsTabVisible(tab, isOwner),
              ).map((tab) => (
                <CommandItem
                  key={tab}
                  keywords={DIALOG_SETTINGS_SEARCH_INDEX[tab].map((key) =>
                    t(key),
                  )}
                  onSelect={openSettings(tab)}
                >
                  <HugeiconsIcon icon={Settings02Icon} strokeWidth={1.75} />
                  <span className="text-muted-foreground">
                    {t("shell.navigation.settings")}
                  </span>
                  <span className="text-muted-foreground">→</span>
                  <span>{t(SETTINGS_TAB_LABELS[tab])}</span>
                </CommandItem>
              ))}
            </CommandGroup>
          </>
        )}
      </CommandList>
    </Command>
  );
}
