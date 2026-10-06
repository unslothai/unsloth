// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useAnimatedThemeToggle } from "@/components/ui/animated-theme-toggler";
import {
  Command,
  CommandDialog,
  CommandEmpty,
  CommandGroup,
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
import { cn } from "@/lib/utils";
import { useCommandPaletteStore } from "@/stores/command-palette";
import {
  AudioWave01Icon,
  Cancel01Icon,
  ChefHatIcon,
  DashboardCircleIcon,
  DownloadSquare01Icon,
  FlimSlateIcon,
  Folder01Icon,
  Image03Icon,
  LibrariesIcon,
  Message01Icon,
  PencilEdit02Icon,
  Search01Icon,
  Settings02Icon,
  Sun03Icon,
  TestTube01Icon,
  ApiIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useNavigate, useRouterState } from "@tanstack/react-router";
import { Command as CommandPrimitive } from "cmdk";
import { Moon } from "lucide-react";
import { type ComponentProps, useEffect, useState } from "react";

// Surface, header and rows follow chat search (chat-search-dialog.tsx).
const ROW_CLASS =
  "gap-3 rounded-full px-3 py-2.5 text-ui-13 font-medium data-selected:bg-muted data-selected:text-foreground *:[svg]:text-muted-foreground data-selected:*:[svg]:text-muted-foreground";

function PaletteItem({
  className,
  ...props
}: ComponentProps<typeof CommandItem>) {
  return <CommandItem className={cn(ROW_CLASS, className)} {...props} />;
}

function PaletteShortcut({
  className,
  ...props
}: ComponentProps<typeof CommandShortcut>) {
  return (
    <CommandShortcut
      className={cn("text-ui-11 font-normal", className)}
      {...props}
    />
  );
}

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
  sandbox: "settings.tabs.sandbox",
  "api-keys": "settings.tabs.apiKeys",
  "remote-lan": "settings.tabs.remoteLan",
  agents: "settings.tabs.agents",
  "keyboard-shortcuts": "settings.tabs.keyboardShortcuts",
  browser: "browser.settingsTitle",
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

  // A chord pressed over the palette (⌘, / ⌘K, a workspace or new chat) acts behind it; it gives way.
  const settingsOpen = useSettingsDialogStore((s) => s.open);
  const chatSearchOpen = useChatSearchStore((s) => s.isOpen);
  const href = useRouterState({ select: (s) => s.location.href });
  // biome-ignore lint/correctness/useExhaustiveDependencies: href is the trigger, not an input.
  useEffect(() => {
    useCommandPaletteStore.getState().close();
  }, [href]);
  useEffect(() => {
    if (settingsOpen || chatSearchOpen) {
      useCommandPaletteStore.getState().close();
    }
  }, [settingsOpen, chatSearchOpen]);

  return (
    <CommandDialog
      open={isOpen}
      onOpenChange={setOpen}
      className="chat-search-surface rounded-3xl! max-sm:rounded-none! top-[calc(50%+var(--studio-window-chrome-top,0px)/2)] -translate-y-1/2 w-[calc(635px*var(--ui-space-scale,1))] max-w-[calc(100%-2rem)] gap-0 p-0 ring-0 duration-[180ms] ease-[cubic-bezier(0.16,1,0.3,1)] sm:max-w-[calc(635px*var(--ui-space-scale,1))]"
      overlayClassName="bg-transparent supports-backdrop-filter:backdrop-blur-none"
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
    <PaletteItem onSelect={onSelect} keywords={workspace.aliases}>
      <HugeiconsIcon icon={workspace.icon} strokeWidth={1.75} />
      <span>{t(workspace.labelKey)}</span>
      {label && <PaletteShortcut>{label}</PaletteShortcut>}
    </PaletteItem>
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
    <Command className="rounded-3xl p-0">
      <div className="flex items-center gap-3 border-b border-border/40 px-4 py-3">
        <HugeiconsIcon
          icon={Search01Icon}
          strokeWidth={2}
          className="size-4 shrink-0 text-muted-foreground"
        />
        <CommandPrimitive.Input
          placeholder={t("shell.commandPalette.placeholder")}
          value={query}
          onValueChange={setQuery}
          className="flex-1 bg-transparent text-sm outline-none placeholder:text-muted-foreground"
        />
        {/* cmdk runs the selected row on an Enter from anywhere inside it, so keep this one on the button. */}
        <button
          type="button"
          onClick={close}
          onKeyDown={(e) => {
            if (e.key === "Enter") e.stopPropagation();
          }}
          className="flex size-6 items-center justify-center rounded-md text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
          aria-label={t("common.close")}
        >
          <HugeiconsIcon
            icon={Cancel01Icon}
            strokeWidth={2}
            className="size-4"
          />
        </button>
      </div>
      <CommandList className="cmd-native-scrollbar hover-scrollbar h-[calc(420px*var(--ui-space-scale,1))] max-h-[60dvh] p-1">
        <CommandEmpty className="py-6 text-center text-xs text-muted-foreground">
          {t("shell.commandPalette.noResults")}
        </CommandEmpty>
        <CommandGroup
          heading={t("shell.commandPalette.navigation")}
          className="p-0"
        >
          {WORKSPACES.map((workspace) => (
            <WorkspaceItem
              key={workspace.id}
              workspace={workspace}
              onSelect={runAndClose(() => void triggerShortcut(workspace.id))}
            />
          ))}
          <PaletteItem
            onSelect={runAndClose(() => void navigate({ to: "/library" }))}
          >
            <HugeiconsIcon icon={LibrariesIcon} strokeWidth={1.75} />
            <span>{t("shell.navigation.library")}</span>
          </PaletteItem>
          <PaletteItem
            onSelect={runAndClose(() => void navigate({ to: "/api-monitor" }))}
            keywords={["api", "monitor", "requests"]}
          >
            <HugeiconsIcon icon={ApiIcon} strokeWidth={1.75} />
            <span>{t("shell.navigation.api")}</span>
          </PaletteItem>
          <PaletteItem onSelect={openSettings()} keywords={["preferences"]}>
            <HugeiconsIcon icon={Settings02Icon} strokeWidth={1.75} />
            <span>{t("shell.navigation.settings")}</span>
            {settingsLabel && (
              <PaletteShortcut>{settingsLabel}</PaletteShortcut>
            )}
          </PaletteItem>
        </CommandGroup>
        <CommandSeparator />
        <CommandGroup
          heading={t("shell.commandPalette.actions")}
          className="p-0"
        >
          {newChatAvailable && (
            <PaletteItem
              onSelect={runAndClose(() => void triggerShortcut("newChat"))}
            >
              <HugeiconsIcon icon={PencilEdit02Icon} strokeWidth={1.75} />
              <span>{t("shell.navigation.newChat")}</span>
              {newChatLabel && (
                <PaletteShortcut>{newChatLabel}</PaletteShortcut>
              )}
            </PaletteItem>
          )}
          <PaletteItem
            onSelect={runAndClose(() =>
              useChatSearchStore.getState().open({
                opener: useCommandPaletteStore.getState().opener,
              }),
            )}
          >
            <HugeiconsIcon icon={Search01Icon} strokeWidth={1.75} />
            <span>{t("shell.commandPalette.searchChats")}</span>
            {searchLabel && <PaletteShortcut>{searchLabel}</PaletteShortcut>}
          </PaletteItem>
          <PaletteItem
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
          </PaletteItem>
        </CommandGroup>
        {/* Settings pages only once the user types. */}
        {hasQuery && (
          <>
            <CommandSeparator />
            <CommandGroup
              heading={t("shell.navigation.settings")}
              className="p-0"
            >
              {SETTINGS_TABS.filter((tab) =>
                settingsTabVisible(tab, isOwner),
              ).map((tab) => (
                <PaletteItem
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
                </PaletteItem>
              ))}
            </CommandGroup>
          </>
        )}
      </CommandList>
    </Command>
  );
}
