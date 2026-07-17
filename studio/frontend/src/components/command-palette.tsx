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
import { detectLocalPlatform, usePlatformStore } from "@/config/env";
import {
  clearNewChatDraft,
  useChatRuntimeStore,
  useChatSearchStore,
} from "@/features/chat";
import { useSettingsDialogStore, type SettingsTab } from "@/features/settings";
import { useT, type TranslationKey } from "@/i18n";
import { createNavigationNonce } from "@/lib/navigation-nonce";
import { useCommandPaletteStore } from "@/stores/command-palette";
import {
  ChefHatIcon,
  DashboardCircleIcon,
  DownloadSquare01Icon,
  Folder01Icon,
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

const isMacClient = detectLocalPlatform() === "mac";

// matches sidebar: drop interior bubble paths
const TestTubeOutlineIcon = TestTube01Icon.slice(
  0,
  3,
) as typeof TestTube01Icon;

const SETTINGS_TAB_ENTRIES: {
  id: SettingsTab;
  labelKey: TranslationKey;
  keywords: string[];
}[] = [
  { id: "general", labelKey: "settings.tabs.general", keywords: ["account", "password", "token"] },
  { id: "profile", labelKey: "settings.tabs.profile", keywords: ["personalization"] },
  { id: "appearance", labelKey: "settings.tabs.appearance", keywords: ["theme"] },
  { id: "resources", labelKey: "settings.tabs.resources", keywords: ["resources", "hardware", "storage"] },
  { id: "chat", labelKey: "settings.tabs.chat", keywords: ["archived"] },
  { id: "connections", labelKey: "settings.tabs.connections", keywords: ["providers"] },
  { id: "api-keys", labelKey: "settings.tabs.apiKeys", keywords: ["api keys"] },
  { id: "voice", labelKey: "settings.tabs.voice", keywords: ["dictation", "microphone", "read aloud"] },
  { id: "about", labelKey: "settings.tabs.about", keywords: ["help", "version", "updates"] },
];

export function CommandPalette() {
  const isOpen = useCommandPaletteStore((s) => s.isOpen);
  const setOpen = useCommandPaletteStore((s) => s.setOpen);

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.defaultPrevented) return;
      if (!(e.metaKey || e.ctrlKey) || e.shiftKey || e.altKey) return;
      if (e.key.toLowerCase() !== "p") return;
      e.preventDefault(); // prevents the browser print dialog
      useCommandPaletteStore.getState().toggle();
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, []);

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

function PaletteContent() {
  const t = useT();
  const navigate = useNavigate();
  const close = useCommandPaletteStore((s) => s.close);
  const chatOnly = usePlatformStore((s) => s.isChatOnly());
  const { isDark, toggleTheme, anchorRef } = useAnimatedThemeToggle();
  const [query, setQuery] = useState("");
  const hasQuery = query.trim().length > 0;

  const runAndClose = (action: () => void) => () => {
    close();
    action();
  };

  // At selection time focus still sits inside the palette (the close hasn't
  // flushed), so Settings must restore to the palette's own opener instead
  // of capturing document.activeElement.
  const openSettings = (tab?: SettingsTab) =>
    runAndClose(() => {
      useSettingsDialogStore.getState().openDialog(tab, {
        opener: useCommandPaletteStore.getState().opener,
      });
    });

  const openNewChat = runAndClose(() => {
    clearNewChatDraft();
    const chatRuntime = useChatRuntimeStore.getState();
    chatRuntime.setActiveThreadId(null);
    chatRuntime.setActiveProjectId(null);
    chatRuntime.setIncognito(false);
    // Detach the staging UI but keep any in-flight download running, like Hub.
    if (chatRuntime.pendingSelection)
      chatRuntime.abandonStagedModel({ keepDownload: true });
    void navigate({ to: "/chat", search: { new: createNavigationNonce() } });
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
          <CommandItem onSelect={runAndClose(() => navigate({ to: "/chat" }))}>
            <HugeiconsIcon icon={Message01Icon} strokeWidth={1.75} />
            <span>{t("shell.commandPalette.chat")}</span>
          </CommandItem>
          <CommandItem
            onSelect={runAndClose(() => navigate({ to: "/projects" }))}
          >
            <HugeiconsIcon icon={Folder01Icon} strokeWidth={1.75} />
            <span>{t("shell.commandPalette.projects")}</span>
          </CommandItem>
          <CommandItem
            onSelect={runAndClose(() => navigate({ to: "/hub" }))}
            keywords={["models"]}
          >
            <HugeiconsIcon icon={DashboardCircleIcon} strokeWidth={1.75} />
            <span>{t("shell.navigation.hub")}</span>
          </CommandItem>
          {/* chat-only guard redirects /studio; omit rather than dead-end */}
          {!chatOnly && (
            <CommandItem
              onSelect={runAndClose(() => navigate({ to: "/studio" }))}
              keywords={["studio", "fine-tune", "training"]}
            >
              <HugeiconsIcon icon={TestTubeOutlineIcon} strokeWidth={1.75} />
              <span>{t("shell.navigation.train")}</span>
            </CommandItem>
          )}
          <CommandItem
            onSelect={runAndClose(() => navigate({ to: "/data-recipes" }))}
            keywords={["data", "datasets"]}
          >
            <HugeiconsIcon icon={ChefHatIcon} strokeWidth={1.75} />
            <span>{t("shell.navigation.recipes")}</span>
          </CommandItem>
          <CommandItem
            onSelect={runAndClose(() => navigate({ to: "/export" }))}
            keywords={["gguf", "checkpoint"]}
          >
            <HugeiconsIcon icon={DownloadSquare01Icon} strokeWidth={1.75} />
            <span>{t("shell.navigation.export")}</span>
          </CommandItem>
          <CommandItem onSelect={openSettings()} keywords={["preferences"]}>
            <HugeiconsIcon icon={Settings02Icon} strokeWidth={1.75} />
            <span>{t("shell.navigation.settings")}</span>
            <CommandShortcut>{isMacClient ? "⌘," : "Ctrl+,"}</CommandShortcut>
          </CommandItem>
        </CommandGroup>
        <CommandSeparator />
        <CommandGroup heading={t("shell.commandPalette.actions")}>
          <CommandItem onSelect={openNewChat}>
            <HugeiconsIcon icon={PencilEdit02Icon} strokeWidth={1.75} />
            <span>{t("shell.navigation.newChat")}</span>
            <CommandShortcut>
              {isMacClient ? "⌘⇧O" : "Ctrl+Shift+O"}
            </CommandShortcut>
          </CommandItem>
          <CommandItem
            onSelect={runAndClose(() => useChatSearchStore.getState().open())}
          >
            <HugeiconsIcon icon={Search01Icon} strokeWidth={1.75} />
            <span>{t("shell.commandPalette.searchChats")}</span>
            <CommandShortcut>{isMacClient ? "⌘K" : "Ctrl+K"}</CommandShortcut>
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
        {hasQuery && (
          <>
            <CommandSeparator />
            <CommandGroup heading={t("shell.navigation.settings")}>
              {SETTINGS_TAB_ENTRIES.map((tab) => (
                <CommandItem
                  key={tab.id}
                  keywords={tab.keywords}
                  onSelect={openSettings(tab.id)}
                >
                  <HugeiconsIcon icon={Settings02Icon} strokeWidth={1.75} />
                  <span className="text-muted-foreground">
                    {t("shell.navigation.settings")}
                  </span>
                  <span className="text-muted-foreground">→</span>
                  <span>{t(tab.labelKey)}</span>
                </CommandItem>
              ))}
            </CommandGroup>
          </>
        )}
      </CommandList>
    </Command>
  );
}
