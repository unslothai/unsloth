// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogTitle,
} from "@/components/ui/dialog";
import { type TranslationKey, useT } from "@/i18n";
import { cn } from "@/lib/utils";
import { MicIcon } from "@/lib/mic-icon";
import {
  Cancel01Icon,
  CloudIcon,
  CpuIcon,
  Globe02Icon,
  HelpCircleIcon,
  Message01Icon,
  PaintBrush02Icon,
  Settings02Icon,
  UserIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { motion, useReducedMotion } from "motion/react";
import { type FC, useEffect, useRef } from "react";
import {
  type SettingsTab,
  useSettingsDialogStore,
} from "./stores/settings-dialog-store";
import { AboutTab } from "./tabs/about-tab";
import { ApiKeysTab } from "./tabs/api-keys-tab";
import { AppearanceTab } from "./tabs/appearance-tab";
import { ChatTab } from "./tabs/chat-tab";
import { ConnectionsTab } from "./tabs/connections-tab";
import { GeneralTab } from "./tabs/general-tab";
import { ProfileTab } from "./tabs/profile-tab";
import { ResourcesTab } from "./tabs/resources-tab";
import { VoiceTab } from "./tabs/voice-tab";
import { FloatingMonitor } from "@/components/floating-monitor";

interface TabDef {
  id: SettingsTab;
  labelKey: TranslationKey;
  icon?: typeof Settings02Icon;
  /** Plain component icon, for icons shared with chat (not hugeicons). */
  iconComponent?: FC<{ className?: string }>;
  badgeKey?: TranslationKey;
}

const TABS: TabDef[] = [
  { id: "general", labelKey: "settings.tabs.general", icon: Settings02Icon },
  { id: "profile", labelKey: "settings.tabs.profile", icon: UserIcon },
  {
    id: "appearance",
    labelKey: "settings.tabs.appearance",
    icon: PaintBrush02Icon,
  },
  {
    id: "resources",
    labelKey: "settings.tabs.resources",
    icon: CpuIcon,
    badgeKey: "common.new",
  },
  {
    id: "chat",
    labelKey: "settings.tabs.chat",
    icon: Message01Icon,
    badgeKey: "common.new",
  },
  {
    id: "api-keys",
    labelKey: "settings.tabs.apiKeys",
    icon: Globe02Icon,
  },
  {
    id: "connections",
    labelKey: "settings.tabs.connections",
    icon: CloudIcon,
  },
  {
    id: "voice",
    labelKey: "settings.tabs.voice",
    iconComponent: MicIcon,
    badgeKey: "common.new",
  },
  { id: "about", labelKey: "settings.tabs.about", icon: HelpCircleIcon },
];

function renderTab(tab: SettingsTab) {
  switch (tab) {
    case "general":
      return <GeneralTab />;
    case "profile":
      return <ProfileTab />;
    case "appearance":
      return <AppearanceTab />;
    case "resources":
      return <ResourcesTab />;
    case "chat":
      return <ChatTab />;
    case "voice":
      return <VoiceTab />;
    case "connections":
      return <ConnectionsTab />;
    case "api-keys":
      return <ApiKeysTab />;
    case "about":
      return <AboutTab />;
  }
}

export function SettingsDialog() {
  const t = useT();
  const open = useSettingsDialogStore((s) => s.open);
  const activeTab = useSettingsDialogStore((s) => s.activeTab);
  const setActiveTab = useSettingsDialogStore((s) => s.setActiveTab);
  const closeDialog = useSettingsDialogStore((s) => s.closeDialog);
  const opener = useSettingsDialogStore((s) => s.opener);
  const reduced = useReducedMotion();
  const tabButtonRefs = useRef<Record<SettingsTab, HTMLButtonElement | null>>({
    general: null,
    profile: null,
    appearance: null,
    resources: null,
    chat: null,
    voice: null,
    connections: null,
    "api-keys": null,
    about: null,
  });

  useEffect(() => {
    if (!open) return;
    const frame = window.requestAnimationFrame(() => {
      tabButtonRefs.current[activeTab]?.focus({ preventScroll: true });
    });
    return () => window.cancelAnimationFrame(frame);
  }, [open, activeTab]);

  return (
    <>
      <Dialog open={open} onOpenChange={(o) => !o && closeDialog()}>
        <DialogContent
          showCloseButton={false}
          overlayClassName="bg-black/30 supports-backdrop-filter:backdrop-blur-[2px]"
          onCloseAutoFocus={(e) => {
            // Restore focus to the element that triggered openDialog(). Radix's
            // FocusScope races our rAF-scheduled tab focus and loses the
            // previous-focus reference, so restore it by hand.
            if (opener && opener.isConnected) {
              e.preventDefault();
              opener.focus({ preventScroll: true });
            }
          }}
          className={cn(
            // Cap at 820px but shrink to the viewport so it doesn't clip on
            // iPad-portrait widths (640-820px) where fixed `w-[820px]` overflows.
            "settings-surface !max-w-[min(820px,calc(100vw-2rem))] h-[560px] w-[min(820px,calc(100vw-2rem))] p-0 overflow-hidden",
            // Soft shadow, no outline ring. Pin --radius to the light value so
            // corner rounding matches in dark mode.
            "shadow-border rounded-xl ring-0 [--radius:1.1rem]",
            "max-sm:h-dvh max-sm:w-dvw max-sm:!max-w-none max-sm:rounded-none",
          )}
        >
          <DialogTitle className="sr-only">
            {t("settings.dialog.title")}
          </DialogTitle>
          <DialogDescription className="sr-only">
            {t("settings.dialog.description")}
          </DialogDescription>
          <div className="flex h-full min-h-0 max-sm:flex-col">
            <aside className="font-heading flex w-[216px] shrink-0 flex-col border-r border-sidebar-border bg-muted/20 p-2 dark:border-r-0 max-sm:w-full max-sm:border-r-0 max-sm:border-b max-sm:border-sidebar-border">
              <h2 className="pl-3 pr-2.5 pt-3.5 pb-3.5 text-[19px] font-semibold text-foreground max-sm:hidden">
                {t("settings.dialog.title")}
              </h2>
              <nav className="flex flex-col gap-0.5 max-sm:flex-row max-sm:overflow-x-auto">
                {TABS.map((tab) => {
                  const active = activeTab === tab.id;
                  return (
                    <button
                      key={tab.id}
                      ref={(node) => {
                        tabButtonRefs.current[tab.id] = node;
                      }}
                      type="button"
                      onClick={() => setActiveTab(tab.id)}
                      className={cn(
                        "relative flex h-[32px] items-center gap-2.5 rounded-full pl-3 pr-2.5 text-[14.5px] leading-[19px] tracking-nav font-medium transition-colors",
                        "max-sm:shrink-0",
                        "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-1 focus-visible:ring-offset-background",
                        active
                          ? "text-black dark:text-white"
                          : "text-[#383835] dark:text-[#c7c7c4] hover:bg-[#ececec] dark:hover:bg-[#3a3d43] hover:text-black dark:hover:text-white",
                      )}
                    >
                      {active && (
                        <motion.span
                          layoutId="settings-active-pill"
                          className="absolute inset-0 rounded-full bg-[#ececec] dark:bg-[#3a3d43]"
                          transition={
                            reduced
                              ? { duration: 0 }
                              : {
                                type: "spring",
                                stiffness: 500,
                                damping: 35,
                                mass: 0.5,
                              }
                          }
                        />
                      )}
                      {tab.iconComponent ? (
                        <tab.iconComponent className="relative z-10 size-icon" />
                      ) : tab.icon ? (
                        <HugeiconsIcon
                          icon={tab.icon}
                          strokeWidth={1.75}
                          className="relative z-10 size-icon"
                        />
                      ) : null}
                      <span className="relative z-10 min-w-0 truncate">
                        {t(tab.labelKey)}
                      </span>
                      {tab.badgeKey ? (
                        <span className="relative z-10 ml-auto rounded-full bg-emerald-500/10 px-2 py-1 text-[10px] leading-none font-semibold text-emerald-700 dark:text-emerald-300">
                          {t(tab.badgeKey)}
                        </span>
                      ) : null}
                    </button>
                  );
                })}
              </nav>
            </aside>

            <main className="relative flex min-h-0 min-w-0 flex-1 flex-col">
              <button
                type="button"
                onClick={closeDialog}
                className="absolute top-3 right-3 z-10 flex size-7 items-center justify-center rounded-full text-[#383835] dark:text-[#c7c7c4] transition-colors hover:bg-[#ececec] dark:hover:bg-[#3a3d43] hover:text-black dark:hover:text-white focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                aria-label={t("settings.dialog.closeAriaLabel")}
              >
                <HugeiconsIcon icon={Cancel01Icon} className="size-4" />
              </button>
              <div className="hover-scrollbar flex min-h-0 min-w-0 flex-1 flex-col overflow-y-auto p-6 [scrollbar-gutter:stable]">
                {renderTab(activeTab)}
              </div>
            </main>
          </div>
        </DialogContent>
      </Dialog>
      <FloatingMonitor />
    </>
  );
}
