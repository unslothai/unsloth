// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogTitle,
} from "@/components/ui/dialog";
import { useT, type TranslationKey } from "@/i18n";
import { cn } from "@/lib/utils";
import {
  Cancel01Icon,
  CloudIcon,
  Globe02Icon,
  HelpCircleIcon,
  Message01Icon,
  PaintBrush02Icon,
  Settings02Icon,
  UserIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { motion, useReducedMotion } from "motion/react";
import { useEffect, useRef } from "react";
import {
  useSettingsDialogStore,
  type SettingsTab,
} from "./stores/settings-dialog-store";
import { AboutTab } from "./tabs/about-tab";
import { ApiKeysTab } from "./tabs/api-keys-tab";
import { AppearanceTab } from "./tabs/appearance-tab";
import { ChatTab } from "./tabs/chat-tab";
import { ConnectionsTab } from "./tabs/connections-tab";
import { GeneralTab } from "./tabs/general-tab";
import { ProfileTab } from "./tabs/profile-tab";

interface TabDef {
  id: SettingsTab;
  labelKey: TranslationKey;
  icon: typeof Settings02Icon;
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
  { id: "chat", labelKey: "settings.tabs.chat", icon: Message01Icon },
  {
    id: "connections",
    labelKey: "settings.tabs.connections",
    icon: CloudIcon,
    badgeKey: "common.new",
  },
  {
    id: "api-keys",
    labelKey: "settings.tabs.apiKeys",
    icon: Globe02Icon,
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
    case "chat":
      return <ChatTab />;
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
    chat: null,
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
    <Dialog open={open} onOpenChange={(o) => !o && closeDialog()}>
      <DialogContent
        showCloseButton={false}
        overlayClassName="bg-background/40"
        onCloseAutoFocus={(e) => {
          // Restore focus to the element that triggered openDialog().
          // Radix's FocusScope races our rAF-scheduled tab-button focus
          // and loses the previous-focus reference, so we restore by hand.
          if (opener && opener.isConnected) {
            e.preventDefault();
            opener.focus({ preventScroll: true });
          }
        }}
        className={cn(
          // Cap at 820px but shrink to the viewport so we don't clip
          // on iPad-portrait widths (640-820px) where the fixed
          // `w-[820px]` overflows by 26px on each side.
          "settings-surface !max-w-[min(820px,calc(100vw-2rem))] h-[560px] w-[min(820px,calc(100vw-2rem))] p-0 overflow-hidden",
          // Soft shadow only, no outline ring.
          "shadow-border rounded-xl ring-0",
          "max-sm:h-dvh max-sm:w-dvw max-sm:!max-w-none max-sm:rounded-none",
        )}
      >
        <DialogTitle className="sr-only">{t("settings.dialog.title")}</DialogTitle>
        <DialogDescription className="sr-only">
          {t("settings.dialog.description")}
        </DialogDescription>
        <div className="flex h-full min-h-0 max-sm:flex-col">
          <aside className="font-heading flex w-[216px] shrink-0 flex-col border-r border-border bg-muted/20 p-2 max-sm:w-full max-sm:border-r-0 max-sm:border-b">
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
                      "relative flex h-[32px] items-center gap-2.5 rounded-[8px] px-2.5 text-[14.5px] leading-[19px] tracking-nav font-medium transition-colors",
                      "max-sm:shrink-0",
                      "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-1 focus-visible:ring-offset-background",
                      active
                        ? "text-black dark:text-white"
                        : "text-[#383835] dark:text-[#c7c7c4] hover:bg-[#ececec] dark:hover:bg-[#2d2f33] hover:text-black dark:hover:text-white",
                    )}
                  >
                    {active && (
                      <motion.span
                        layoutId="settings-active-pill"
                        className="absolute inset-0 rounded-[8px] bg-[#ececec] dark:bg-[#2d2f33]"
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
                    <HugeiconsIcon
                      icon={tab.icon}
                      strokeWidth={1.75}
                      className="relative z-10 size-icon"
                    />
                    <span className="relative z-10 min-w-0 truncate">
                      {t(tab.labelKey)}
                    </span>
                    {tab.badgeKey ? (
                      <span className="relative z-10 ml-auto rounded-[6px] border border-emerald-500/25 bg-emerald-500/10 px-1.5 py-0.5 text-[10px] leading-none font-semibold text-emerald-700 dark:text-emerald-300">
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
              className="absolute top-3 right-3 z-10 flex size-7 items-center justify-center rounded-full text-[#383835] dark:text-[#c7c7c4] transition-colors hover:bg-[#ececec] dark:hover:bg-[#2d2f33] hover:text-black dark:hover:text-white focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
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
  );
}
