// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogTitle,
} from "@/components/ui/dialog";
import { cn } from "@/lib/utils";
import {
  Cancel01Icon,
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
import { useSettingsDialogStore, type SettingsTab } from "./stores/settings-dialog-store";
import { AboutTab } from "./tabs/about-tab";
import { ApiKeysTab } from "./tabs/api-keys-tab";
import { AppearanceTab } from "./tabs/appearance-tab";
import { ChatTab } from "./tabs/chat-tab";
import { GeneralTab } from "./tabs/general-tab";
import { ProfileTab } from "./tabs/profile-tab";

interface TabDef {
  id: SettingsTab;
  label: string;
  icon: typeof Settings02Icon;
  badge?: string;
}

const TABS: TabDef[] = [
  { id: "general", label: "General", icon: Settings02Icon },
  { id: "profile", label: "Profile", icon: UserIcon },
  { id: "appearance", label: "Appearance", icon: PaintBrush02Icon },
  { id: "chat", label: "Chat", icon: Message01Icon },
  { id: "api-keys", label: "API", icon: Globe02Icon, badge: "New" },
  { id: "about", label: "Help", icon: HelpCircleIcon },
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
    case "api-keys":
      return <ApiKeysTab />;
    case "about":
      return <AboutTab />;
  }
}

export function SettingsDialog() {
  const open = useSettingsDialogStore((s) => s.open);
  const activeTab = useSettingsDialogStore((s) => s.activeTab);
  const setActiveTab = useSettingsDialogStore((s) => s.setActiveTab);
  const closeDialog = useSettingsDialogStore((s) => s.closeDialog);
  const reduced = useReducedMotion();
  const tabButtonRefs = useRef<Record<SettingsTab, HTMLButtonElement | null>>({
    general: null,
    profile: null,
    appearance: null,
    chat: null,
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
        className={cn(
          "!max-w-none h-[560px] w-[820px] p-0 overflow-hidden",
          "shadow-border rounded-xl border-border",
          "sm:h-[560px] sm:w-[820px]",
          "max-sm:h-dvh max-sm:w-dvw max-sm:rounded-none",
        )}
      >
        <DialogTitle className="sr-only">Settings</DialogTitle>
        <DialogDescription className="sr-only">
          Manage your Unsloth Studio preferences.
        </DialogDescription>
        <div className="flex h-full min-h-0">
          <aside className="font-heading flex w-[200px] shrink-0 flex-col border-r border-border bg-muted/20 p-2">
            <nav className="flex flex-col gap-0.5">
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
                      "relative flex h-[30px] items-center gap-2.5 rounded-[8px] px-2.5 text-sm font-medium transition-colors",
                      "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-1 focus-visible:ring-offset-background",
                      active
                        ? "text-black dark:text-white"
                        : "text-[#383835] dark:text-[#c7c7c4] hover:bg-[#ececec] dark:hover:bg-[#2e3035] hover:text-black dark:hover:text-white",
                    )}
                  >
                    {active && (
                      <motion.span
                        layoutId="settings-active-pill"
                        className="absolute inset-0 rounded-[8px] bg-[#ececec] dark:bg-[#2e3035]"
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
                      strokeWidth={1.5}
                      className="relative z-10 size-[18px]"
                    />
                    <span className="relative z-10 min-w-0 truncate">{tab.label}</span>
                    {tab.badge ? (
                      <span className="relative z-10 ml-auto rounded-[6px] border border-emerald-500/25 bg-emerald-500/10 px-1.5 py-0.5 text-[10px] leading-none font-semibold text-emerald-700 dark:text-emerald-300">
                        {tab.badge}
                      </span>
                    ) : null}
                  </button>
                );
              })}
            </nav>
          </aside>

          <main className="relative flex min-w-0 flex-1 flex-col">
            <button
              type="button"
              onClick={closeDialog}
              className="absolute top-3 right-3 z-10 flex size-7 items-center justify-center rounded-[8px] text-[#383835] dark:text-[#c7c7c4] transition-colors hover:bg-[#ececec] dark:hover:bg-[#2e3035] hover:text-black dark:hover:text-white focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
              aria-label="Close settings"
            >
              <HugeiconsIcon icon={Cancel01Icon} className="size-4" />
            </button>
            <div className="flex min-h-0 min-w-0 flex-1 flex-col overflow-y-auto p-6">
              {renderTab(activeTab)}
            </div>
          </main>
        </div>
      </DialogContent>
    </Dialog>
  );
}
