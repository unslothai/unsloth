// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Dialog, DialogContent } from "@/components/ui/dialog";
import { cn } from "@/lib/utils";
import {
  Cancel01Icon,
  Key01Icon,
  Message01Icon,
  PaintBrush02Icon,
  Settings02Icon,
  SparklesIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { motion } from "motion/react";
import { useSettingsDialogStore, type SettingsTab } from "./stores/settings-dialog-store";
import { AboutTab } from "./tabs/about-tab";
import { ApiKeysTab } from "./tabs/api-keys-tab";
import { AppearanceTab } from "./tabs/appearance-tab";
import { ChatTab } from "./tabs/chat-tab";
import { GeneralTab } from "./tabs/general-tab";

interface TabDef {
  id: SettingsTab;
  label: string;
  icon: typeof Settings02Icon;
}

const TABS: TabDef[] = [
  { id: "general", label: "General", icon: Settings02Icon },
  { id: "appearance", label: "Appearance", icon: PaintBrush02Icon },
  { id: "chat", label: "Chat", icon: Message01Icon },
  { id: "api-keys", label: "API Keys", icon: Key01Icon },
  { id: "about", label: "About", icon: SparklesIcon },
];

function TabPlaceholder({ id }: { id: SettingsTab }) {
  return (
    <div className="flex flex-col gap-6">
      <header className="flex flex-col gap-1">
        <h1 className="text-lg font-semibold font-heading capitalize">{id}</h1>
        <p className="text-xs text-muted-foreground">
          Placeholder — wired in a later task.
        </p>
      </header>
    </div>
  );
}

function renderTab(tab: SettingsTab) {
  switch (tab) {
    case "general":
      return <GeneralTab />;
    case "appearance":
      return <AppearanceTab />;
    case "chat":
      return <ChatTab />;
    case "api-keys":
      return <ApiKeysTab />;
    case "about":
      return <AboutTab />;
    default:
      return <TabPlaceholder id={tab} />;
  }
}

export function SettingsDialog() {
  const open = useSettingsDialogStore((s) => s.open);
  const activeTab = useSettingsDialogStore((s) => s.activeTab);
  const setActiveTab = useSettingsDialogStore((s) => s.setActiveTab);
  const closeDialog = useSettingsDialogStore((s) => s.closeDialog);

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
        <div className="flex h-full min-h-0">
          <aside className="flex w-[200px] shrink-0 flex-col border-r border-border bg-muted/20 p-2">
            <nav className="flex flex-col gap-0.5">
              {TABS.map((tab) => {
                const active = activeTab === tab.id;
                return (
                  <button
                    key={tab.id}
                    type="button"
                    onClick={() => setActiveTab(tab.id)}
                    className={cn(
                      "relative flex h-9 items-center gap-2 rounded-md px-2.5 text-sm font-medium transition-colors",
                      active
                        ? "text-foreground"
                        : "text-muted-foreground hover:text-foreground",
                    )}
                  >
                    {active && (
                      <motion.span
                        layoutId="settings-active-pill"
                        className="absolute inset-0 rounded-md bg-accent"
                        transition={{
                          type: "spring",
                          stiffness: 500,
                          damping: 35,
                          mass: 0.5,
                        }}
                      />
                    )}
                    <HugeiconsIcon
                      icon={tab.icon}
                      className="relative z-10 size-4"
                    />
                    <span className="relative z-10">{tab.label}</span>
                  </button>
                );
              })}
            </nav>
          </aside>

          <main className="relative flex min-w-0 flex-1 flex-col">
            <button
              type="button"
              onClick={closeDialog}
              className="absolute top-3 right-3 z-10 flex size-7 items-center justify-center rounded-md text-muted-foreground transition-colors hover:bg-accent hover:text-foreground"
              aria-label="Close settings"
            >
              <HugeiconsIcon icon={Cancel01Icon} className="size-4" />
            </button>
            <div className="flex min-h-0 flex-1 flex-col overflow-y-auto p-6 pr-12">
              {renderTab(activeTab)}
            </div>
          </main>
        </div>
      </DialogContent>
    </Dialog>
  );
}
