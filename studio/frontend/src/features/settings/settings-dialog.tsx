// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { FloatingMonitor } from "@/components/floating-monitor";
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
  BotIcon,
  Cancel01Icon,
  CloudIcon,
  CpuIcon,
  DatabaseSettingIcon,
  Globe02Icon,
  HelpCircleIcon,
  Message01Icon,
  PaintBrush02Icon,
  Search01Icon,
  Settings02Icon,
  UserIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { motion, useReducedMotion } from "motion/react";
import {
  type FC,
  useDeferredValue,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { SETTINGS_SEARCH_INDEX } from "./settings-search";
import {
  type SettingsTab,
  useSettingsDialogStore,
} from "./stores/settings-dialog-store";
import { AboutTab } from "./tabs/about-tab";
import { AgentsTab } from "./tabs/agents-tab";
import { ApiKeysTab } from "./tabs/api-keys-tab";
import { AppearanceTab } from "./tabs/appearance-tab";
import { ChatTab } from "./tabs/chat-tab";
import { ConnectionsTab } from "./tabs/connections-tab";
import { DataTab } from "./tabs/data-tab";
import { GeneralTab } from "./tabs/general-tab";
import { ProfileTab } from "./tabs/profile-tab";
import { ResourcesTab } from "./tabs/resources-tab";
import { VoiceTab } from "./tabs/voice-tab";

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
  },
  {
    id: "chat",
    labelKey: "settings.tabs.chat",
    icon: Message01Icon,
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
    id: "agents",
    labelKey: "settings.tabs.agents",
    icon: BotIcon,
    badgeKey: "common.new",
  },
  {
    id: "voice",
    labelKey: "settings.tabs.voice",
    iconComponent: MicIcon,
    badgeKey: "common.new",
  },
  {
    id: "data",
    labelKey: "settings.tabs.data",
    icon: DatabaseSettingIcon,
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
    case "data":
      return <DataTab />;
    case "api-keys":
      return <ApiKeysTab />;
    case "agents":
      return <AgentsTab />;
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
  // Mounting a heavy tab panel (System, Connections) in the same commit as
  // the nav highlight makes the highlight lag the click. Render the panel
  // from a deferred value so the nav updates first.
  const panelTab = useDeferredValue(activeTab);
  const [query, setQuery] = useState("");

  const results = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return null;
    return TABS.map((tab) => {
      const tabLabel = t(tab.labelKey);
      const entries = SETTINGS_SEARCH_INDEX[tab.id]
        .map((key) => t(key))
        .filter((label) => label.toLowerCase().includes(q));
      const deduped = [...new Set(entries)];
      return {
        tab,
        tabLabel,
        entries: deduped,
        tabMatches: tabLabel.toLowerCase().includes(q),
      };
    }).filter((r) => r.tabMatches || r.entries.length > 0);
  }, [query, t]);

  const [pendingScroll, setPendingScroll] = useState<{
    tab: SettingsTab;
    entry: string;
  } | null>(null);
  const mainScrollRef = useRef<HTMLDivElement | null>(null);

  const openResult = (tab: SettingsTab, entry?: string) => {
    setActiveTab(tab);
    setQuery("");
    setPendingScroll(entry ? { tab, entry } : null);
  };

  // Scroll to the row/section a search result points at once the tab has
  // rendered, and flash it so the eye lands on the right place. The tab panel
  // renders deferred, so retry across frames until the row exists instead of
  // racing a single fixed delay (which silently missed under render lag).
  useEffect(() => {
    if (!pendingScroll) return;
    // Wait until the destination tab is mounted before matching, so a same-named
    // row in the previous tab (for example "Storage") is not scrolled to instead.
    if (panelTab !== pendingScroll.tab) return;
    let frame = 0;
    let tries = 0;
    const attempt = () => {
      const root = mainScrollRef.current;
      const target = root
        ? [
            ...root.querySelectorAll<HTMLElement>("[data-settings-label]"),
          ].find((el) => el.dataset.settingsLabel === pendingScroll.entry)
        : undefined;
      if (target) {
        target.scrollIntoView({ behavior: "smooth", block: "center" });
        target.classList.add("settings-search-hit");
        window.setTimeout(
          () => target.classList.remove("settings-search-hit"),
          1600,
        );
        setPendingScroll(null);
      } else if (tries++ < 30) {
        frame = window.requestAnimationFrame(attempt);
      } else {
        setPendingScroll(null);
      }
    };
    frame = window.requestAnimationFrame(attempt);
    return () => window.cancelAnimationFrame(frame);
  }, [pendingScroll, panelTab]);

  useEffect(() => {
    if (!open) setQuery("");
  }, [open]);
  const tabButtonRefs = useRef<Record<SettingsTab, HTMLButtonElement | null>>({
    general: null,
    profile: null,
    appearance: null,
    resources: null,
    chat: null,
    voice: null,
    connections: null,
    data: null,
    "api-keys": null,
    agents: null,
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
            // Cap at 880px but shrink to the viewport so it doesn't clip on
            // iPad-portrait widths where a fixed width overflows.
            "settings-surface !max-w-[min(880px,calc(100vw-2rem))] h-[560px] w-[min(880px,calc(100vw-2rem))] p-0 overflow-hidden",
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
          {/* Keep tab content from expanding the dialog grid. */}
          <div className="flex h-full min-h-0 min-w-0 w-full max-sm:flex-col">
            <aside className="font-heading flex w-[248px] shrink-0 flex-col border-r border-sidebar-border bg-muted/20 p-2 dark:border-r-0 max-sm:w-full max-sm:border-r-0 max-sm:border-b max-sm:border-sidebar-border">
              <div className="relative mx-1 mt-3 mb-2 max-sm:hidden">
                <HugeiconsIcon
                  icon={Search01Icon}
                  strokeWidth={2}
                  className="pointer-events-none absolute top-1/2 left-2.5 size-4 -translate-y-1/2 text-muted-foreground"
                />
                <input
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Escape" && query) {
                      e.stopPropagation();
                      setQuery("");
                    }
                  }}
                  placeholder={t("settings.dialog.searchPlaceholder")}
                  aria-label={t("settings.dialog.searchPlaceholder")}
                  className="h-8 w-full rounded-full border border-border bg-background pr-8 pl-8 text-sm text-foreground outline-none transition-colors placeholder:text-muted-foreground focus-visible:border-ring dark:focus-visible:border-transparent dark:focus-visible:bg-white/[0.12] dark:border-transparent dark:bg-white/[0.06]"
                />
                {query && (
                  <button
                    type="button"
                    onClick={() => setQuery("")}
                    aria-label={t("settings.dialog.closeAriaLabel")}
                    className="absolute top-1/2 right-2 flex size-5 -translate-y-1/2 items-center justify-center rounded-full text-muted-foreground hover:text-foreground"
                  >
                    <HugeiconsIcon icon={Cancel01Icon} className="size-3.5" />
                  </button>
                )}
              </div>
              {results ? (
                <div className="hover-scrollbar flex min-h-0 flex-1 flex-col gap-1 overflow-y-auto px-1 pb-1 max-sm:hidden">
                  {results.length === 0 ? (
                    <p className="px-3 py-2 text-sm text-muted-foreground">
                      {t("settings.dialog.searchNoResults")}
                    </p>
                  ) : (
                    results.map(({ tab, tabLabel, entries }) => (
                      <div key={tab.id} className="flex flex-col">
                        <button
                          type="button"
                          onClick={() => openResult(tab.id)}
                          className="flex h-[30px] items-center gap-2.5 rounded-full pl-3 pr-2.5 text-ui-13p5 font-medium text-muted-foreground transition-colors hover:bg-accent hover:text-accent-foreground"
                        >
                          {tab.iconComponent ? (
                            <tab.iconComponent className="size-icon shrink-0" />
                          ) : tab.icon ? (
                            <HugeiconsIcon
                              icon={tab.icon}
                              strokeWidth={1.75}
                              className="size-icon shrink-0"
                            />
                          ) : null}
                          <span className="min-w-0 truncate">{tabLabel}</span>
                        </button>
                        {entries.map((entry) => (
                          <button
                            key={entry}
                            type="button"
                            onClick={() => openResult(tab.id, entry)}
                            className="flex h-[30px] items-center rounded-full pl-10 pr-2.5 text-left text-ui-14 text-foreground transition-colors hover:bg-accent hover:text-accent-foreground"
                          >
                            <span className="min-w-0 truncate">{entry}</span>
                          </button>
                        ))}
                      </div>
                    ))
                  )}
                </div>
              ) : null}
              <p
                className={cn(
                  "pl-4 pt-3 pb-2.5 text-ui-13 font-medium text-muted-foreground max-sm:hidden",
                  results !== null && "hidden",
                )}
              >
                {t("settings.dialog.title")}
              </p>
              <nav
                className={cn(
                  "flex flex-col gap-0.5 px-1 max-sm:flex-row max-sm:overflow-x-auto",
                  results !== null && "max-sm:flex hidden",
                )}
              >
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
                        "relative flex h-[32px] items-center gap-2.5 rounded-full pl-3 pr-2.5 text-ui-14p5 leading-ui-19 tracking-nav font-medium transition-colors",
                        "max-sm:shrink-0",
                        "focus-visible:outline-none",
                        // The active pill already marks the current tab, so
                        // only unselected items get a keyboard focus ring.
                        active
                          ? "text-accent-foreground"
                          : "text-[#383835] dark:text-[#c7c7c4] hover:bg-accent hover:text-accent-foreground focus-visible:ring-1 focus-visible:ring-ring",
                      )}
                    >
                      {active && (
                        <motion.span
                          layoutId="settings-active-pill"
                          className="absolute inset-0 rounded-full bg-accent"
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
                        <span className="relative z-10 ml-auto rounded-full bg-control-accent/10 px-2 py-1 text-ui-10 leading-none font-semibold text-control-accent">
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
                className="absolute top-3 right-3 z-10 flex size-7 items-center justify-center rounded-full text-[#383835] dark:text-[#c7c7c4] transition-colors hover:bg-accent hover:text-accent-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
                aria-label={t("settings.dialog.closeAriaLabel")}
              >
                <HugeiconsIcon icon={Cancel01Icon} className="size-4" />
              </button>
              <div
                ref={mainScrollRef}
                className="hover-scrollbar flex min-h-0 min-w-0 flex-1 flex-col overflow-y-auto p-6 [scrollbar-gutter:stable]"
              >
                {renderTab(panelTab)}
              </div>
            </main>
          </div>
        </DialogContent>
      </Dialog>
      <FloatingMonitor />
    </>
  );
}
