// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Spinner } from "@/components/ui/spinner";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { useSettingsDialogStore } from "@/features/settings";
import { useT } from "@/i18n";
import { copyToClipboard } from "@/lib/copy-to-clipboard";
import { downloadFile } from "@/lib/native-files";
import { openExternalLink } from "@/lib/open-link";
import { toast } from "@/lib/toast";
import { cn } from "@/lib/utils";
import {
  Add01Icon,
  ArrowLeft02Icon,
  ArrowRight02Icon,
  Cancel01Icon,
  Copy01Icon,
  Download01Icon,
  File01Icon,
  Globe02Icon,
  LinkSquare02Icon,
  MoreHorizontalIcon,
  Settings02Icon,
} from "@hugeicons/core-free-icons";
import { type IconSvgElement, HugeiconsIcon } from "@hugeicons/react";
import { RefreshGlyph } from "@/lib/refresh-icon";
import { type ReactNode, useEffect, useRef, useState } from "react";
import { hostOf, resolveAddress } from "./address";
import { useBrowserPrefsStore } from "./prefs-store";
import { type BrowserTab, browserFile, currentEntry, pageDownload, useBrowserStore } from "./store";
import { TabView } from "./tab-view";

function tabAddress(tab: BrowserTab | undefined): string {
  if (!tab) return "";
  const entry = currentEntry(tab);
  if (entry.kind === "web") return tab.displayUrl ?? entry.url;
  if (entry.kind === "file") return entry.name;
  return "";
}

function IconButton({
  label,
  icon,
  onClick,
  disabled,
  children,
}: {
  label: string;
  icon?: IconSvgElement;
  onClick?: () => void;
  disabled?: boolean;
  children?: ReactNode;
}) {
  return (
    <Tooltip>
      <TooltipTrigger asChild={true}>
        <button
          type="button"
          aria-label={label}
          onClick={onClick}
          disabled={disabled}
          className="flex size-8 shrink-0 cursor-pointer items-center justify-center rounded-full text-muted-foreground transition-colors hover:bg-muted hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring disabled:cursor-default disabled:opacity-40 disabled:hover:bg-transparent"
        >
          {icon ? <HugeiconsIcon icon={icon} strokeWidth={1.75} className="size-4.5" /> : children}
        </button>
      </TooltipTrigger>
      <TooltipContent side="bottom" className="tooltip-compact">
        {label}
      </TooltipContent>
    </Tooltip>
  );
}

function TabIcon({ tab }: { tab: BrowserTab }) {
  const [failedFor, setFailedFor] = useState<string | null>(null);
  const entry = currentEntry(tab);
  if (tab.loading) return <Spinner className="size-3.5 shrink-0" />;
  if (entry.kind === "file") {
    return <HugeiconsIcon icon={File01Icon} strokeWidth={1.75} className="size-3.5 shrink-0" />;
  }
  if (tab.favicon && failedFor !== tab.favicon) {
    return (
      <img
        src={tab.favicon}
        alt=""
        referrerPolicy="no-referrer"
        onError={() => setFailedFor(tab.favicon)}
        className="size-3.5 shrink-0 rounded-sm object-contain"
      />
    );
  }
  return <HugeiconsIcon icon={Globe02Icon} strokeWidth={1.75} className="size-3.5 shrink-0" />;
}

function TabStrip({ tabs, activeTabId }: { tabs: BrowserTab[]; activeTabId: string | null }) {
  const t = useT();
  const { activateTab, closeTab, newTab, closePanel } = useBrowserStore.getState();
  return (
    <div className="flex h-[var(--studio-chat-header-height,48px)] min-w-0 shrink-0 items-center gap-1 pl-2 pr-[calc(0.5rem*var(--ui-space-scale,1)+var(--studio-window-control-inset,0px))]">
      <div role="tablist" aria-label={t("browser.tabs")} className="flex min-w-0 flex-1 items-center gap-1 overflow-x-auto [scrollbar-width:none]">
        {tabs.map((tab) => {
          const active = tab.id === activeTabId;
          const entry = currentEntry(tab);
          // Unloaded background tabs show their site.
          const title =
            tab.title ||
            (entry.kind === "newtab" ? t("browser.newTab") : entry.kind === "web" ? hostOf(entry.url) : entry.name);
          return (
            <div
              key={tab.id}
              role="tab"
              aria-selected={active}
              tabIndex={0}
              title={title}
              onClick={() => activateTab(tab.id)}
              onAuxClick={(event) => {
                if (event.button === 1) closeTab(tab.id);
              }}
              onKeyDown={(event) => {
                if (event.key === "Enter" || event.key === " ") activateTab(tab.id);
              }}
              className={cn(
                "group/tab flex h-8 min-w-26 max-w-52 flex-1 cursor-pointer items-center gap-2 rounded-full pl-3 pr-1 text-ui-12p5 transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
                active
                  ? "bg-background text-foreground shadow-xs ring-1 ring-border/60"
                  : "text-muted-foreground hover:bg-muted/70 hover:text-foreground",
              )}
            >
              <TabIcon tab={tab} />
              <span className="min-w-0 flex-1 truncate">{title}</span>
              <button
                type="button"
                aria-label={t("browser.closeTab")}
                onClick={(event) => {
                  event.stopPropagation();
                  closeTab(tab.id);
                }}
                className={cn(
                  "flex size-6 shrink-0 cursor-pointer items-center justify-center rounded-full text-muted-foreground hover:bg-muted hover:text-foreground",
                  !active && "opacity-0 group-hover/tab:opacity-100 focus-visible:opacity-100",
                )}
              >
                <HugeiconsIcon icon={Cancel01Icon} strokeWidth={2} className="size-3.5" />
              </button>
            </div>
          );
        })}
      </div>
      <IconButton label={t("browser.newTab")} icon={Add01Icon} onClick={newTab} />
      <IconButton label={t("browser.close")} icon={Cancel01Icon} onClick={closePanel} />
    </div>
  );
}

function AddressBar({ tab }: { tab: BrowserTab | undefined }) {
  const t = useT();
  const engine = useBrowserPrefsStore((state) => state.searchEngine);
  const focusSequence = useBrowserStore((state) => state.focusAddressSequence);
  const inputRef = useRef<HTMLInputElement | null>(null);
  const address = tabAddress(tab);
  const [value, setValue] = useState(address);
  const [editing, setEditing] = useState(false);
  const shown = editing ? value : address;

  useEffect(() => {
    if (focusSequence === 0) return;
    const input = inputRef.current;
    if (!input) return;
    input.focus();
    input.select();
  }, [focusSequence]);

  return (
    <form
      className="min-w-0 flex-1"
      onSubmit={(event) => {
        event.preventDefault();
        const url = resolveAddress(value, engine);
        if (!url || !tab) return;
        useBrowserStore.getState().navigate(tab.id, { url });
        setEditing(false);
        inputRef.current?.blur();
      }}
    >
      <input
        ref={inputRef}
        value={shown}
        onChange={(event) => {
          setEditing(true);
          setValue(event.target.value);
        }}
        onFocus={(event) => {
          setValue(address);
          setEditing(true);
          event.currentTarget.select();
        }}
        onBlur={() => setEditing(false)}
        onKeyDown={(event) => {
          if (event.key === "Escape") {
            setValue(address);
            event.currentTarget.blur();
          }
        }}
        placeholder={t("browser.addressPlaceholder")}
        aria-label={t("browser.addressPlaceholder")}
        spellCheck={false}
        autoCapitalize="off"
        autoCorrect="off"
        className="h-8 w-full min-w-0 rounded-full bg-muted/60 px-3.5 text-ui-13 text-foreground outline-none transition-colors placeholder:text-muted-foreground hover:bg-muted focus:bg-background focus:ring-2 focus:ring-ring/40"
      />
    </form>
  );
}

function PanelMenu({ tab }: { tab: BrowserTab | undefined }) {
  const t = useT();
  const entry = tab ? currentEntry(tab) : null;
  const webUrl = entry?.kind === "web" ? (tab?.displayUrl ?? entry.url) : null;
  const download =
    entry?.kind === "file"
      ? (() => {
          const blob = browserFile(entry.fileId);
          return blob ? { blob, name: entry.name, contentType: entry.contentType } : undefined;
        })()
      : tab
        ? pageDownload(tab.id)
        : undefined;
  return (
    <DropdownMenu>
      <Tooltip>
        <TooltipTrigger asChild={true}>
          <DropdownMenuTrigger asChild={true}>
            <button
              type="button"
              aria-label={t("browser.more")}
              className="flex size-8 shrink-0 cursor-pointer items-center justify-center rounded-full text-muted-foreground transition-colors hover:bg-muted hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
            >
              <HugeiconsIcon icon={MoreHorizontalIcon} strokeWidth={1.75} className="size-4.5" />
            </button>
          </DropdownMenuTrigger>
        </TooltipTrigger>
        <TooltipContent side="bottom" className="tooltip-compact">
          {t("browser.more")}
        </TooltipContent>
      </Tooltip>
      <DropdownMenuContent align="end" className="min-w-52">
        {webUrl ? (
          <>
            <DropdownMenuItem onSelect={() => openExternalLink(webUrl)}>
              <HugeiconsIcon icon={LinkSquare02Icon} strokeWidth={1.75} className="size-4" />
              {t("browser.openExternal")}
            </DropdownMenuItem>
            <DropdownMenuItem
              onSelect={() =>
                void copyToClipboard(webUrl).then((ok) => ok && toast.success(t("browser.linkCopied")))
              }
            >
              <HugeiconsIcon icon={Copy01Icon} strokeWidth={1.75} className="size-4" />
              {t("browser.copyLink")}
            </DropdownMenuItem>
          </>
        ) : null}
        {download ? (
          <DropdownMenuItem
            onSelect={() => void downloadFile(download.blob, download.name, download.contentType || undefined)}
          >
            <HugeiconsIcon icon={Download01Icon} strokeWidth={1.75} className="size-4" />
            {t("browser.download")}
          </DropdownMenuItem>
        ) : null}
        {webUrl || download ? <DropdownMenuSeparator /> : null}
        <DropdownMenuItem
          onSelect={() => {
            const { tabs, closeTab } = useBrowserStore.getState();
            for (const open of tabs) closeTab(open.id);
          }}
        >
          <HugeiconsIcon icon={Cancel01Icon} strokeWidth={1.75} className="size-4" />
          {t("browser.closeAllTabs")}
        </DropdownMenuItem>
        <DropdownMenuItem onSelect={() => useSettingsDialogStore.getState().openDialog("chat")}>
          <HugeiconsIcon icon={Settings02Icon} strokeWidth={1.75} className="size-4" />
          {t("browser.settings")}
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}

/** The chat's in-app browser: tabs of web pages and opened files. */
export function BrowserPanel() {
  const t = useT();
  const tabs = useBrowserStore((state) => state.tabs);
  const activeTabId = useBrowserStore((state) => state.activeTabId);
  const activeTab = tabs.find((tab) => tab.id === activeTabId);
  const { goBack, goForward, reload } = useBrowserStore.getState();
  // Mount tabs on first view, so reopening the panel loads one page, not all.
  const [shown, setShown] = useState<ReadonlySet<string>>(() => new Set(activeTabId ? [activeTabId] : []));
  if (activeTabId && !shown.has(activeTabId)) setShown(new Set(shown).add(activeTabId));

  return (
    // Full-height pane; the top inset clears a desktop titlebar.
    <section
      aria-label={t("browser.title")}
      className="relative flex h-full min-h-0 flex-col overflow-hidden bg-background pt-[var(--studio-content-top-inset,0px)]"
    >
      <TabStrip tabs={tabs} activeTabId={activeTabId} />
      <div className="flex shrink-0 items-center gap-1 px-2 pb-2">
        <IconButton
          label={t("browser.back")}
          icon={ArrowLeft02Icon}
          disabled={!activeTab || activeTab.index === 0}
          onClick={() => activeTab && goBack(activeTab.id)}
        />
        <IconButton
          label={t("browser.forward")}
          icon={ArrowRight02Icon}
          disabled={!activeTab || activeTab.index >= activeTab.history.length - 1}
          onClick={() => activeTab && goForward(activeTab.id)}
        />
        <IconButton
          label={t("browser.reload")}
          disabled={!activeTab || currentEntry(activeTab).kind !== "web"}
          onClick={() => activeTab && reload(activeTab.id)}
        >
          <RefreshGlyph strokeWidth={1.75} className="size-4.5" />
        </IconButton>
        <AddressBar key={activeTab?.id ?? "none"} tab={activeTab} />
        <PanelMenu tab={activeTab} />
      </div>
      <div className="relative min-h-0 flex-1 overflow-hidden border-t border-border/70 bg-background">
        {activeTab?.loading ? (
          <div className="absolute inset-x-0 top-0 z-10 h-[2.5px] overflow-hidden">
            <span aria-hidden={true} className="artifact-loading-line block h-full rounded-full motion-reduce:hidden" />
          </div>
        ) : null}
        {tabs.map((tab) =>
          shown.has(tab.id) ? <TabView key={tab.id} tab={tab} active={tab.id === activeTabId} /> : null,
        )}
      </div>
    </section>
  );
}
