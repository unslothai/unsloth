// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { ATTACHMENT_PAGE_SCALES } from "@/components/assistant-ui/attachment-viewer-meta";
import { ScaleMenu } from "@/components/media-viewer";
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
import {
  DropdownMenu,
  DropdownMenuCheckboxItem,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuShortcut,
  DropdownMenuSub,
  DropdownMenuSubContent,
  DropdownMenuSubTrigger,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Spinner } from "@/components/ui/spinner";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { ATTACHMENT_KIND_ICONS, ATTACHMENT_KIND_ICON_CLASS, attachmentFileKind } from "@/features/chat";
import { startLibraryChat } from "@/features/library";
import { useSettingsDialogStore } from "@/features/settings";
import { useLocale, useT } from "@/i18n";
import { isTauri } from "@/lib/api-base";
import { ChevronDownStandardIcon } from "@/lib/chevron-icons";
import { copyToClipboard } from "@/lib/copy-to-clipboard";
import { openExternalLink } from "@/lib/open-link";
import { RefreshGlyph } from "@/lib/refresh-icon";
import { toast } from "@/lib/toast";
import { cn } from "@/lib/utils";
import {
  Add01Icon,
  ArrowDown01Icon,
  ArrowLeft02Icon,
  ArrowRight02Icon,
  ArrowUp01Icon,
  ArrowUpRight01Icon,
  Cancel01Icon,
  Clock01Icon,
  CursorMagicSelection02Icon,
  Download01Icon,
  InternetIcon,
  LinkSquare02Icon,
  MinusSignIcon,
  MoreHorizontalIcon,
  PlusSignIcon,
  SidebarRightIcon,
  SmartPhone01Icon,
  Tablet01Icon,
} from "@hugeicons/core-free-icons";
import { type IconSvgElement, HugeiconsIcon } from "@hugeicons/react";
import { useNavigate } from "@tanstack/react-router";
import { type ReactNode, memo, useEffect, useRef, useState } from "react";
import { fileNameFromUrl, hostOf, resolveAddress } from "./address";
import { type BrowserDownload, saveBrowserDownload } from "./downloads";
import { useBrowserHistoryStore } from "./history-store";
import { sendFrameCommand } from "./page-frame";
import { useBrowserPrefsStore } from "./prefs-store";
import {
  type BrowserEntry,
  type BrowserTab,
  type DeviceMode,
  browserFile,
  clearPageCache,
  currentEntry,
  pageDownload,
  useBrowserStore,
} from "./store";
import { TabView } from "./tab-view";

function tabAddress(tab: BrowserTab | undefined): string {
  if (!tab) return "";
  const entry = currentEntry(tab);
  if (entry.kind === "web") return tab.displayUrl ?? entry.url;
  if (entry.kind === "file") return entry.name;
  return "";
}

/** The address as shown when not editing: no scheme or bare trailing slash, and only the site and path
 *  unless the full URL is asked for. */
export function displayAddress(address: string, full: boolean): string {
  if (!/^https?:\/\//i.test(address)) return address;
  let shown = address;
  if (!full) {
    try {
      const url = new URL(address);
      shown = url.host + url.pathname;
    } catch {
      // Show it as typed.
    }
  }
  shown = shown.replace(/^https?:\/\//i, "").replace(/\/$/, "");
  try {
    return decodeURI(shown);
  } catch {
    return shown;
  }
}

/** "invoice_INV-6-1.pdf" -> "Invoice Inv 6 1", as the file button shows it. */
export function fileTitle(name: string): string {
  const base = name.replace(/\.[^./]{1,8}$/, "").replace(/[_-]+/g, " ").replace(/\s+/g, " ").trim();
  return (base || name).replace(/\S+/g, (word) => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase());
}

// Browser zoom steps, as Chrome has them.
const ZOOM_STEPS = [0.25, 0.33, 0.5, 0.67, 0.75, 0.8, 0.9, 1, 1.1, 1.25, 1.5, 1.75, 2, 2.5, 3, 4, 5];

function stepZoom(zoom: number, direction: 1 | -1): number {
  if (direction > 0) return ZOOM_STEPS.find((step) => step > zoom + 0.001) ?? zoom;
  return [...ZOOM_STEPS].reverse().find((step) => step < zoom - 0.001) ?? zoom;
}

// Bordered pill in light mode, filled in dark, like the toolbar controls it groups.
const PILL = "border border-border/80 bg-card dark:border-transparent dark:bg-accent";

type ButtonProps = {
  label: string;
  icon?: IconSvgElement;
  onClick?: () => void;
  disabled?: boolean;
  className?: string;
  children?: ReactNode;
};

function IconButton({ label, icon, onClick, disabled, className, children }: ButtonProps) {
  return (
    <Tooltip>
      <TooltipTrigger asChild={true}>
        <button
          type="button"
          aria-label={label}
          onClick={onClick}
          disabled={disabled}
          className={cn(
            "flex size-7 shrink-0 cursor-pointer items-center justify-center rounded-full text-muted-foreground transition-colors hover:bg-[color-mix(in_oklab,var(--foreground)_calc(6%*var(--contrast-wash-gain,1)),transparent)] hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring disabled:cursor-default disabled:opacity-35 disabled:hover:bg-transparent disabled:hover:text-muted-foreground",
            className,
          )}
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

/** A round toolbar button on its own pill. */
function CircleButton(props: ButtonProps) {
  return <IconButton {...props} className={cn(PILL, "size-8 hover:bg-card dark:hover:bg-accent", props.className)} />;
}

const DOCUMENT_KINDS = new Set(["pdf", "word", "spreadsheet", "presentation"]);

function KindIcon({ name, contentType, className }: { name: string; contentType?: string; className?: string }) {
  const kind = attachmentFileKind(name, contentType);
  return (
    <HugeiconsIcon
      icon={ATTACHMENT_KIND_ICONS[kind]}
      strokeWidth={1.75}
      className={cn("size-4 shrink-0", ATTACHMENT_KIND_ICON_CLASS[kind], className)}
    />
  );
}

function TabIcon({ tab }: { tab: BrowserTab }) {
  const [failedFor, setFailedFor] = useState<string | null>(null);
  const entry = currentEntry(tab);
  if (tab.loading) return <Spinner className="size-4 shrink-0" />;
  if (entry.kind === "file") return <KindIcon name={entry.name} contentType={entry.contentType} />;
  if (entry.kind === "internal") {
    const icon = entry.page === "history" ? Clock01Icon : Download01Icon;
    return <HugeiconsIcon icon={icon} strokeWidth={1.75} className="size-4 shrink-0" />;
  }
  if (tab.favicon && failedFor !== tab.favicon) {
    return (
      <img
        src={tab.favicon}
        alt=""
        referrerPolicy="no-referrer"
        onError={() => setFailedFor(tab.favicon)}
        className="size-4 shrink-0 rounded-[3px] object-contain"
      />
    );
  }
  // A document opened from the web shows its type.
  if (entry.kind === "web") {
    const name = fileNameFromUrl(tab.displayUrl ?? entry.url);
    const type = tab.documentType ?? undefined;
    if (DOCUMENT_KINDS.has(attachmentFileKind(name, type))) return <KindIcon name={name} contentType={type} />;
  }
  return <HugeiconsIcon icon={InternetIcon} strokeWidth={1.75} className="size-4 shrink-0" />;
}

function useTabTitle() {
  const t = useT();
  return (tab: BrowserTab, entry: BrowserEntry) => {
    if (tab.title) return tab.title;
    if (entry.kind === "newtab") return t("browser.newTab");
    if (entry.kind === "internal") return t(entry.page === "history" ? "browser.pages.history" : "browser.pages.downloads");
    // Unloaded background tabs show their site.
    return entry.kind === "web" ? hostOf(entry.url) : entry.name;
  };
}

function TabStrip({ tabs, activeTabId }: { tabs: BrowserTab[]; activeTabId: string | null }) {
  const t = useT();
  const tabTitle = useTabTitle();
  const { activateTab, closeTab, newTab, closePanel } = useBrowserStore.getState();
  return (
    <div className="flex h-[var(--studio-chat-header-height,48px)] min-w-0 shrink-0 items-center gap-1 pl-1.5 pr-[calc(0.5rem*var(--ui-space-scale,1)+var(--studio-window-control-inset,0px))]">
      <div
        role="tablist"
        aria-label={t("browser.tabs")}
        className="flex min-w-0 flex-1 items-center overflow-x-auto [scrollbar-width:none]"
      >
        {tabs.map((tab, index) => {
          const active = tab.id === activeTabId;
          const title = tabTitle(tab, currentEntry(tab));
          // Dividers sit between inactive tabs only.
          const divided = index > 0 && !active && tabs[index - 1]?.id !== activeTabId;
          return (
            <div key={tab.id} className="flex min-w-24 max-w-56 flex-1 items-center">
              <span aria-hidden={true} className={cn("h-4 w-px shrink-0 bg-border", !divided && "invisible")} />
              <div
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
                  "group/tab mx-0.5 flex h-[calc(30px*var(--ui-space-scale,1))] min-w-0 flex-1 cursor-pointer items-center gap-2 rounded-[10px] pl-2.5 pr-1 text-ui-13 transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
                  active
                    ? "bg-card text-foreground shadow-[0_1px_2px_rgb(0_0_0/0.06)] dark:bg-accent dark:shadow-none"
                    : "text-muted-foreground hover:bg-[color-mix(in_oklab,var(--foreground)_calc(5%*var(--contrast-wash-gain,1)),transparent)] hover:text-foreground",
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
                    "flex size-5 shrink-0 cursor-pointer items-center justify-center rounded-md text-muted-foreground hover:bg-[color-mix(in_oklab,var(--foreground)_calc(8%*var(--contrast-wash-gain,1)),transparent)] hover:text-foreground",
                    !active && "opacity-0 group-hover/tab:opacity-100 focus-visible:opacity-100",
                  )}
                >
                  <HugeiconsIcon icon={Cancel01Icon} strokeWidth={2} className="size-3.5" />
                </button>
              </div>
            </div>
          );
        })}
      </div>
      <IconButton label={t("browser.newTab")} icon={Add01Icon} onClick={newTab} className="size-8" />
      <span aria-hidden={true} className="mx-1 h-4 w-px shrink-0 bg-border" />
      <IconButton
        label={t("browser.close")}
        icon={SidebarRightIcon}
        onClick={closePanel}
        className="size-8 rounded-[10px] bg-[color-mix(in_oklab,var(--foreground)_calc(6%*var(--contrast-wash-gain,1)),transparent)] text-foreground"
      />
    </div>
  );
}

function AddressBar({ tab }: { tab: BrowserTab | undefined }) {
  const t = useT();
  const engine = useBrowserPrefsStore((state) => state.searchEngine);
  const showFullUrl = useBrowserPrefsStore((state) => state.showFullUrl);
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
      <div className="relative">
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
          className={cn(
            PILL,
            "h-8 w-full min-w-0 rounded-full px-4 text-ui-13 text-foreground outline-none transition-colors placeholder:text-center placeholder:text-muted-foreground focus:ring-2 focus:ring-ring/40 focus:placeholder:text-left",
            // The input keeps the full URL, so focusing never changes its text or selection.
            !editing && address && "text-transparent",
          )}
        />
        {!editing && address ? (
          <span
            aria-hidden={true}
            className="pointer-events-none absolute inset-0 flex items-center justify-center px-4 text-ui-13 text-foreground"
          >
            <span className="truncate">{displayAddress(address, showFullUrl)}</span>
          </span>
        ) : null}
      </div>
    </form>
  );
}

function tabDownload(tab: BrowserTab | undefined): BrowserDownload | undefined {
  if (!tab) return undefined;
  const entry = currentEntry(tab);
  if (entry.kind === "file") {
    const blob = browserFile(entry.fileId);
    return blob ? { blob, name: entry.name, contentType: entry.contentType, url: null } : undefined;
  }
  const page = pageDownload(tab.id);
  return page ? { ...page, url: tab.displayUrl ?? (entry.kind === "web" ? entry.url : null) } : undefined;
}

function webAddress(tab: BrowserTab | undefined): string | null {
  const entry = tab ? currentEntry(tab) : null;
  return entry?.kind === "web" ? (tab?.displayUrl ?? entry.url) : null;
}

function WebActions({ tab }: { tab: BrowserTab | undefined }) {
  const t = useT();
  const webUrl = webAddress(tab);
  const download = tabDownload(tab);
  return (
    <div className={cn(PILL, "flex h-8 shrink-0 items-center gap-0.5 rounded-full px-0.5")}>
      <IconButton
        label={t("browser.openExternal")}
        icon={LinkSquare02Icon}
        disabled={!webUrl}
        onClick={() => webUrl && openExternalLink(webUrl)}
      />
      <IconButton
        label={t("browser.download")}
        icon={Download01Icon}
        disabled={!download}
        onClick={() => download && void saveBrowserDownload(download)}
      />
    </div>
  );
}

/** Whether the tab shows a web page (not a document) that page commands reach. */
function showsWebPage(tab: BrowserTab | undefined): boolean {
  return Boolean(tab && currentEntry(tab).kind === "web" && !tab.documentType && !tab.loading);
}

function ZoomControl({ tab }: { tab: BrowserTab | undefined }) {
  const t = useT();
  const locale = useLocale();
  const zoom = tab?.zoom ?? 1;
  const setZoom = (next: number) => tab && useBrowserStore.getState().setZoom(tab.id, next);
  const percent = new Intl.NumberFormat(locale, { style: "percent", maximumFractionDigits: 0 }).format(zoom);
  const step =
    "flex size-7 cursor-pointer items-center justify-center text-muted-foreground hover:text-foreground disabled:cursor-default disabled:opacity-35";
  return (
    <div className="flex items-center gap-2 px-2 py-1.5 text-sm">
      <span className="flex-1">{t("browser.menu.zoom")}</span>
      <div className="flex h-7 items-center rounded-lg border border-border">
        <button
          type="button"
          aria-label={t("browser.menu.zoomOut")}
          disabled={!tab || zoom <= (ZOOM_STEPS[0] ?? 0)}
          onClick={() => setZoom(stepZoom(zoom, -1))}
          className={step}
        >
          <HugeiconsIcon icon={MinusSignIcon} strokeWidth={1.75} className="size-3.5" />
        </button>
        <span className="min-w-12 border-x border-border text-center tabular-nums">{percent}</span>
        <button
          type="button"
          aria-label={t("browser.menu.zoomIn")}
          disabled={!tab || zoom >= (ZOOM_STEPS[ZOOM_STEPS.length - 1] ?? 5)}
          onClick={() => setZoom(stepZoom(zoom, 1))}
          className={step}
        >
          <HugeiconsIcon icon={PlusSignIcon} strokeWidth={1.75} className="size-3.5" />
        </button>
      </div>
      <button
        type="button"
        aria-label={t("browser.menu.zoomReset")}
        disabled={!tab || zoom === 1}
        onClick={() => setZoom(1)}
        className={cn(step, "rounded-md")}
      >
        <RefreshGlyph strokeWidth={1.75} className="size-3.5" />
      </button>
    </div>
  );
}

export function ClearBrowsingDataDialog({
  open,
  onOpenChange,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}) {
  const t = useT();
  return (
    <AlertDialog open={open} onOpenChange={onOpenChange}>
      <AlertDialogContent>
        <AlertDialogHeader>
          <AlertDialogTitle>{t("browser.clearData.title")}</AlertDialogTitle>
          <AlertDialogDescription>{t("browser.clearData.description")}</AlertDialogDescription>
        </AlertDialogHeader>
        <AlertDialogFooter>
          <AlertDialogCancel>{t("browser.clearData.cancel")}</AlertDialogCancel>
          <AlertDialogAction
            onClick={() => {
              const history = useBrowserHistoryStore.getState();
              history.clearHistory();
              history.clearDownloads();
              clearPageCache();
              toast.success(t("browser.clearData.done"));
            }}
          >
            {t("browser.clearData.confirm")}
          </AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  );
}

function PanelMenu({ tab }: { tab: BrowserTab | undefined }) {
  const t = useT();
  const device = useBrowserStore((state) => state.device);
  const [clearOpen, setClearOpen] = useState(false);
  const webUrl = webAddress(tab);
  const webPage = showsWebPage(tab);
  const store = useBrowserStore.getState();
  const mod = typeof navigator !== "undefined" && /Mac|iPhone|iPad/.test(navigator.platform) ? "⌘" : "Ctrl+";
  return (
    <>
      <DropdownMenu>
        <Tooltip>
          <TooltipTrigger asChild={true}>
            <DropdownMenuTrigger asChild={true}>
              <button
                type="button"
                aria-label={t("browser.more")}
                className={cn(
                  PILL,
                  "flex size-8 shrink-0 cursor-pointer items-center justify-center rounded-full text-muted-foreground transition-colors hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
                )}
              >
                <HugeiconsIcon icon={MoreHorizontalIcon} strokeWidth={1.75} className="size-4.5" />
              </button>
            </DropdownMenuTrigger>
          </TooltipTrigger>
          <TooltipContent side="bottom" className="tooltip-compact">
            {t("browser.more")}
          </TooltipContent>
        </Tooltip>
        <DropdownMenuContent align="end" className="min-w-64">
          <DropdownMenuItem disabled={!webPage} onSelect={() => store.setFindOpen(true)}>
            {t("browser.menu.find")}
            <DropdownMenuShortcut>{mod}F</DropdownMenuShortcut>
          </DropdownMenuItem>
          <DropdownMenuItem
            disabled={!webUrl}
            onSelect={() =>
              webUrl && void copyToClipboard(webUrl).then((ok) => ok && toast.success(t("browser.linkCopied")))
            }
          >
            {t("browser.copyLink")}
          </DropdownMenuItem>
          <DropdownMenuSeparator />
          <ZoomControl tab={tab} />
          <DropdownMenuSeparator />
          <DropdownMenuCheckboxItem
            disabled={!webPage && device === "off"}
            checked={device !== "off"}
            onCheckedChange={(checked) => store.setDevice(checked ? "mobile" : "off")}
          >
            {t("browser.menu.deviceToolbar")}
          </DropdownMenuCheckboxItem>
          <DropdownMenuSeparator />
          <DropdownMenuItem onSelect={() => store.openInternal("downloads")}>
            {t("browser.pages.downloads")}
          </DropdownMenuItem>
          <DropdownMenuItem onSelect={() => store.openInternal("history")}>{t("browser.pages.history")}</DropdownMenuItem>
          <DropdownMenuItem onSelect={() => setClearOpen(true)}>{t("browser.menu.clearData")}</DropdownMenuItem>
          <DropdownMenuSeparator />
          <DropdownMenuItem onSelect={() => useSettingsDialogStore.getState().openDialog("chat")}>
            {t("browser.settings")}
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>
      <ClearBrowsingDataDialog open={clearOpen} onOpenChange={setClearOpen} />
    </>
  );
}

function WebToolbar({ tab }: { tab: BrowserTab | undefined }) {
  const t = useT();
  const { goBack, goForward, reload } = useBrowserStore.getState();
  return (
    <>
      <div className={cn(PILL, "flex h-8 shrink-0 items-center gap-0.5 rounded-full px-0.5")}>
        <IconButton
          label={t("browser.back")}
          icon={ArrowLeft02Icon}
          disabled={!tab || tab.index === 0}
          onClick={() => tab && goBack(tab.id)}
        />
        <IconButton
          label={t("browser.forward")}
          icon={ArrowRight02Icon}
          disabled={!tab || tab.index >= tab.history.length - 1}
          onClick={() => tab && goForward(tab.id)}
        />
        <span aria-hidden={true} className="mx-0.5 h-4 w-px shrink-0 bg-[color-mix(in_oklab,var(--foreground)_calc(15%*var(--contrast-wash-gain,1)),transparent)]" />
        <IconButton
          label={t("browser.reload")}
          disabled={!tab || currentEntry(tab).kind !== "web"}
          onClick={() => tab && reload(tab.id)}
        >
          <RefreshGlyph strokeWidth={1.75} className="size-4" />
        </IconButton>
      </div>
      <AddressBar key={tab?.id ?? "none"} tab={tab} />
      <WebActions tab={tab} />
      <PanelMenu tab={tab} />
    </>
  );
}

/** Toolbar for an opened file: its menu, Request edits, zoom and download. */
function FileToolbar({ tab, entry }: { tab: BrowserTab; entry: Extract<BrowserEntry, { kind: "file" }> }) {
  const t = useT();
  const navigate = useNavigate();
  const requestEdits = useBrowserStore((state) => state.requestEdits);
  const download = tabDownload(tab);
  const blob = download?.blob;
  const openInNewChat = () => {
    if (!blob) return;
    startLibraryChat(navigate, { files: [new File([blob], entry.name, { type: entry.contentType || blob.type })] });
  };
  const openInBrowser = () => {
    if (!blob) return;
    const url = URL.createObjectURL(blob);
    window.open(url, "_blank", "noopener,noreferrer");
    window.setTimeout(() => URL.revokeObjectURL(url), 60_000);
  };
  return (
    <>
      <DropdownMenu>
        <DropdownMenuTrigger asChild={true}>
          <button
            type="button"
            className="flex h-9 min-w-0 max-w-[55%] shrink cursor-pointer items-center gap-2 rounded-full bg-[color-mix(in_oklab,var(--foreground)_calc(5%*var(--contrast-wash-gain,1)),transparent)] pl-3 pr-2.5 text-ui-13p5 text-foreground outline-none transition-colors hover:bg-[color-mix(in_oklab,var(--foreground)_calc(8%*var(--contrast-wash-gain,1)),transparent)] focus-visible:ring-2 focus-visible:ring-ring data-[state=open]:bg-[color-mix(in_oklab,var(--foreground)_calc(8%*var(--contrast-wash-gain,1)),transparent)]"
          >
            <KindIcon name={entry.name} contentType={entry.contentType} className="size-4.5" />
            <span className="min-w-0 truncate">{fileTitle(entry.name)}</span>
            <HugeiconsIcon
              icon={ChevronDownStandardIcon}
              strokeWidth={1.75}
              className="size-4 shrink-0 text-muted-foreground"
            />
          </button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="start" className="w-72 max-w-[calc(100vw-2rem)]">
          <div className="flex items-start gap-3 px-2 py-2 text-sm">
            <KindIcon name={entry.name} contentType={entry.contentType} className="mt-0.5 size-4.5" />
            <span className="min-w-0 break-words">{entry.name}</span>
          </div>
          <DropdownMenuSeparator />
          <DropdownMenuSub>
            <DropdownMenuSubTrigger disabled={!blob}>
              <HugeiconsIcon icon={ArrowUpRight01Icon} strokeWidth={1.75} className="size-4" />
              {t("browser.file.openIn")}
            </DropdownMenuSubTrigger>
            <DropdownMenuSubContent>
              <DropdownMenuItem onSelect={openInNewChat}>{t("browser.file.newChat")}</DropdownMenuItem>
              {/* A blob URL can't be handed to another app from the desktop app. */}
              {isTauri ? null : (
                <DropdownMenuItem onSelect={openInBrowser}>{t("browser.file.newBrowserTab")}</DropdownMenuItem>
              )}
            </DropdownMenuSubContent>
          </DropdownMenuSub>
        </DropdownMenuContent>
      </DropdownMenu>
      {requestEdits ? (
        <button
          type="button"
          onClick={() => requestEdits(t("browser.file.requestEditsPrompt", { name: entry.name }))}
          className={cn(
            PILL,
            "flex h-9 shrink-0 cursor-pointer items-center gap-2 rounded-full px-3.5 text-ui-13p5 text-foreground outline-none transition-colors focus-visible:ring-2 focus-visible:ring-ring",
          )}
        >
          <HugeiconsIcon icon={CursorMagicSelection02Icon} strokeWidth={1.75} className="size-4.5" />
          {t("browser.file.requestEdits")}
        </button>
      ) : null}
      <span className="min-w-0 flex-1" />
      <ScaleMenu
        value={tab.zoom}
        scales={ATTACHMENT_PAGE_SCALES}
        onChange={(value) => useBrowserStore.getState().setZoom(tab.id, value === "fit" ? 1 : value)}
        className={cn(PILL, "mr-0 h-9 hover:bg-card dark:hover:bg-accent")}
      />
      <CircleButton
        label={t("browser.download")}
        icon={Download01Icon}
        disabled={!download}
        onClick={() => download && void saveBrowserDownload(download)}
        className="size-9"
      />
    </>
  );
}

function FindBar({ tab }: { tab: BrowserTab | undefined }) {
  const t = useT();
  const miss = useBrowserStore((state) => state.findMiss);
  const [query, setQuery] = useState("");
  const inputRef = useRef<HTMLInputElement | null>(null);
  const { setFindOpen, setFindMiss } = useBrowserStore.getState();
  useEffect(() => {
    inputRef.current?.focus();
  }, []);
  const find = (backwards = false) => {
    if (!tab || !query) return;
    if (!sendFrameCommand(tab.id, { command: "find", query, backwards })) setFindMiss(true);
  };
  return (
    <div className="flex shrink-0 items-center gap-1.5 px-2.5 pb-2">
      <input
        ref={inputRef}
        value={query}
        onChange={(event) => {
          setQuery(event.target.value);
          setFindMiss(false);
        }}
        onKeyDown={(event) => {
          if (event.key === "Enter") find(event.shiftKey);
          else if (event.key === "Escape") setFindOpen(false);
        }}
        placeholder={t("browser.find.placeholder")}
        aria-label={t("browser.menu.find")}
        spellCheck={false}
        className={cn(
          PILL,
          "h-8 min-w-0 flex-1 rounded-full px-4 text-ui-13 outline-none placeholder:text-muted-foreground focus:ring-2 focus:ring-ring/40",
          miss && query && "ring-2 ring-destructive/40 focus:ring-destructive/40",
        )}
      />
      {miss && query ? (
        <span className="shrink-0 text-ui-12 text-muted-foreground">{t("browser.find.noMatches")}</span>
      ) : null}
      <div className={cn(PILL, "flex h-8 shrink-0 items-center gap-0.5 rounded-full px-0.5")}>
        <IconButton
          label={t("browser.find.previous")}
          icon={ArrowUp01Icon}
          disabled={!query}
          onClick={() => find(true)}
        />
        <IconButton label={t("browser.find.next")} icon={ArrowDown01Icon} disabled={!query} onClick={() => find()} />
      </div>
      <CircleButton label={t("browser.find.close")} icon={Cancel01Icon} onClick={() => setFindOpen(false)} />
    </div>
  );
}

const DEVICE_WIDTHS: Record<Exclude<DeviceMode, "off">, number> = { mobile: 390, tablet: 820 };

function DeviceBar() {
  const t = useT();
  const device = useBrowserStore((state) => state.device);
  const { setDevice } = useBrowserStore.getState();
  const option = (mode: Exclude<DeviceMode, "off">, icon: IconSvgElement, label: string) => (
    <button
      type="button"
      aria-pressed={device === mode}
      onClick={() => setDevice(mode)}
      className={cn(
        "flex h-7 cursor-pointer items-center gap-1.5 rounded-full px-3 text-ui-12 transition-colors",
        device === mode ? "bg-[color-mix(in_oklab,var(--foreground)_calc(8%*var(--contrast-wash-gain,1)),transparent)] text-foreground" : "text-muted-foreground hover:text-foreground",
      )}
    >
      <HugeiconsIcon icon={icon} strokeWidth={1.75} className="size-3.5" />
      {label}
      <span className="tabular-nums text-muted-foreground">{DEVICE_WIDTHS[mode]}</span>
    </button>
  );
  return (
    <div className="flex shrink-0 items-center justify-center gap-1 border-t border-border/70 px-2.5 py-1.5">
      {option("mobile", SmartPhone01Icon, t("browser.device.mobile"))}
      {option("tablet", Tablet01Icon, t("browser.device.tablet"))}
      <IconButton
        label={t("browser.device.close")}
        icon={Cancel01Icon}
        onClick={() => setDevice("off")}
        className="ml-1"
      />
    </div>
  );
}

// Hidden pages still run scripts, so only the most recent tabs stay mounted.
const MAX_MOUNTED_TABS = 4;

/** The chat's in-app browser: tabs of web pages and opened files. Memoized so chat renders skip it. */
export const BrowserPanel = memo(function BrowserPanel() {
  const t = useT();
  const tabs = useBrowserStore((state) => state.tabs);
  const activeTabId = useBrowserStore((state) => state.activeTabId);
  const findOpen = useBrowserStore((state) => state.findOpen);
  const device = useBrowserStore((state) => state.device);
  const activeTab = tabs.find((tab) => tab.id === activeTabId);
  const activeEntry = activeTab ? currentEntry(activeTab) : null;
  // Mount tabs on first view. Most recent last.
  const [mounted, setMounted] = useState<readonly string[]>(() => (activeTabId ? [activeTabId] : []));
  if (activeTabId && mounted[mounted.length - 1] !== activeTabId) {
    const open = new Set(tabs.map((tab) => tab.id));
    setMounted(
      [...mounted.filter((id) => id !== activeTabId && open.has(id)), activeTabId].slice(-MAX_MOUNTED_TABS),
    );
  }
  // Files sit on the viewer's gray, with no line under the toolbar.
  const fileTab = activeEntry?.kind === "file";
  const documentShown = fileTab || Boolean(activeTab?.documentType);
  const deviceWidth = device !== "off" && activeEntry?.kind === "web" ? DEVICE_WIDTHS[device] : null;

  return (
    // Full-height pane; the top inset clears a desktop titlebar.
    <section
      aria-label={t("browser.title")}
      className="relative flex h-full min-h-0 flex-col overflow-hidden bg-muted pt-[var(--studio-content-top-inset,0px)]"
    >
      <TabStrip tabs={tabs} activeTabId={activeTabId} />
      {/* Toolbar and page sit on a card, raised off the tab strip. */}
      <div
        className={cn(
          "flex min-h-0 flex-1 flex-col overflow-hidden rounded-tl-[14px] border-t border-border/60 dark:border-transparent",
          documentShown ? "bg-[color-mix(in_oklab,var(--muted)_55%,var(--card))]" : "bg-card",
        )}
      >
        <div className="flex shrink-0 items-center gap-1.5 px-2.5 py-2">
          {activeTab && activeEntry?.kind === "file" ? (
            <FileToolbar tab={activeTab} entry={activeEntry} />
          ) : (
            <WebToolbar tab={activeTab} />
          )}
        </div>
        {findOpen && showsWebPage(activeTab) ? <FindBar key={activeTabId} tab={activeTab} /> : null}
        {deviceWidth ? <DeviceBar /> : null}
        <div
          className={cn(
            "relative min-h-0 flex-1 overflow-hidden",
            !fileTab && "border-t border-border/70",
            documentShown ? "bg-transparent" : "bg-background",
            deviceWidth && "bg-muted/60",
          )}
        >
          {activeTab?.loading ? (
            <div className="absolute inset-x-0 top-0 z-10 h-[2.5px] overflow-hidden">
              <span aria-hidden={true} className="artifact-loading-line block h-full rounded-full motion-reduce:hidden" />
            </div>
          ) : null}
          <div
            className={cn("relative mx-auto h-full", deviceWidth && "border-x border-border/70 bg-background shadow-sm")}
            style={deviceWidth ? { width: `min(100%, ${deviceWidth}px)` } : undefined}
          >
            {tabs.map((tab) =>
              mounted.includes(tab.id) ? <TabView key={tab.id} tab={tab} active={tab.id === activeTabId} /> : null,
            )}
          </div>
        </div>
      </div>
    </section>
  );
});
