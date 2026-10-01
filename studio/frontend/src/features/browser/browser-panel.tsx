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
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  ATTACHMENT_KIND_ICONS,
  ATTACHMENT_KIND_ICON_CLASS,
  attachmentFileKind,
} from "@/features/chat";
import { startLibraryChat } from "@/features/library";
import {
  useSettingsDialogStore,
  useShortcut,
  useShortcutLabel,
} from "@/features/settings";
import { useLocale, useT } from "@/i18n";
import { isTauri } from "@/lib/api-base";
import { ChevronDownStandardIcon } from "@/lib/chevron-icons";
import { copyToClipboard, copyToClipboardFrom } from "@/lib/copy-to-clipboard";
import { openExternalLink } from "@/lib/open-link";
import { RefreshGlyph } from "@/lib/refresh-icon";
import { Tick02Icon } from "@/lib/tick-icon";
import { toast } from "@/lib/toast";
import { cn } from "@/lib/utils";
import {
  Add01Icon,
  ArrowDown01Icon,
  ArrowLeft02Icon,
  ArrowRight02Icon,
  ArrowUp01Icon,
  ArrowUpRight01Icon,
  BubbleChatAddIcon,
  Cancel01Icon,
  Clock01Icon,
  ComputerTerminal01Icon,
  Copy01Icon,
  CursorRectangleSelection02Icon,
  Download01Icon,
  InternetIcon,
  LinkSquare02Icon,
  MinusSignIcon,
  MoreHorizontalIcon,
  PaintBoardIcon,
  PlusSignIcon,
  SmartPhone01Icon,
  SourceCodeIcon,
  Tablet01Icon,
  TextWrapIcon,
  ViewIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon, type IconSvgElement } from "@hugeicons/react";
import { useNavigate } from "@tanstack/react-router";
import { type ReactNode, memo, useEffect, useRef, useState } from "react";
import { fileNameFromUrl, hostOf, resolveAddress } from "./address";
import { type BrowserDownload, saveBrowserDownload } from "./downloads";
import { useBrowserHistoryStore } from "./history-store";
import { AnnotateLayer } from "./annotate-layer";
import { browserTabType, textFileKind } from "./file-view";
import { EnterFullViewIcon, ExitFullViewIcon, SplitPaneIcon } from "./icons";
import { sendFrameCommand } from "./page-frame";
import {
  clearNativeBrowsingData,
  hasNativeView,
  nativeAction,
  nativeBrowser,
  nativeFind,
  returnToNativePage,
  startNativeViews,
} from "./native-view";
import { useBrowserPrefsStore } from "./prefs-store";
import {
  type BrowserEntry,
  type BrowserTab,
  DEFAULT_FILE_VIEW,
  type DeviceMode,
  type FileViewState,
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

// Bidi controls can reverse the text after them, making one site's path read like another's.
const BIDI_CONTROLS = /[\u061c\u200e\u200f\u202a-\u202e\u2066-\u2069]/g;

/** The address as shown when not editing: no scheme or bare trailing slash, and only the site and path
 *  unless the full URL is asked for. From the parsed URL: punycode host, no credentials. */
export function displayAddress(address: string, full: boolean): string {
  if (!/^https?:\/\//i.test(address)) return address;
  let shown: string;
  try {
    const url = new URL(address);
    shown = url.host + url.pathname + (full ? url.search + url.hash : "");
  } catch {
    shown = address.replace(/^https?:\/\//i, "");
  }
  shown = shown.replace(/\/$/, "");
  try {
    shown = decodeURI(shown);
  } catch {
    // Show it encoded.
  }
  return shown.replace(BIDI_CONTROLS, "");
}

/** "invoice_INV-6-1.pdf" -> "Invoice Inv 6 1", as the file button shows it. */
export function fileTitle(name: string): string {
  const base = name
    .replace(/\.[^./]{1,8}$/, "")
    .replace(/[_-]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
  return (base || name).replace(
    /\S+/g,
    (word) => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase(),
  );
}

// Browser zoom steps, as Chrome has them.
const ZOOM_STEPS = [
  0.25, 0.33, 0.5, 0.67, 0.75, 0.8, 0.9, 1, 1.1, 1.25, 1.5, 1.75, 2, 2.5, 3, 4,
  5,
];

function stepZoom(zoom: number, direction: 1 | -1): number {
  if (direction > 0)
    return ZOOM_STEPS.find((step) => step > zoom + 0.001) ?? zoom;
  return [...ZOOM_STEPS].reverse().find((step) => step < zoom - 0.001) ?? zoom;
}

// Bordered pill in light mode, filled in dark, like the toolbar controls it groups.
const PILL =
  "border border-border/80 bg-card dark:border-transparent dark:bg-accent";

type ButtonProps = {
  label: string;
  /** Key chord shown beside the label in the tooltip. */
  shortcut?: string | null;
  icon?: IconSvgElement;
  onClick?: () => void;
  disabled?: boolean;
  className?: string;
  children?: ReactNode;
};

function IconButton({
  label,
  shortcut,
  icon,
  onClick,
  disabled,
  className,
  children,
}: ButtonProps) {
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
          {icon ? (
            <HugeiconsIcon
              icon={icon}
              strokeWidth={1.75}
              className="size-4.5"
            />
          ) : (
            children
          )}
        </button>
      </TooltipTrigger>
      <TooltipContent side="bottom" className="tooltip-compact">
        {shortcut ? (
          <span className="flex items-center gap-1.5">
            {label}
            <kbd className="rounded bg-[rgb(0_0_0_/_calc(0.1*var(--contrast-wash-gain,1)))] px-1 py-px text-ui-10 font-medium leading-none dark:bg-[rgb(255_255_255_/_calc(0.15*var(--contrast-wash-gain,1)))]">
              {shortcut}
            </kbd>
          </span>
        ) : (
          label
        )}
      </TooltipContent>
    </Tooltip>
  );
}

/** A round toolbar button on its own pill. */
function CircleButton(props: ButtonProps) {
  return (
    <IconButton
      {...props}
      className={cn(
        PILL,
        "size-8 hover:bg-card dark:hover:bg-accent",
        props.className,
      )}
    />
  );
}

const DOCUMENT_KINDS = new Set(["pdf", "word", "spreadsheet", "presentation"]);

function KindIcon({
  name,
  contentType,
  className,
}: { name: string; contentType?: string; className?: string }) {
  const kind = attachmentFileKind(name, contentType);
  return (
    <HugeiconsIcon
      icon={ATTACHMENT_KIND_ICONS[kind]}
      strokeWidth={1.75}
      className={cn(
        "size-4 shrink-0",
        ATTACHMENT_KIND_ICON_CLASS[kind],
        className,
      )}
    />
  );
}

function TabIcon({ tab }: { tab: BrowserTab }) {
  const [failedFor, setFailedFor] = useState<string | null>(null);
  const entry = currentEntry(tab);
  if (tab.loading) return <Spinner className="size-4 shrink-0" />;
  if (entry.kind === "file")
    return <KindIcon name={entry.name} contentType={entry.contentType} />;
  if (entry.kind === "internal") {
    const icon = entry.page === "history" ? Clock01Icon : Download01Icon;
    return (
      <HugeiconsIcon
        icon={icon}
        strokeWidth={1.75}
        className="size-4 shrink-0"
      />
    );
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
    if (DOCUMENT_KINDS.has(attachmentFileKind(name, type)))
      return <KindIcon name={name} contentType={type} />;
  }
  return (
    <HugeiconsIcon
      icon={InternetIcon}
      strokeWidth={1.75}
      className="size-4 shrink-0"
    />
  );
}

function useTabTitle() {
  const t = useT();
  return (tab: BrowserTab, entry: BrowserEntry) => {
    if (tab.title) return tab.title;
    if (entry.kind === "newtab") return t("browser.newTab");
    if (entry.kind === "internal")
      return t(
        entry.page === "history"
          ? "browser.pages.history"
          : "browser.pages.downloads",
      );
    // Unloaded background tabs show their site.
    return entry.kind === "web" ? hostOf(entry.url) : entry.name;
  };
}

function TabStrip({
  tabs,
  activeTabId,
}: { tabs: BrowserTab[]; activeTabId: string | null }) {
  const t = useT();
  const tabTitle = useTabTitle();
  const fullView = useBrowserStore((state) => state.fullView);
  const { activateTab, closeTab, newTab, closePanel, setFullView } =
    useBrowserStore.getState();
  const fullViewShortcut = useShortcutLabel("toggleBrowserFullView");
  useShortcut("toggleBrowserFullView", (event) => {
    event.preventDefault();
    const state = useBrowserStore.getState();
    state.setFullView(!state.fullView);
  });
  return (
    // Above the desktop titlebar's drag strip (z-40), which would swallow tab clicks.
    <div
      data-tauri-drag-region={true}
      className="relative z-40 flex h-[var(--studio-chat-header-height,48px)] min-w-0 shrink-0 items-center gap-1 pl-1.5 pr-[calc(0.5rem*var(--ui-space-scale,1)+var(--studio-window-control-inset,0px))]"
    >
      <div
        role="tablist"
        data-tauri-drag-region={true}
        aria-label={t("browser.tabs")}
        className="flex min-w-0 flex-1 items-center overflow-x-auto [scrollbar-width:none]"
      >
        {tabs.map((tab, index) => {
          const active = tab.id === activeTabId;
          const title = tabTitle(tab, currentEntry(tab));
          // Dividers sit between inactive tabs only.
          const divided =
            index > 0 && !active && tabs[index - 1]?.id !== activeTabId;
          return (
            <div
              key={tab.id}
              className="flex min-w-24 max-w-56 flex-1 items-center"
            >
              <span
                aria-hidden={true}
                className={cn(
                  "h-4 w-px shrink-0 bg-border",
                  !divided && "invisible",
                )}
              />
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
                  if (event.key === "Enter" || event.key === " ")
                    activateTab(tab.id);
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
                    !active &&
                      "opacity-0 group-hover/tab:opacity-100 focus-visible:opacity-100",
                  )}
                >
                  <HugeiconsIcon
                    icon={Cancel01Icon}
                    strokeWidth={2}
                    className="size-3.5"
                  />
                </button>
              </div>
            </div>
          );
        })}
      </div>
      <IconButton
        label={t("browser.newTab")}
        icon={Add01Icon}
        onClick={newTab}
        className="size-8"
      />
      <span aria-hidden={true} className="mx-1 h-4 w-px shrink-0 bg-border" />
      <IconButton
        label={t(fullView ? "browser.fullView.exit" : "browser.fullView.enter")}
        shortcut={fullViewShortcut}
        icon={fullView ? ExitFullViewIcon : EnterFullViewIcon}
        onClick={() => setFullView(!fullView)}
        className="size-8"
      />
      <IconButton
        label={t("browser.close")}
        icon={SplitPaneIcon}
        onClick={closePanel}
        className="size-8 rounded-[10px] bg-[color-mix(in_oklab,var(--foreground)_calc(6%*var(--contrast-wash-gain,1)),transparent)] text-foreground"
      />
    </div>
  );
}

// The last focusAddress request an address bar acted on.
let handledFocusSequence = 0;

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
    // Each request once: the bar remounts per tab, and refocusing would leave it editing.
    if (focusSequence === handledFocusSequence) return;
    const input = inputRef.current;
    if (!input) return;
    handledFocusSequence = focusSequence;
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
            <span className="truncate">
              {displayAddress(address, showFullUrl)}
            </span>
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
    return blob
      ? { blob, name: entry.name, contentType: entry.contentType, url: null }
      : undefined;
  }
  const page = pageDownload(tab.id);
  return page
    ? {
        ...page,
        url: tab.displayUrl ?? (entry.kind === "web" ? entry.url : null),
      }
    : undefined;
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
    <div
      className={cn(
        PILL,
        "flex h-8 shrink-0 items-center gap-0.5 rounded-full px-0.5",
      )}
    >
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
  return Boolean(
    tab &&
      currentEntry(tab).kind === "web" &&
      !tab.documentType &&
      !tab.loading,
  );
}

function ZoomControl({ tab }: { tab: BrowserTab | undefined }) {
  const t = useT();
  const locale = useLocale();
  const zoom = tab?.zoom ?? 1;
  const setZoom = (next: number) =>
    tab && useBrowserStore.getState().setZoom(tab.id, next);
  const percent = new Intl.NumberFormat(locale, {
    style: "percent",
    maximumFractionDigits: 0,
  }).format(zoom);
  const step =
    "flex h-full w-8 cursor-pointer items-center justify-center text-muted-foreground hover:text-foreground disabled:cursor-default disabled:opacity-35";
  return (
    <div className="flex items-center gap-2 py-1 pl-3 pr-1 text-sm">
      <span className="flex-1">{t("browser.menu.zoom")}</span>
      <div className="flex h-8 items-center rounded-[10px] border border-border">
        <button
          type="button"
          aria-label={t("browser.menu.zoomOut")}
          disabled={!tab || zoom <= (ZOOM_STEPS[0] ?? 0)}
          onClick={() => setZoom(stepZoom(zoom, -1))}
          className={step}
        >
          <HugeiconsIcon
            icon={MinusSignIcon}
            strokeWidth={1.75}
            className="size-3.5"
          />
        </button>
        <span className="min-w-14 border-x border-border text-center tabular-nums">
          {percent}
        </span>
        <button
          type="button"
          aria-label={t("browser.menu.zoomIn")}
          disabled={!tab || zoom >= (ZOOM_STEPS[ZOOM_STEPS.length - 1] ?? 5)}
          onClick={() => setZoom(stepZoom(zoom, 1))}
          className={step}
        >
          <HugeiconsIcon
            icon={PlusSignIcon}
            strokeWidth={1.75}
            className="size-3.5"
          />
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
          <AlertDialogDescription>
            {t(nativeBrowser ? "browser.native.clearDataDescription" : "browser.clearData.description")}
          </AlertDialogDescription>
        </AlertDialogHeader>
        <AlertDialogFooter>
          <AlertDialogCancel>{t("browser.clearData.cancel")}</AlertDialogCancel>
          <AlertDialogAction
            onClick={() => {
              const history = useBrowserHistoryStore.getState();
              history.clearHistory();
              history.clearDownloads();
              clearPageCache();
              clearNativeBrowsingData();
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
  const mod =
    typeof navigator !== "undefined" &&
    /Mac|iPhone|iPad/.test(navigator.platform)
      ? "⌘"
      : "Ctrl+";
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
                <HugeiconsIcon
                  icon={MoreHorizontalIcon}
                  strokeWidth={1.75}
                  className="size-4.5"
                />
              </button>
            </DropdownMenuTrigger>
          </TooltipTrigger>
          <TooltipContent side="bottom" className="tooltip-compact">
            {t("browser.more")}
          </TooltipContent>
        </Tooltip>
        <DropdownMenuContent
          align="end"
          sideOffset={6}
          className="min-w-72 rounded-[20px] p-1.5 [&_[data-slot=dropdown-menu-separator]]:mx-3 [&_[data-slot=dropdown-menu-separator]]:my-1.5"
        >
          <DropdownMenuItem
            disabled={!webPage}
            onSelect={() => store.setFindOpen(true)}
          >
            {t("browser.menu.find")}
            <DropdownMenuShortcut>{mod}F</DropdownMenuShortcut>
          </DropdownMenuItem>
          <DropdownMenuItem
            disabled={!webUrl}
            onSelect={() =>
              webUrl &&
              void copyToClipboard(webUrl).then(
                (ok) => ok && toast.success(t("browser.linkCopied")),
              )
            }
          >
            {t("browser.copyLink")}
          </DropdownMenuItem>
          <DropdownMenuSeparator />
          <ZoomControl tab={tab} />
          <DropdownMenuSeparator />
          <DropdownMenuItem
            disabled={!webPage && device === "off"}
            onSelect={() =>
              store.setDevice(device === "off" ? "mobile" : "off")
            }
          >
            {t(
              device === "off"
                ? "browser.menu.deviceToolbar"
                : "browser.device.close",
            )}
          </DropdownMenuItem>
          <DropdownMenuSeparator />
          <DropdownMenuItem onSelect={() => store.openInternal("downloads")}>
            {t("browser.pages.downloads")}
          </DropdownMenuItem>
          <DropdownMenuItem onSelect={() => store.openInternal("history")}>
            {t("browser.pages.history")}
          </DropdownMenuItem>
          <DropdownMenuItem onSelect={() => setClearOpen(true)}>
            {t("browser.menu.clearData")}
          </DropdownMenuItem>
          <DropdownMenuSeparator />
          <DropdownMenuItem onSelect={() => useSettingsDialogStore.getState().openDialog("browser")}>
            {t("browser.settings")}
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>
      <ClearBrowsingDataDialog open={clearOpen} onOpenChange={setClearOpen} />
    </>
  );
}

/** Whether the tab shows a native view, whose own history Back and Forward use first. */
function nativePage(tab: BrowserTab | undefined): boolean {
  return Boolean(nativeBrowser && tab && currentEntry(tab).kind === "web" && hasNativeView(tab.id));
}

function WebToolbar({ tab }: { tab: BrowserTab | undefined }) {
  const t = useT();
  const { goBack, goForward, reload } = useBrowserStore.getState();
  const native = nativePage(tab);
  const canGoBack = Boolean(
    tab && (tab.index > 0 || (native && (tab.nativeError || tab.nativeHistory?.back))),
  );
  const canGoForward = Boolean(
    tab && (tab.index < tab.history.length - 1 || (native && tab.nativeHistory?.forward)),
  );
  const back = () => {
    if (!tab) return;
    if (native && tab.nativeError && returnToNativePage(tab.id)) return;
    if (native && tab.nativeHistory?.back) nativeAction(tab.id, "back");
    else goBack(tab.id);
  };
  const forward = () => {
    if (!tab) return;
    if (native && tab.nativeHistory?.forward) nativeAction(tab.id, "forward");
    else goForward(tab.id);
  };
  return (
    <>
      <div
        className={cn(
          PILL,
          "flex h-8 shrink-0 items-center gap-0.5 rounded-full px-0.5",
        )}
      >
        <IconButton
          label={t("browser.back")}
          icon={ArrowLeft02Icon}
          disabled={!canGoBack}
          onClick={back}
        />
        <IconButton
          label={t("browser.forward")}
          icon={ArrowRight02Icon}
          disabled={!canGoForward}
          onClick={forward}
        />
        <span
          aria-hidden={true}
          className="mx-0.5 h-4 w-px shrink-0 bg-[color-mix(in_oklab,var(--foreground)_calc(15%*var(--contrast-wash-gain,1)),transparent)]"
        />
        <IconButton
          label={t("browser.reload")}
          disabled={!tab || currentEntry(tab).kind !== "web"}
          onClick={() => {
            if (!tab) return;
            if (native) nativeAction(tab.id, "reload");
            else reload(tab.id);
          }}
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
function FileToolbar({
  tab,
  entry,
}: { tab: BrowserTab; entry: Extract<BrowserEntry, { kind: "file" }> }) {
  const t = useT();
  const navigate = useNavigate();
  const requestEdits = useBrowserStore((state) => state.requestEdits);
  const canAnnotate = useBrowserStore((state) => state.sendAnnotations !== null);
  const annotating = useBrowserStore((state) => state.annotateTabId === tab.id);
  const openInCanvas = useBrowserStore((state) => state.openInCanvas);
  const view = useBrowserStore((state) => state.fileViews[tab.id]) ?? DEFAULT_FILE_VIEW;
  const [copied, setCopied] = useState(false);
  const download = tabDownload(tab);
  const blob = download?.blob;
  const kind = textFileKind(entry.name, entry.contentType, entry.plainText);
  // HTML and Markdown render, so they switch to their source as the canvas does.
  const hasSource = kind === "html" || kind === "markdown";
  const showsSource = kind === "code" || kind === "text" || (hasSource && view.mode === "source");
  const htmlPreview = kind === "html" && view.mode === "preview";
  const setView = (patch: Partial<FileViewState>) => useBrowserStore.getState().setFileView(tab.id, patch);
  // The preview is a frame the annotation layer can't select in, so marks go on the source.
  useEffect(() => {
    if (annotating && htmlPreview) useBrowserStore.getState().setAnnotating(null);
  }, [annotating, htmlPreview]);
  const toggleAnnotating = () => {
    if (!annotating && htmlPreview) setView({ mode: "source" });
    useBrowserStore.getState().setAnnotating(annotating ? null : tab.id);
  };
  const copyContents = () => {
    if (!blob) return;
    void copyToClipboardFrom(() => blob.text()).then((ok) => {
      if (!ok) {
        toast.error(t("browser.file.copyFailed"));
        return;
      }
      setCopied(true);
      window.setTimeout(() => setCopied(false), 2000);
    });
  };
  const runAgain = () => {
    setView({ mode: "preview" });
    useBrowserStore.getState().reload(tab.id);
  };
  const toggleConsole = () =>
    htmlPreview
      ? setView({ consoleOpen: !view.consoleOpen })
      : setView({ mode: "preview", consoleOpen: true });
  const openCanvas = () => {
    if (!blob || !openInCanvas) return;
    void blob.text().then((code) => openInCanvas({ title: fileTitle(entry.name), code }));
  };
  const openInNewChat = () => {
    if (!blob) return;
    startLibraryChat(navigate, {
      files: [
        new File([blob], entry.name, { type: entry.contentType || blob.type }),
      ],
    });
  };
  // Null for HTML and SVG, which would run on Studio's origin from a blob URL.
  const tabType = browserTabType(entry.name, entry.contentType || blob?.type || "");
  const openInBrowser = () => {
    if (!blob || !tabType) return;
    const url = URL.createObjectURL(new Blob([blob], { type: tabType }));
    window.open(url, "_blank", "noopener,noreferrer");
    window.setTimeout(() => URL.revokeObjectURL(url), 60_000);
  };
  return (
    <>
      <DropdownMenu>
        <DropdownMenuTrigger asChild={true}>
          <button
            type="button"
            className={cn(
              PILL,
              "flex h-9 min-w-24 max-w-[55%] shrink cursor-pointer items-center gap-2 rounded-full pl-3.5 pr-3 text-ui-13p5 text-foreground outline-none transition-colors focus-visible:ring-2 focus-visible:ring-ring",
            )}
          >
            <KindIcon
              name={entry.name}
              contentType={entry.contentType}
              className="size-4.5"
            />
            <span className="min-w-0 truncate">{fileTitle(entry.name)}</span>
            <HugeiconsIcon
              icon={ChevronDownStandardIcon}
              strokeWidth={1.75}
              className="size-4 shrink-0 text-muted-foreground"
            />
          </button>
        </DropdownMenuTrigger>
        <DropdownMenuContent
          align="start"
          sideOffset={6}
          className="w-80 max-w-[calc(100vw-2rem)] rounded-[20px] p-1.5 [&_[data-slot=dropdown-menu-separator]]:mx-3 [&_[data-slot=dropdown-menu-separator]]:my-1.5"
        >
          <div className="flex items-start gap-3 px-3 py-2 text-sm">
            <KindIcon
              name={entry.name}
              contentType={entry.contentType}
              className="mt-0.5 size-4.5"
            />
            <span className="min-w-0 break-words">{entry.name}</span>
          </div>
          <DropdownMenuSeparator />
          <DropdownMenuSub>
            <DropdownMenuSubTrigger disabled={!blob}>
              <HugeiconsIcon
                icon={ArrowUpRight01Icon}
                strokeWidth={1.75}
                className="size-4"
              />
              {t("browser.file.openIn")}
            </DropdownMenuSubTrigger>
            <DropdownMenuSubContent className="min-w-52 rounded-[20px] p-1.5">
              <DropdownMenuItem onSelect={openInNewChat}>
                <HugeiconsIcon
                  icon={BubbleChatAddIcon}
                  strokeWidth={1.75}
                  className="size-4.5"
                />
                {t("browser.file.newChat")}
              </DropdownMenuItem>
              {kind === "html" && openInCanvas ? (
                <DropdownMenuItem onSelect={openCanvas}>
                  <HugeiconsIcon
                    icon={PaintBoardIcon}
                    strokeWidth={1.75}
                    className="size-4.5"
                  />
                  {t("browser.file.canvas")}
                </DropdownMenuItem>
              ) : null}
              {/* A blob URL can't be handed to another app from the desktop app. */}
              {isTauri || !tabType ? null : (
                <DropdownMenuItem onSelect={openInBrowser}>
                  <HugeiconsIcon
                    icon={InternetIcon}
                    strokeWidth={1.75}
                    className="size-4.5"
                  />
                  {t("browser.file.newBrowserTab")}
                </DropdownMenuItem>
              )}
            </DropdownMenuSubContent>
          </DropdownMenuSub>
          {kind === "html" ? (
            <>
              <DropdownMenuItem onSelect={runAgain}>
                <RefreshGlyph strokeWidth={1.75} className="size-4" />
                {t("browser.file.runAgain")}
              </DropdownMenuItem>
              <DropdownMenuItem onSelect={toggleConsole}>
                <HugeiconsIcon
                  icon={ComputerTerminal01Icon}
                  strokeWidth={1.75}
                  className="size-4"
                />
                {t("browser.file.console")}
                {view.errorCount > 0 ? (
                  <span className="ml-auto rounded-full bg-destructive px-1.5 text-ui-10 font-medium leading-4 text-destructive-foreground">
                    {view.errorCount}
                  </span>
                ) : null}
              </DropdownMenuItem>
            </>
          ) : null}
          {kind ? (
            <DropdownMenuItem disabled={!blob} onSelect={copyContents}>
              <HugeiconsIcon
                icon={Copy01Icon}
                strokeWidth={1.75}
                className="size-4"
              />
              {t("browser.file.copy")}
            </DropdownMenuItem>
          ) : null}
          <DropdownMenuItem
            disabled={!download}
            onSelect={() => download && void saveBrowserDownload(download)}
          >
            <HugeiconsIcon
              icon={Download01Icon}
              strokeWidth={1.75}
              className="size-4"
            />
            {t("browser.download")}
          </DropdownMenuItem>
          {showsSource ? (
            <>
              <DropdownMenuSeparator />
              <DropdownMenuCheckboxItem
                checked={view.wrap || kind === "text"}
                disabled={kind === "text"}
                onCheckedChange={(wrap) => setView({ wrap })}
                onSelect={(event) => event.preventDefault()}
              >
                <HugeiconsIcon
                  icon={TextWrapIcon}
                  strokeWidth={1.75}
                  className="size-4"
                />
                {t("browser.file.wrap")}
              </DropdownMenuCheckboxItem>
            </>
          ) : null}
        </DropdownMenuContent>
      </DropdownMenu>
      {requestEdits || canAnnotate ? (
        <button
          type="button"
          aria-pressed={canAnnotate ? annotating : undefined}
          aria-label={t("browser.file.requestEdits")}
          // Marks parts of the file to change, as ChatGPT does; without a chat to send them to, it
          // stages a prompt naming the file instead.
          onClick={() =>
            canAnnotate
              ? toggleAnnotating()
              : requestEdits?.(
                  t("browser.file.requestEditsPrompt", { name: entry.name }),
                )
          }
          className={cn(
            PILL,
            "flex h-9 shrink-0 cursor-pointer items-center gap-2 rounded-full px-2.5 text-ui-13p5 text-foreground outline-none transition-colors focus-visible:ring-2 focus-visible:ring-ring @[34rem]:px-3.5",
            annotating && "text-primary",
          )}
        >
          <HugeiconsIcon
            icon={CursorRectangleSelection02Icon}
            strokeWidth={1.75}
            className="size-4.5"
          />
          <span className="hidden @[34rem]:inline">{t("browser.file.requestEdits")}</span>
        </button>
      ) : null}
      <span
        aria-hidden={true}
        className="min-w-0 flex-1 pointer-events-none!"
      />
      {hasSource ? (
        <div
          role="tablist"
          aria-label={t("browser.file.viewMode")}
          className={cn(PILL, "flex h-9 shrink-0 items-center gap-0.5 rounded-full p-0.5")}
        >
          {(["preview", "source"] as const).map((mode) => (
            <Tooltip key={mode}>
              <TooltipTrigger asChild={true}>
                <button
                  type="button"
                  role="tab"
                  aria-selected={view.mode === mode}
                  aria-label={t(`browser.file.${mode}`)}
                  onClick={() => setView({ mode })}
                  className={cn(
                    "flex size-8 cursor-pointer items-center justify-center rounded-full text-muted-foreground transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
                    view.mode === mode
                      ? "bg-background text-foreground shadow-sm dark:bg-card"
                      : "hover:text-foreground",
                  )}
                >
                  <HugeiconsIcon
                    icon={mode === "preview" ? ViewIcon : SourceCodeIcon}
                    strokeWidth={1.75}
                    className="size-4"
                  />
                </button>
              </TooltipTrigger>
              <TooltipContent side="bottom" className="tooltip-compact">
                {t(`browser.file.${mode}`)}
              </TooltipContent>
            </Tooltip>
          ))}
        </div>
      ) : null}
      {kind === "html" ? (
        <>
          <CircleButton
            label={t("browser.file.runAgain")}
            onClick={runAgain}
            // In a narrow pane these fold into the file menu, keeping the file's name in view.
            className="hidden size-9 @[40rem]:flex"
          >
            <RefreshGlyph strokeWidth={1.75} className="size-4" />
          </CircleButton>
          <Tooltip>
            <TooltipTrigger asChild={true}>
              <button
                type="button"
                aria-label={t("browser.file.console")}
                aria-pressed={htmlPreview && view.consoleOpen}
                onClick={toggleConsole}
                className={cn(
                  PILL,
                  "hidden h-9 min-w-9 @[40rem]:flex shrink-0 cursor-pointer items-center justify-center gap-1.5 rounded-full px-2.5 text-muted-foreground transition-colors hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
                  htmlPreview && view.consoleOpen && "text-foreground",
                )}
              >
                <HugeiconsIcon
                  icon={ComputerTerminal01Icon}
                  strokeWidth={1.75}
                  className="size-4"
                />
                {view.errorCount > 0 ? (
                  <span className="rounded-full bg-destructive px-1.5 text-ui-10 font-medium leading-4 text-destructive-foreground">
                    {view.errorCount}
                  </span>
                ) : null}
              </button>
            </TooltipTrigger>
            <TooltipContent side="bottom" className="tooltip-compact">
              {t("browser.file.console")}
            </TooltipContent>
          </Tooltip>
        </>
      ) : null}
      {kind ? (
        <CircleButton
          label={copied ? t("browser.file.copied") : t("browser.file.copy")}
          icon={copied ? Tick02Icon : Copy01Icon}
          disabled={!blob}
          onClick={copyContents}
          className="hidden size-9 @[40rem]:flex"
        />
      ) : null}
      <ScaleMenu
        value={tab.zoom}
        scales={ATTACHMENT_PAGE_SCALES}
        onChange={(value) =>
          useBrowserStore
            .getState()
            .setZoom(tab.id, value === "fit" ? 1 : value)
        }
        className={cn(PILL, "mr-0 hidden h-9 hover:bg-card @[28rem]:flex dark:hover:bg-accent")}
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
    if (nativePage(tab)) {
      void nativeFind(tab.id, query, backwards).then((found) => setFindMiss(!found));
      return;
    }
    if (!sendFrameCommand(tab.id, { command: "find", query, backwards }))
      setFindMiss(true);
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
          miss &&
            query &&
            "ring-2 ring-destructive/40 focus:ring-destructive/40",
        )}
      />
      {miss && query ? (
        <span className="shrink-0 text-ui-12 text-muted-foreground">
          {t("browser.find.noMatches")}
        </span>
      ) : null}
      <div
        className={cn(
          PILL,
          "flex h-8 shrink-0 items-center gap-0.5 rounded-full px-0.5",
        )}
      >
        <IconButton
          label={t("browser.find.previous")}
          icon={ArrowUp01Icon}
          disabled={!query}
          onClick={() => find(true)}
        />
        <IconButton
          label={t("browser.find.next")}
          icon={ArrowDown01Icon}
          disabled={!query}
          onClick={() => find()}
        />
      </div>
      <CircleButton
        label={t("browser.find.close")}
        icon={Cancel01Icon}
        onClick={() => setFindOpen(false)}
      />
    </div>
  );
}

const DEVICE_WIDTHS: Record<Exclude<DeviceMode, "off">, number> = {
  mobile: 390,
  tablet: 820,
};

function DeviceBar() {
  const t = useT();
  const device = useBrowserStore((state) => state.device);
  const { setDevice } = useBrowserStore.getState();
  const option = (
    mode: Exclude<DeviceMode, "off">,
    icon: IconSvgElement,
    label: string,
  ) => (
    <button
      type="button"
      aria-pressed={device === mode}
      onClick={() => setDevice(mode)}
      className={cn(
        "flex h-7 cursor-pointer items-center gap-1.5 rounded-full px-3 text-ui-12 transition-colors",
        device === mode
          ? "bg-[color-mix(in_oklab,var(--foreground)_calc(8%*var(--contrast-wash-gain,1)),transparent)] text-foreground"
          : "text-muted-foreground hover:text-foreground",
      )}
    >
      <HugeiconsIcon icon={icon} strokeWidth={1.75} className="size-3.5" />
      {label}
      <span className="tabular-nums text-muted-foreground">
        {DEVICE_WIDTHS[mode]}
      </span>
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
// Hidden web pages run scripts on the chat's thread, so only the newest stays live.
const MAX_HIDDEN_WEB_PAGES = 1;

/** Mounted tabs: all, less hidden web pages past the newest few. */
function liveTabIds(mounted: readonly string[], tabs: BrowserTab[], activeTabId: string | null): Set<string> {
  const byId = new Map(tabs.map((tab) => [tab.id, tab]));
  const live = new Set<string>();
  let hiddenWeb = 0;
  // Newest first.
  for (const id of [...mounted].reverse()) {
    const tab = byId.get(id);
    if (!tab) continue;
    if (id !== activeTabId && currentEntry(tab).kind === "web") {
      if (hiddenWeb >= MAX_HIDDEN_WEB_PAGES) continue;
      hiddenWeb += 1;
    }
    live.add(id);
  }
  return live;
}

/** The chat's in-app browser: tabs of web pages and opened files. Memoized so chat renders skip it. */
export const BrowserPanel = memo(function BrowserPanel() {
  const t = useT();
  const tabs = useBrowserStore((state) => state.tabs);
  const activeTabId = useBrowserStore((state) => state.activeTabId);
  const findOpen = useBrowserStore((state) => state.findOpen);
  const device = useBrowserStore((state) => state.device);
  const annotateTabId = useBrowserStore((state) => state.annotateTabId);
  const [pageElement, setPageElement] = useState<HTMLDivElement | null>(null);
  useEffect(() => (nativeBrowser ? startNativeViews() : undefined), []);
  const activeTab = tabs.find((tab) => tab.id === activeTabId);
  const activeEntry = activeTab ? currentEntry(activeTab) : null;
  // Mount tabs on first view. Most recent last.
  const [mounted, setMounted] = useState<readonly string[]>(() =>
    activeTabId ? [activeTabId] : [],
  );
  if (activeTabId && mounted[mounted.length - 1] !== activeTabId) {
    const open = new Set(tabs.map((tab) => tab.id));
    setMounted(
      [
        ...mounted.filter((id) => id !== activeTabId && open.has(id)),
        activeTabId,
      ].slice(-MAX_MOUNTED_TABS),
    );
  }
  const live = liveTabIds(mounted, tabs, activeTabId);
  // Files sit on the viewer's gray, with no line under the toolbar.
  const fileTab = activeEntry?.kind === "file";
  const documentShown = fileTab || Boolean(activeTab?.documentType);
  const deviceWidth =
    device !== "off" && activeEntry?.kind === "web"
      ? DEVICE_WIDTHS[device]
      : null;

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
          "relative flex min-h-0 flex-1 flex-col overflow-hidden rounded-tl-[14px] border-t border-border/60 dark:border-transparent",
          documentShown
            ? "bg-[color-mix(in_oklab,var(--muted)_55%,var(--card))]"
            : "bg-card",
        )}
      >
        <div
          className={cn(
            "flex shrink-0 items-center gap-1.5 px-2.5 py-2",
            // A file's controls float over it with nothing behind them, so the page scrolls under.
            fileTab &&
              "browser-file-toolbar @container pointer-events-none absolute inset-x-0 top-0 z-20 *:pointer-events-auto",
          )}
        >
          {activeTab && activeEntry?.kind === "file" ? (
            <FileToolbar tab={activeTab} entry={activeEntry} />
          ) : (
            <WebToolbar tab={activeTab} />
          )}
        </div>
        {findOpen && showsWebPage(activeTab) ? (
          <FindBar key={activeTabId} tab={activeTab} />
        ) : null}
        {deviceWidth ? <DeviceBar /> : null}
        <div
          ref={setPageElement}
          className={cn(
            "relative min-h-0 flex-1 overflow-hidden",
            fileTab ? "browser-file-page" : "border-t border-border/70",
            documentShown ? "bg-transparent" : "bg-background",
            deviceWidth && "bg-muted/60",
          )}
        >
          {activeTab?.loading ? (
            <div className="absolute inset-x-0 top-0 z-10 h-[2.5px] overflow-hidden">
              <span
                aria-hidden={true}
                className="artifact-loading-line block h-full rounded-full motion-reduce:hidden"
              />
            </div>
          ) : null}
          <div
            className={cn(
              "relative mx-auto h-full",
              deviceWidth &&
                "border-x border-border/70 bg-background shadow-sm",
            )}
            style={
              deviceWidth ? { width: `min(100%, ${deviceWidth}px)` } : undefined
            }
          >
            {tabs.map((tab) =>
              live.has(tab.id) ? (
                <TabView
                  key={tab.id}
                  tab={tab}
                  active={tab.id === activeTabId}
                />
              ) : null,
            )}
          </div>
          {activeTab &&
          activeEntry?.kind === "file" &&
          annotateTabId === activeTab.id &&
          pageElement ? (
            <AnnotateLayer
              key={activeTab.id}
              page={pageElement}
              fileName={activeEntry.name}
            />
          ) : null}
        </div>
      </div>
    </section>
  );
});
