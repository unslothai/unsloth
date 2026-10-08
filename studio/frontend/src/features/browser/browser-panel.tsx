// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { ATTACHMENT_PAGE_SCALES } from "@/components/assistant-ui/attachment-viewer-meta";
import { ScaleMenu } from "@/components/media-viewer";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import {
  ContextMenu,
  ContextMenuContent,
  ContextMenuTrigger,
} from "@/components/ui/context-menu";
import {
  DropdownMenu,
  DropdownMenuCheckboxItem,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
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
  useChatRuntimeStore,
} from "@/features/chat";
import { formatBytes } from "@/features/hub";
import { startLibraryChat } from "@/features/library";
import {
  useSettingsDialogStore,
  useShortcut,
  useShortcutLabel,
} from "@/features/settings";
import { FIND_SKIP_ATTRIBUTE, requestFind } from "@/features/find-in-page";
import { registerZoomScope } from "@/features/interface-zoom";
import { useLocale, useT } from "@/i18n";
import { isTauri } from "@/lib/api-base";
import { ChevronDownStandardIcon } from "@/lib/chevron-icons";
import { copyToClipboard, copyToClipboardFrom } from "@/lib/copy-to-clipboard";
import { openExternalLink } from "@/lib/open-link";
import { RefreshGlyph } from "@/lib/refresh-icon";
import { ShieldAlertGlyph } from "@/lib/shield-alert-icon";
import { Tick02Icon } from "@/lib/tick-icon";
import { toast } from "@/lib/toast";
import { cn } from "@/lib/utils";
import {
  Add01Icon,
  Search01Icon,
  Settings02Icon,
  Delete02Icon,
  ArrowUpRight01Icon,
  BubbleChatAddIcon,
  Cancel01Icon,
  Clock01Icon,
  ComputerTerminal01Icon,
  Copy01Icon,
  CursorRectangleSelection02Icon,
  Download01Icon,
  InternetIcon,
  MinusSignIcon,
  MoreHorizontalIcon,
  PlusSignIcon,
  SmartPhone01Icon,
  SourceCodeIcon,
  StarIcon,
  Tablet01Icon,
  TextWrapIcon,
  ViewIcon,
  VolumeMute02Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon, type IconSvgElement } from "@hugeicons/react";
import {
  ArrowLeft,
  ArrowRight,
  ChevronRight,
  RotateCw,
  ShieldCheck,
  XIcon,
} from "lucide-react";
import { useNavigate } from "@tanstack/react-router";
import {
  type PointerEvent as ReactPointerEvent,
  type ReactElement,
  type ReactNode,
  type RefObject,
  memo,
  useEffect,
  useRef,
  useState,
} from "react";
import { fileNameFromUrl, hostOf, resolveAddress } from "./address";
import { OtherSurfaceError, canPrintFrames, printPage, screenshotPage } from "./capture";
import { canScreenshot } from "./screenshot-support";
import { stageEditsPrompt } from "./stage-edits";
import { type BrowserDownload, saveBrowserDownload, saveNeedsClick } from "./downloads";
import { BROWSER_FIND_TARGET, registerBrowserFind } from "./find";
import { ClearBrowsingDataDialog } from "./clear-data-dialog";
import { SiteFavicon } from "./site-favicon";
import { AnnotateLayer, WebAnnotateLayer } from "./annotate-layer";
import { BookmarkStar, BookmarksBar } from "./bookmarks";
import { useBookmarkFor } from "./bookmarks-store";
import { browserTabType, mediaKind, textFileKind } from "./file-kind";
import { canCopyVideoFrame, copyVideoFrame, tabVideo } from "./video-registry";
import { CONTEXT_MENU } from "./link-context-menu";
import { CONTEXT_TAB_MENU, TabMenuItems, focusRenameField, renameTabTo, setTabMuted } from "./tab-menu";
import {
  CertificateIcon,
  EnterFullViewIcon,
  ExitFullViewIcon,
  PadlockIcon,
  PadlockOpenIcon,
} from "./icons";
import {
  hasNativeView,
  nativeAction,
  returnToNativePage,
  startNativeViews,
  useNativeBrowser,
} from "./native-view";
import { type BookmarksToolbarMode, useBrowserPrefsStore } from "./prefs-store";
import {
  type BrowserEntry,
  type BrowserTab,
  DEFAULT_FILE_VIEW,
  MAX_TAB_TITLE_CHARS,
  type DeviceMode,
  type FileViewState,
  browserFile,
  cachedPage,
  currentEntry,
  entryKey,
  pageDownload,
  useBrowserStore,
} from "./store";
import { TabView } from "./tab-view";
import { ZOOM_STEPS, canZoom, homeZoom, stepZoom, zoomTab } from "./zoom";

function tabAddress(tab: BrowserTab | undefined): string {
  if (!tab) return "";
  const entry = currentEntry(tab);
  if (entry.kind === "web") return tab.displayUrl ?? entry.url;
  if (entry.kind === "file") return entry.name;
  return "";
}

// Bidi controls can reverse the text after them, making one site's path read like another's.
const BIDI_CONTROLS = /[\u061c\u200e\u200f\u202a-\u202e\u2066-\u2069]/g;

/** Display address: no scheme/trailing slash, site+path unless full; punycode host, no credentials. */
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

// Borderless: the floating toolbar shadow (index.css) separates it. Dark values live in
// variables so dark: cannot outrank hover.
const PILL_SURFACE = "bg-(--pill-bg) [--pill-bg:var(--card)] dark:[--pill-bg:var(--accent)]";
// Hover and press shade the pill: darker in light mode, lighter in dark.
const PILL = cn(
  PILL_SURFACE,
  "[--pill-hover:8%] [--pill-press:12%] dark:[--pill-hover:7%] dark:[--pill-press:12%]",
  "transition-colors disabled:pointer-events-none",
  "hover:bg-[color-mix(in_oklab,var(--pill-bg),var(--foreground)_var(--pill-hover))]",
  "data-[state=open]:bg-[color-mix(in_oklab,var(--pill-bg),var(--foreground)_var(--pill-hover))]",
  "active:bg-[color-mix(in_oklab,var(--pill-bg),var(--foreground)_var(--pill-press))]",
);

const TOOLBAR_BUTTON =
  "size-8 text-foreground disabled:hover:text-foreground disabled:opacity-30";

const NAV_BUTTON =
  "size-8 rounded-md text-foreground disabled:hover:text-foreground disabled:opacity-30";
const NAV_ICON = "size-4.5";
const ANNOTATE_BUTTON = "size-8 shrink-0 rounded-full p-0 text-foreground";
// The dashed box sits up-left of the glyph's centre, so nudge it to look centred.
const ANNOTATE_GLYPH = "size-4.5 translate-x-[4%] translate-y-[4%]";
const NAV_STROKE = 2;
const URLBAR =
  "bg-[color-mix(in_oklab,var(--foreground)_calc(6%*var(--contrast-wash-gain,1)),transparent)] focus-within:bg-[color-mix(in_oklab,var(--foreground)_calc(9%*var(--contrast-wash-gain,1)),transparent)]";

type ButtonProps = {
  label: string;
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
      <TooltipContent side="top" className="tooltip-compact">
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

function CircleButton(props: ButtonProps) {
  return (
    <IconButton
      {...props}
      className={cn(
        PILL,
        "size-8",
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
  mono = false,
}: {
  name: string;
  contentType?: string;
  className?: string;
  mono?: boolean;
}) {
  const kind = attachmentFileKind(name, contentType);
  return (
    <HugeiconsIcon
      icon={ATTACHMENT_KIND_ICONS[kind]}
      strokeWidth={1.75}
      className={cn(
        "size-4 shrink-0",
        !mono && ATTACHMENT_KIND_ICON_CLASS[kind],
        className,
      )}
    />
  );
}

const INTERNAL_PAGE_ICONS = {
  history: Clock01Icon,
  downloads: Download01Icon,
  bookmarks: StarIcon,
} as const;

function TabIcon({ tab }: { tab: BrowserTab }) {
  const [failedFor, setFailedFor] = useState<string | null>(null);
  const entry = currentEntry(tab);
  if (tab.loading) return <Spinner className="size-4 shrink-0" />;
  if (entry.kind === "file")
    return <KindIcon name={entry.name} contentType={entry.contentType} />;
  if (entry.kind === "internal") {
    const icon = INTERNAL_PAGE_ICONS[entry.page];
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
      className="mx-px size-3.5 shrink-0"
    />
  );
}

function useTabTitle() {
  const t = useT();
  return (tab: BrowserTab, entry: BrowserEntry) => {
    if (tab.customTitle) return tab.customTitle;
    if (tab.title) return tab.title;
    if (entry.kind === "newtab") return t("browser.newTab");
    if (entry.kind === "internal") return t(`browser.pages.${entry.page}`);
    return entry.kind === "web" ? hostOf(entry.url) : entry.name;
  };
}

function TabContextMenu({ tab, children }: { tab: BrowserTab; children: ReactElement }) {
  // Rename moves focus into the tab; the closing menu mustn't take it back to the tab.
  const keepFocus = useRef(false);
  return (
    <ContextMenu>
      <ContextMenuTrigger asChild={true}>{children}</ContextMenuTrigger>
      <ContextMenuContent
        className={CONTEXT_MENU}
        onCloseAutoFocus={(event) => {
          if (!keepFocus.current) return;
          keepFocus.current = false;
          event.preventDefault();
          // The menu's focus trap held focus while the field mounted, so autoFocus came to nothing.
          focusRenameField(`[data-tab-id="${CSS.escape(tab.id)}"] input`);
        }}
      >
        <TabMenuItems
          P={CONTEXT_TAB_MENU}
          tab={tab}
          strip={true}
          onRename={() => {
            keepFocus.current = true;
            useBrowserStore.getState().setRenamingTab(tab.id);
          }}
        />
      </ContextMenuContent>
    </ContextMenu>
  );
}

/** The tab's name, edited in place: Enter or leaving keeps it, Escape doesn't, empty clears it. */
function TabRenameInput({ tab, title }: { tab: BrowserTab; title: string }) {
  const t = useT();
  const done = useRef(false);
  const finish = (value: string | null) => {
    if (done.current) return;
    done.current = true;
    if (value !== null && value.trim() !== title) renameTabTo(tab, undefined, value);
    useBrowserStore.getState().setRenamingTab(null);
  };
  return (
    <input
      // biome-ignore lint/a11y/noAutofocus: opened from the tab's Rename, to type the name
      autoFocus={true}
      defaultValue={title}
      maxLength={MAX_TAB_TITLE_CHARS}
      aria-label={t("browser.tabMenu.renameLabel")}
      onFocus={(event) => event.currentTarget.select()}
      onBlur={(event) => finish(event.currentTarget.value)}
      onKeyDown={(event) => {
        event.stopPropagation();
        if (event.key === "Enter") finish(event.currentTarget.value);
        else if (event.key === "Escape") finish(null);
      }}
      // The tab drags and activates on these; the field is for typing and selecting.
      onPointerDown={(event) => event.stopPropagation()}
      onClick={(event) => event.stopPropagation()}
      onDoubleClick={(event) => event.stopPropagation()}
      className="min-w-0 flex-1 rounded-[5px] bg-background px-1 text-ui-13 text-foreground outline-none ring-1 ring-ring"
    />
  );
}

const TAB_DRAG_THRESHOLD_PX = 5;
const TAB_SHIFT_TRANSITION = "transform 160ms cubic-bezier(0.2, 0, 0, 1)";

/** Which way the tab at `position` slides while the one from `slot` is held over `target`: the
 *  tabs it has passed move one place toward where it came from. */
function tabShift(position: number, slot: number, target: number): -1 | 0 | 1 {
  if (slot < target && position > slot && position <= target) return -1;
  if (target < slot && position >= target && position < slot) return 1;
  return 0;
}

type TabDrag = {
  id: string;
  pointerId: number;
  startX: number;
  grabX: number;
  moved: boolean;
  elements: HTMLElement[];
  lefts: number[];
  widths: number[];
  slot: number;
  step: number;
  target: number;
};

/** Drag a tab to move it; the order changes once, on release: moving tabs mid-drag would move the dragged node and drop its pointer capture. */
function useTabDrag(listRef: RefObject<HTMLDivElement | null>, tabCount: number) {
  const drag = useRef<TabDrag | null>(null);
  // A drag ends in a click on the tab it started on; that click shouldn't count.
  const dragged = useRef(false);
  const reset = () => {
    const current = drag.current;
    drag.current = null;
    if (!current?.moved) return;
    for (const element of current.elements) {
      element.style.transition = "none";
      element.style.transform = "";
      element.style.zIndex = "";
    }
    requestAnimationFrame(() => {
      for (const element of current.elements) element.style.transition = "";
    });
  };
  // A tab opened or closed mid-drag makes the measured places wrong: drop the drag.
  // biome-ignore lint/correctness/useExhaustiveDependencies: the tab count is the trigger
  useEffect(() => reset(), [tabCount]);
  const begin = (current: TabDrag, element: HTMLElement) => {
    const list = listRef.current;
    if (!list) return false;
    const elements = [...list.querySelectorAll<HTMLElement>("[data-tab-id]")];
    const slot = elements.indexOf(element);
    if (slot < 0) return false;
    const rects = elements.map((tab) => tab.getBoundingClientRect());
    current.elements = elements;
    current.lefts = rects.map((rect) => rect.left);
    current.widths = rects.map((rect) => rect.width);
    current.slot = slot;
    current.target = slot;
    // Measured now, not on press: selecting the tab can scroll the strip under the pointer.
    current.grabX = current.startX - (rects[slot]?.left ?? 0);
    current.step = (rects[slot]?.width ?? 0) + (Number.parseFloat(getComputedStyle(list).columnGap) || 0);
    for (const [position, tab] of elements.entries()) {
      tab.style.transition = position === slot ? "none" : TAB_SHIFT_TRANSITION;
    }
    element.style.zIndex = "10";
    return true;
  };
  const follow = (current: TabDrag, clientX: number) => {
    const list = listRef.current;
    const element = current.elements[current.slot];
    if (!list || !element) return;
    const width = current.widths[current.slot] ?? 0;
    const natural = current.lefts[current.slot] ?? 0;
    const last = current.lefts.length - 1;
    const first = current.lefts[0] ?? natural;
    const end = (current.lefts[last] ?? natural) + (current.widths[last] ?? width) - width;
    const left = Math.min(Math.max(clientX - current.grabX, first), end);
    element.style.transform = `translateX(${left - natural}px)`;
    // The drop slot nearest where it's held: tabs share one width, so slots are a step apart.
    const target = Math.min(
      Math.max(current.slot + Math.round((left - natural) / (current.step || 1)), 0),
      last,
    );
    if (target === current.target) return;
    current.target = target;
    for (const [position, tab] of current.elements.entries()) {
      if (position === current.slot) continue;
      const shift = tabShift(position, current.slot, target) * current.step;
      tab.style.transform = shift ? `translateX(${shift}px)` : "";
    }
  };
  return {
    consumeClick: () => {
      const was = dragged.current;
      dragged.current = false;
      return was;
    },
    handlers: (tabId: string) => ({
      onPointerDown: (event: ReactPointerEvent<HTMLElement>) => {
        if (event.button !== 0 || (event.target as Element).closest("button")) return;
        dragged.current = false;
        drag.current = {
          id: tabId,
          pointerId: event.pointerId,
          startX: event.clientX,
          grabX: 0,
          moved: false,
          elements: [],
          lefts: [],
          widths: [],
          slot: -1,
          step: 0,
          target: -1,
        };
        event.currentTarget.setPointerCapture(event.pointerId);
        // Selected on press, as Firefox does, so the strip has settled before a drag measures it.
        useBrowserStore.getState().activateTab(tabId);
      },
      onPointerMove: (event: ReactPointerEvent<HTMLElement>) => {
        const current = drag.current;
        if (!current || current.pointerId !== event.pointerId) return;
        if (!current.moved) {
          if (Math.abs(event.clientX - current.startX) < TAB_DRAG_THRESHOLD_PX) return;
          if (!begin(current, event.currentTarget)) return;
          current.moved = true;
          dragged.current = true;
        }
        follow(current, event.clientX);
      },
      onPointerUp: (event: ReactPointerEvent<HTMLElement>) => {
        const current = drag.current;
        if (current?.moved && current.pointerId === event.pointerId) {
          follow(current, event.clientX);
          const { id, target, slot } = current;
          reset();
          if (target !== slot) useBrowserStore.getState().moveTab(id, target);
        } else {
          reset();
        }
      },
      onPointerCancel: reset,
      onLostPointerCapture: reset,
    }),
  };
}

function TabStrip({
  tabs,
  activeTabId,
  active,
}: { tabs: BrowserTab[]; activeTabId: string | null; active: boolean }) {
  const t = useT();
  const tabTitle = useTabTitle();
  const fullView = useBrowserStore((state) => state.fullView);
  const renamingTabId = useBrowserStore((state) => state.renamingTabId);
  const { activateTab, closeTab, newTab, closePanel, setFullView } =
    useBrowserStore.getState();
  const fullViewShortcut = useShortcutLabel("toggleBrowserFullView");
  useShortcut(
    "toggleBrowserFullView",
    (event) => {
      event.preventDefault();
      const state = useBrowserStore.getState();
      state.setFullView(!state.fullView);
    },
    { enabled: active },
  );
  const listRef = useRef<HTMLDivElement>(null);
  const tabDrag = useTabDrag(listRef, tabs.length);
  // Tabs shrink to fit first, so this only scrolls once they're at their narrowest.
  // biome-ignore lint/correctness/useExhaustiveDependencies: the active tab and tab count are triggers, read from the DOM
  useEffect(() => {
    listRef.current
      ?.querySelector('[role="tab"][aria-selected="true"]')
      ?.scrollIntoView({ block: "nearest", inline: "nearest" });
  }, [activeTabId, tabs.length]);
  return (
    // Above the desktop titlebar's drag strip (z-40), which would swallow tab clicks.
    <div
      data-tauri-drag-region={true}
      className="browser-chrome relative z-40 flex h-[var(--studio-chat-header-height,48px)] min-w-0 shrink-0 items-center gap-1 pl-1.5 pr-[calc(0.5rem*var(--ui-space-scale,1)+var(--studio-window-control-inset,0px))]"
    >
      <div
        ref={listRef}
        role="tablist"
        data-tauri-drag-region={true}
        aria-label={t("browser.tabs")}
        onWheel={(event) => {
          if (Math.abs(event.deltaY) > Math.abs(event.deltaX))
            event.currentTarget.scrollLeft += event.deltaY;
        }}
        className="browser-tablist flex min-w-0 flex-1 items-center gap-1 overflow-x-auto overflow-y-hidden"
      >
        {tabs.map((tab) => {
          const active = tab.id === activeTabId;
          const title = tabTitle(tab, currentEntry(tab));
          return (
            <TabContextMenu key={tab.id} tab={tab}>
              <div
                role="tab"
                aria-selected={active}
                tabIndex={0}
                title={title}
                data-tab-id={tab.id}
                {...tabDrag.handlers(tab.id)}
                onClick={() => {
                  if (!tabDrag.consumeClick()) activateTab(tab.id);
                }}
                onAuxClick={(event) => {
                  if (event.button === 1) closeTab(tab.id);
                }}
                onKeyDown={(event) => {
                  if (event.key === "Enter" || event.key === " ")
                    activateTab(tab.id);
                }}
                className={cn(
                  "group/tab relative flex h-[calc(34px*var(--ui-space-scale,1))] min-w-[calc(88px*var(--ui-space-scale,1))] max-w-60 flex-1 basis-0 cursor-pointer touch-none select-none items-center gap-1.5 rounded-[10px] pl-2 pr-1.5 text-ui-13 transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-ring",
                  active
                    ? "bg-card text-foreground dark:bg-accent"
                    : "text-muted-foreground hover:bg-[color-mix(in_oklab,var(--foreground)_calc(6%*var(--contrast-wash-gain,1)),transparent)] hover:text-foreground",
                )}
              >
                <TabIcon tab={tab} />
                {renamingTabId === tab.id ? (
                  <TabRenameInput tab={tab} title={title} />
                ) : (
                  <span className="min-w-0 flex-1 overflow-hidden whitespace-nowrap [mask-image:linear-gradient(to_right,black_calc(100%_-_0.625rem),transparent)]">
                    {title}
                  </span>
                )}
                {tab.muted ? (
                  <button
                    type="button"
                    aria-label={t("browser.tabMenu.unmute")}
                    title={t("browser.tabMenu.unmute")}
                    onClick={(event) => {
                      event.stopPropagation();
                      setTabMuted(tab, false);
                    }}
                    className="flex size-5 shrink-0 cursor-pointer items-center justify-center rounded-md text-muted-foreground hover:bg-[color-mix(in_oklab,var(--foreground)_calc(10%*var(--contrast-wash-gain,1)),transparent)] hover:text-foreground"
                  >
                    <HugeiconsIcon icon={VolumeMute02Icon} strokeWidth={1.75} className="size-3.5" />
                  </button>
                ) : null}
                <button
                  type="button"
                  aria-label={t("browser.closeTab")}
                  onClick={(event) => {
                    event.stopPropagation();
                    closeTab(tab.id);
                  }}
                  className={cn(
                    "flex size-5 shrink-0 cursor-pointer items-center justify-center rounded-md text-muted-foreground hover:bg-[color-mix(in_oklab,var(--foreground)_calc(10%*var(--contrast-wash-gain,1)),transparent)] hover:text-foreground",
                    !active &&
                      "hidden group-hover/tab:flex group-focus-visible/tab:flex focus-visible:flex",
                  )}
                >
                  <HugeiconsIcon
                    icon={Cancel01Icon}
                    strokeWidth={2}
                    className="size-3.5"
                  />
                </button>
              </div>
            </TabContextMenu>
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
        icon={Cancel01Icon}
        onClick={closePanel}
        className="size-8"
      />
    </div>
  );
}

let handledFocusSequence = 0;

function AddressBar({
  tab,
  leading,
  actions,
}: {
  tab: BrowserTab | undefined;
  /** Shown in place of the site button. */
  leading?: ReactNode;
  actions?: ReactNode;
}) {
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
      <div className={cn("flex h-9 items-center gap-0.5 rounded-lg pl-1 pr-1 transition-colors", URLBAR)}>
        {/* Icons step aside while typing and come back after. */}
        {editing ? null : (leading ?? <SiteIdentity address={address} tab={tab} />)}
        <div className="relative min-w-0 flex-1">
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
              "h-9 w-full min-w-0 bg-transparent pe-1 text-ui-13 text-foreground outline-none placeholder:text-muted-foreground",
              editing ? "ps-2" : "ps-0",
              // The input keeps the full URL, so focusing never changes its text or selection.
              !editing && address && "text-transparent",
            )}
          />
          {!editing && address ? (
            <span
              aria-hidden={true}
              className="pointer-events-none absolute inset-0 flex items-center ps-0 pe-1 text-ui-13 text-foreground"
            >
              <span className="truncate">{displayAddress(address, showFullUrl)}</span>
            </span>
          ) : null}
        </div>
        {editing ? null : actions}
      </div>
    </form>
  );
}

/** Whether `tab` shows its page as fetched: over https, Studio's fetch checks the certificate and
 *  host name, so a page that loaded had a valid one. A native view checks it itself. */
function pageVerified(tab: BrowserTab | undefined): boolean {
  if (!tab) return false;
  const entry = currentEntry(tab);
  if (entry.kind !== "web") return false;
  // A fetched page is cached only once its fetch succeeded, whether or not the frame has drawn it.
  return nativePage(tab) ? !tab.loading && !tab.nativeError : cachedPage(entry) !== undefined;
}

const LEARN_MORE_URL = "https://support.mozilla.org/kb/how-do-i-tell-if-my-connection-is-secure";

/** The shield at the bar's start: the site panel (connection, Security view, data, settings); a magnifier when there is no site. */
function SiteIdentity({ address, tab }: { address: string; tab: BrowserTab | undefined }) {
  const t = useT();
  const [clearOpen, setClearOpen] = useState(false);
  const [open, setOpen] = useState(false);
  const [view, setView] = useState<"site" | "security">("site");
  let url: URL | null = null;
  try {
    url = address ? new URL(address) : null;
  } catch {
    url = null;
  }
  if (!url || !/^https?:$/.test(url.protocol)) {
    return (
      <span className="flex h-7 w-[calc(26px*var(--ui-space-scale,1))] shrink-0 items-center justify-center text-muted-foreground">
        <HugeiconsIcon icon={Search01Icon} strokeWidth={1.75} aria-hidden={true} className="size-4" />
      </span>
    );
  }
  const secure = url.protocol === "https:";
  const verified = secure && pageVerified(tab);
  const label = t("browser.siteInfo.label");
  const row =
    "flex w-full cursor-pointer items-center gap-2.5 rounded-[11px] px-3 py-2 text-start text-sm transition-colors hover:bg-accent focus-visible:bg-accent focus-visible:outline-none";
  const iconButton =
    "flex size-7 shrink-0 cursor-pointer items-center justify-center rounded-md text-muted-foreground transition-colors hover:bg-accent hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring";
  const padlock = secure ? PadlockIcon : PadlockOpenIcon;
  return (
    <>
      <Popover
        open={open}
        onOpenChange={(next) => {
          setOpen(next);
          if (!next) setView("site");
        }}
      >
        <Tooltip>
          <TooltipTrigger asChild={true}>
            <PopoverTrigger asChild={true}>
              <button
                type="button"
                aria-label={label}
                className="flex h-7 w-[calc(26px*var(--ui-space-scale,1))] shrink-0 cursor-pointer items-center justify-center rounded-md text-muted-foreground transition-colors hover:bg-[color-mix(in_oklab,var(--foreground)_calc(8%*var(--contrast-wash-gain,1)),transparent)] hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring aria-expanded:bg-[color-mix(in_oklab,var(--foreground)_calc(10%*var(--contrast-wash-gain,1)),transparent)] aria-expanded:text-foreground"
              >
                {secure ? (
                  <ShieldCheck strokeWidth={2} className="size-4" />
                ) : (
                  <ShieldAlertGlyph strokeWidth={2} className="size-4" />
                )}
              </button>
            </PopoverTrigger>
          </TooltipTrigger>
          <TooltipContent side="top" className="tooltip-compact">
            {label}
          </TooltipContent>
        </Tooltip>
        <PopoverContent align="start" sideOffset={8} className="browser-menu w-80 gap-0 rounded-[14px] p-1.5">
          {view === "site" ? (
            <>
              <div className="flex min-w-0 items-center gap-2.5 px-3 py-2">
                <SiteFavicon
                  url={url.href}
                  className="size-4 rounded-[3px]"
                  fallbackClassName="size-4 text-muted-foreground"
                />
                <span className="min-w-0 flex-1 truncate text-sm font-medium text-foreground">{url.host}</span>
              </div>
              <button type="button" className={row} onClick={() => setView("security")}>
                <HugeiconsIcon icon={padlock} strokeWidth={1.75} className="size-icon shrink-0" />
                <span className="flex-1">
                  {t(secure ? "browser.siteInfo.secure" : "browser.siteInfo.insecure")}
                </span>
                <ChevronRight strokeWidth={2} className="size-4 shrink-0 text-muted-foreground rtl:-scale-x-100" />
              </button>
              <button
                type="button"
                className={row}
                onClick={() => {
                  setOpen(false);
                  setClearOpen(true);
                }}
              >
                <HugeiconsIcon icon={Delete02Icon} strokeWidth={1.75} className="size-icon" />
                {t("browser.menu.clearData")}
              </button>
              <button
                type="button"
                className={row}
                onClick={() => {
                  setOpen(false);
                  useSettingsDialogStore.getState().openDialog("browser");
                }}
              >
                <HugeiconsIcon icon={Settings02Icon} strokeWidth={1.75} className="size-icon" />
                {t("browser.settings")}
              </button>
            </>
          ) : (
            <>
              <div className="flex items-start gap-1.5 px-1.5 pb-2.5 pt-1.5">
                <button
                  type="button"
                  aria-label={t("browser.siteInfo.back")}
                  className={iconButton}
                  onClick={() => setView("site")}
                >
                  <ArrowLeft strokeWidth={2} className="size-4 rtl:-scale-x-100" />
                </button>
                <div className="min-w-0 flex-1 pt-0.5">
                  <p className="text-sm font-semibold text-foreground">{t("browser.siteInfo.security")}</p>
                  <p className="truncate text-ui-13 text-muted-foreground">{url.host}</p>
                </div>
                <button
                  type="button"
                  aria-label={t("browser.siteInfo.close")}
                  className={iconButton}
                  onClick={() => setOpen(false)}
                >
                  <XIcon strokeWidth={2} className="size-4" />
                </button>
              </div>
              <div className="mx-3 h-px bg-border" />
              <div className="flex gap-3 px-3 pb-2 pt-3">
                <HugeiconsIcon
                  icon={padlock}
                  strokeWidth={1.75}
                  className="size-icon shrink-0 text-muted-foreground"
                />
                <div className="min-w-0 flex-1">
                  <p className="text-sm font-medium text-foreground">
                    {t(secure ? "browser.siteInfo.secureTitle" : "browser.siteInfo.insecureTitle")}
                  </p>
                  <p className="mt-0.5 text-ui-13 leading-relaxed text-muted-foreground">
                    {t(secure ? "browser.siteInfo.secureDescription" : "browser.siteInfo.insecureDescription")}{" "}
                    <button
                      type="button"
                      className="cursor-pointer text-primary underline underline-offset-2 hover:opacity-80"
                      onClick={() => {
                        setOpen(false);
                        useBrowserStore.getState().openUrl(LEARN_MORE_URL, { newTab: true });
                      }}
                    >
                      {t("browser.siteInfo.learnMore")}
                    </button>
                  </p>
                </div>
              </div>
              {verified ? (
                <div className="flex items-center gap-3 px-3 pb-3 pt-1">
                  <HugeiconsIcon
                    icon={CertificateIcon}
                    strokeWidth={1.75}
                    className="size-icon shrink-0 text-muted-foreground"
                  />
                  <span className="text-sm text-foreground">{t("browser.siteInfo.certificateValid")}</span>
                </div>
              ) : null}
            </>
          )}
        </PopoverContent>
      </Popover>
      <ClearBrowsingDataDialog open={clearOpen} onOpenChange={setClearOpen} />
    </>
  );
}

function ZoomBadge({ tab }: { tab: BrowserTab | undefined }) {
  const t = useT();
  const locale = useLocale();
  const preferred = useBrowserPrefsStore((state) => state.defaultZoom);
  if (!canZoom(tab)) return null;
  const zoom = tab.zoom;
  const resetZoom = homeZoom(tab, preferred);
  if (Math.abs(zoom - resetZoom) < 0.001) return null;
  const label = t("browser.menu.zoomReset");
  return (
    <Tooltip>
      <TooltipTrigger asChild={true}>
        <button
          type="button"
          aria-label={label}
          onClick={() => useBrowserStore.getState().setZoom(tab.id, resetZoom)}
          className="h-6 shrink-0 cursor-pointer rounded-full bg-[color-mix(in_oklab,var(--foreground)_calc(8%*var(--contrast-wash-gain,1)),transparent)] px-2 text-ui-12 tabular-nums text-foreground transition-colors hover:bg-[color-mix(in_oklab,var(--foreground)_calc(12%*var(--contrast-wash-gain,1)),transparent)]"
        >
          {new Intl.NumberFormat(locale, { style: "percent", maximumFractionDigits: 0 }).format(zoom)}
        </button>
      </TooltipTrigger>
      <TooltipContent side="top" className="tooltip-compact">
        {label}
      </TooltipContent>
    </Tooltip>
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
        temporary: entry.kind === "web" && entry.temporary === true,
      }
    : undefined;
}

function webAddress(tab: BrowserTab | undefined): string | null {
  const entry = tab ? currentEntry(tab) : null;
  return entry?.kind === "web" ? (tab?.displayUrl ?? entry.url) : null;
}

function WebActions({ tab }: { tab: BrowserTab | undefined }) {
  const t = useT();
  const download = tabDownload(tab);
  return (
    <>
      <AnnotatePageButton tab={tab} />
      <IconButton
        label={t("browser.download")}
        disabled={!download}
        onClick={() => download && void saveBrowserDownload(download)}
        className={NAV_BUTTON}
      >
        <HugeiconsIcon icon={Download01Icon} strokeWidth={1.75} className="size-4.5" />
      </IconButton>
    </>
  );
}

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
  const zoomable = canZoom(tab);
  const preferred = useBrowserPrefsStore((state) => state.defaultZoom);
  const resetZoom = zoomable ? homeZoom(tab, preferred) : preferred;
  const zoom = zoomable ? tab.zoom : resetZoom;
  const setZoom = (next: number) =>
    zoomable && useBrowserStore.getState().setZoom(tab.id, next);
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
          disabled={!zoomable || zoom <= (ZOOM_STEPS[0] ?? 0)}
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
          disabled={!zoomable || zoom >= (ZOOM_STEPS[ZOOM_STEPS.length - 1] ?? 5)}
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
        disabled={!zoomable || zoom === resetZoom}
        onClick={() => setZoom(resetZoom)}
        className={cn(step, "rounded-md")}
      >
        <RefreshGlyph strokeWidth={1.75} className="size-3.5" />
      </button>
    </div>
  );
}

async function takeScreenshot(tab: BrowserTab, page: HTMLElement, t: ReturnType<typeof useT>): Promise<void> {
  const temporary = useChatRuntimeStore.getState().incognito;
  let blob: Blob | null;
  try {
    blob = await screenshotPage(tab, page);
  } catch (error) {
    // Declining the browser's prompt is an answer, not a failure.
    if (error instanceof DOMException && error.name === "NotAllowedError") return;
    toast.error(t(error instanceof OtherSurfaceError ? "browser.screenshot.otherSurface" : "browser.screenshot.failed"));
    return;
  }
  if (!blob) {
    toast.error(t("browser.screenshot.failed"));
    return;
  }
  const entry = currentEntry(tab);
  const now = new Date();
  const pad = (value: number) => String(value).padStart(2, "0");
  const stamp = `${now.getFullYear()}-${pad(now.getMonth() + 1)}-${pad(now.getDate())} ${pad(now.getHours())}.${pad(now.getMinutes())}.${pad(now.getSeconds())}`;
  const name = `Screenshot ${entry.kind === "web" ? hostOf(entry.url) : tab.title || "page"} ${stamp}.png`.replace(/[\\/:*?"<>|]+/g, "-");
  const download: BrowserDownload = { blob, name, contentType: "image/png", url: null, temporary };
  const attach = useBrowserStore.getState().attachToChat;
  if (attach && (await attach(new File([blob], name, { type: "image/png" })))) {
    toast.success(t("browser.screenshot.added"), {
      action: { label: t("browser.screenshot.save"), onClick: () => void saveBrowserDownload(download) },
    });
    return;
  }
  // The share prompt outlasts the click, so the save dialog needs a fresh one.
  if (saveNeedsClick()) {
    toast.success(t("browser.screenshot.taken"), {
      action: { label: t("browser.screenshot.save"), onClick: () => void saveBrowserDownload(download) },
    });
    return;
  }
  await saveBrowserDownload(download);
}

function PanelMenu({ tab, children }: { tab: BrowserTab | undefined; children?: ReactNode }) {
  const t = useT();
  // Files list their own items first and skip page-only ones.
  const fileTab = tab !== undefined && currentEntry(tab).kind === "file";
  const device = useBrowserStore((state) => state.device);
  const [clearOpen, setClearOpen] = useState(false);
  const webUrl = webAddress(tab);
  const bookmarked = useBookmarkFor(webUrl) !== undefined;
  const toolbarMode = useBrowserPrefsStore((state) => state.bookmarksToolbar);
  // The star's editor opens as the menu closes; focus going back to the menu button would shut it.
  const keepFocus = useRef(false);
  const webPage = showsWebPage(tab);
  // A native view prints through its engine; a framed page from a copy, where frames can print.
  const printable = webPage && (nativePage(tab) || canPrintFrames());
  const triggerRef = useRef<HTMLButtonElement>(null);
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
                ref={triggerRef}
                type="button"
                aria-label={t("browser.more")}
                className="flex size-8 shrink-0 cursor-pointer items-center justify-center rounded-md text-foreground transition-colors hover:bg-[color-mix(in_oklab,var(--foreground)_calc(6%*var(--contrast-wash-gain,1)),transparent)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring aria-expanded:bg-[color-mix(in_oklab,var(--foreground)_calc(8%*var(--contrast-wash-gain,1)),transparent)]"
              >
                <HugeiconsIcon icon={MoreHorizontalIcon} strokeWidth={1.75} className="size-5" />
              </button>
            </DropdownMenuTrigger>
          </TooltipTrigger>
          <TooltipContent side="top" className="tooltip-compact">
            {t("browser.more")}
          </TooltipContent>
        </Tooltip>
        <DropdownMenuContent
          align="end"
          sideOffset={6}
          className="browser-menu min-w-72 rounded-[20px] p-1.5 [&_[data-slot=dropdown-menu-separator]]:mx-3 [&_[data-slot=dropdown-menu-separator]]:my-1.5"
          onCloseAutoFocus={(event) => {
            if (!keepFocus.current) return;
            keepFocus.current = false;
            event.preventDefault();
          }}
        >
          {children ? (
            <>
              {children}
              <DropdownMenuSeparator />
            </>
          ) : null}
          {fileTab ? null : (
          <>
          <DropdownMenuItem
            disabled={!webPage}
            onSelect={() => {
              // The find bar takes focus; the menu button mustn't take it back.
              keepFocus.current = true;
              requestFind(BROWSER_FIND_TARGET);
            }}
          >
            {t("browser.menu.find")}
            <DropdownMenuShortcut>{mod}F</DropdownMenuShortcut>
          </DropdownMenuItem>
          <DropdownMenuItem
            disabled={!printable}
            onSelect={() =>
              tab &&
              void printPage(tab).then(
                (printed) => printed || toast.error(t("browser.menu.printFailed")),
              )
            }
          >
            {t("browser.menu.print")}
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
          <DropdownMenuItem disabled={!webUrl} onSelect={() => webUrl && openExternalLink(webUrl)}>
            {t("browser.openExternal")}
          </DropdownMenuItem>
          <DropdownMenuSeparator />
          </>
          )}
          <ZoomControl tab={tab} />
          <DropdownMenuSeparator />
          {fileTab ? null : (
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
          )}
          {canScreenshot() ? (
            <DropdownMenuItem
              disabled={!tab}
              onSelect={() => {
                // The innermost page box: with the device toolbar, just the device's width.
                const pages = triggerRef.current?.closest("section")?.querySelectorAll<HTMLElement>("[data-browser-page]");
                const page = pages?.[pages.length - 1];
                if (tab && page) void takeScreenshot(tab, page, t);
              }}
            >
              {t("browser.menu.screenshot")}
            </DropdownMenuItem>
          ) : null}
          <DropdownMenuSeparator />
          <DropdownMenuSub>
            <DropdownMenuSubTrigger>{t("browser.pages.bookmarks")}</DropdownMenuSubTrigger>
            <DropdownMenuSubContent className="browser-menu min-w-60 rounded-[20px] p-1.5 [&_[data-slot=dropdown-menu-separator]]:mx-3 [&_[data-slot=dropdown-menu-separator]]:my-1.5">
              <DropdownMenuItem
                disabled={!webUrl}
                onSelect={() => {
                  keepFocus.current = true;
                  store.bookmarkPage();
                }}
              >
                {t(bookmarked ? "browser.bookmarks.edit" : "browser.bookmarks.bookmarkPage")}
                <DropdownMenuShortcut>{mod}D</DropdownMenuShortcut>
              </DropdownMenuItem>
              <DropdownMenuSeparator />
              <DropdownMenuLabel className="text-ui-12 font-normal text-muted-foreground">
                {t("browser.bookmarks.toolbar")}
              </DropdownMenuLabel>
              <DropdownMenuRadioGroup
                value={toolbarMode}
                onValueChange={(value) =>
                  useBrowserPrefsStore.getState().setBookmarksToolbar(value as BookmarksToolbarMode)
                }
              >
                <DropdownMenuRadioItem value="always">{t("browser.bookmarks.toolbarAlways")}</DropdownMenuRadioItem>
                <DropdownMenuRadioItem value="newtab">{t("browser.bookmarks.toolbarNewTab")}</DropdownMenuRadioItem>
                <DropdownMenuRadioItem value="never">{t("browser.bookmarks.toolbarNever")}</DropdownMenuRadioItem>
              </DropdownMenuRadioGroup>
              <DropdownMenuSeparator />
              <DropdownMenuItem onSelect={() => store.openInternal("bookmarks")}>
                {t("browser.bookmarks.manage")}
              </DropdownMenuItem>
            </DropdownMenuSubContent>
          </DropdownMenuSub>
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

function nativePage(tab: BrowserTab | undefined): boolean {
  return Boolean(tab && currentEntry(tab).kind === "web" && hasNativeView(tab.id));
}

/** How a tab is annotated: a framed page, a native view, or Studio's own markup (new tab,
 *  internal pages, documents, errors). Null while loading. Files use Request edits. */
function annotateMode(tab: BrowserTab | undefined, native: boolean): "frame" | "native" | "dom" | null {
  if (!tab) return null;
  const entry = currentEntry(tab);
  if (entry.kind === "newtab" || entry.kind === "internal") return "dom";
  if (entry.kind !== "web" || tab.loading) return null;
  if (native) return tab.nativeError ? "dom" : "native";
  return tab.documentType || tab.pageError ? "dom" : "frame";
}

/** Ask about the page, as with a file's Request edits. */
function AnnotatePageButton({ tab }: { tab: BrowserTab | undefined }) {
  const t = useT();
  const canAnnotate = useBrowserStore((state) => state.sendAnnotations !== null);
  const annotating = useBrowserStore((state) => tab !== undefined && state.annotateTabId === tab.id);
  const native = useNativeBrowser((state) => state.enabled);
  if (!canAnnotate) return null;
  return (
    <IconButton
      label={t("browser.annotate.page")}
      disabled={annotateMode(tab, native) === null}
      onClick={() => tab && useBrowserStore.getState().setAnnotating(annotating ? null : tab.id)}
      className={cn(
        ANNOTATE_BUTTON,
        annotating &&
          "bg-primary/12 text-primary hover:bg-primary/18 hover:text-primary dark:bg-primary/20 dark:hover:bg-primary/25",
      )}
    >
      <HugeiconsIcon icon={CursorRectangleSelection02Icon} strokeWidth={1.75} className={ANNOTATE_GLYPH} />
    </IconButton>
  );
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
      <div className="flex shrink-0 items-center gap-0.5">
        <IconButton label={t("browser.back")} disabled={!canGoBack} onClick={back} className={NAV_BUTTON}>
          <ArrowLeft strokeWidth={NAV_STROKE} className={NAV_ICON} />
        </IconButton>
        <IconButton
          label={t("browser.forward")}
          disabled={!canGoForward}
          onClick={forward}
          className={NAV_BUTTON}
        >
          <ArrowRight strokeWidth={NAV_STROKE} className={NAV_ICON} />
        </IconButton>
        <IconButton
          label={t("browser.reload")}
          disabled={!tab || currentEntry(tab).kind !== "web"}
          onClick={() => {
            if (!tab) return;
            if (native) nativeAction(tab.id, "reload");
            else reload(tab.id);
          }}
          className={NAV_BUTTON}
        >
          <RotateCw strokeWidth={NAV_STROKE} className={cn(NAV_ICON, "scale-[0.94]")} />
        </IconButton>
      </div>
      <AddressBar
        key={tab?.id ?? "none"}
        tab={tab}
        actions={
          <>
            <ZoomBadge tab={tab} />
            <BookmarkStar url={webAddress(tab)} title={tab?.title ?? ""} favicon={tab?.favicon ?? null} />
          </>
        }
      />
      <div className="flex shrink-0 items-center gap-0.5">
        <WebActions tab={tab} />
        <PanelMenu tab={tab} />
      </div>
    </>
  );
}

/** HTML and code use the browser chrome; other files keep the floating controls. */
function usesBrowserChrome(entry: Extract<BrowserEntry, { kind: "file" }>): boolean {
  const kind = textFileKind(entry.name, entry.contentType, entry.plainText);
  return kind === "html" || kind === "code" || isVideoEntry(entry);
}

function isVideoEntry(entry: Extract<BrowserEntry, { kind: "file" }>): boolean {
  return !entry.plainText && mediaKind(entry.name, entry.contentType) === "video";
}

function BrowserFileToolbar({
  tab,
  entry,
}: { tab: BrowserTab; entry: Extract<BrowserEntry, { kind: "file" }> }) {
  const t = useT();
  const navigate = useNavigate();
  const requestEdits = useBrowserStore((state) => state.requestEdits);
  const canAnnotate = useBrowserStore((state) => state.sendAnnotations !== null);
  const annotating = useBrowserStore((state) => state.annotateTabId === tab.id);
  const view = useBrowserStore((state) => state.fileViews[tab.id]) ?? DEFAULT_FILE_VIEW;
  const [copied, setCopied] = useState(false);
  const { goBack, goForward } = useBrowserStore.getState();
  const download = tabDownload(tab);
  const blob = download?.blob;
  const kind = textFileKind(entry.name, entry.contentType, entry.plainText);
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
    if (kind === "html") setView({ mode: "preview" });
    useBrowserStore.getState().reload(tab.id);
  };
  const toggleConsole = () =>
    htmlPreview
      ? setView({ consoleOpen: !view.consoleOpen })
      : setView({ mode: "preview", consoleOpen: true });
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
  const consoleLabel = t("browser.file.console");
  const errorBadge =
    view.errorCount > 0 ? (
      <span className="rounded-full bg-destructive px-1.5 text-ui-10 font-medium leading-4 text-destructive-foreground">
        {view.errorCount}
      </span>
    ) : null;
  return (
    <>
      <div className="flex shrink-0 items-center gap-0.5">
        <IconButton
          label={t("browser.back")}
          disabled={tab.index === 0}
          onClick={() => goBack(tab.id)}
          className={NAV_BUTTON}
        >
          <ArrowLeft strokeWidth={NAV_STROKE} className={NAV_ICON} />
        </IconButton>
        <IconButton
          label={t("browser.forward")}
          disabled={tab.index >= tab.history.length - 1}
          onClick={() => goForward(tab.id)}
          className={NAV_BUTTON}
        >
          <ArrowRight strokeWidth={NAV_STROKE} className={NAV_ICON} />
        </IconButton>
        <IconButton
          label={t(kind === "html" ? "browser.file.runAgain" : "browser.reload")}
          onClick={runAgain}
          className={NAV_BUTTON}
        >
          <RotateCw strokeWidth={NAV_STROKE} className={cn(NAV_ICON, "scale-[0.94]")} />
        </IconButton>
      </div>
      <AddressBar
        key={tab.id}
        tab={tab}
        leading={<span aria-hidden={true} className="w-1.5 shrink-0" />}
        actions={
          <>
            <ZoomBadge tab={tab} />
            {hasSource ? (
              <div role="tablist" aria-label={t("browser.file.viewMode")} className="flex shrink-0 items-center gap-0.5">
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
                          "flex size-7 cursor-pointer items-center justify-center rounded-md text-muted-foreground transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
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
          </>
        }
      />
      <div className="flex shrink-0 items-center gap-0.5">
        <IconButton
          label={t("browser.file.requestEdits")}
          // Without a chat to send marks to, stages a prompt naming the file.
          onClick={() =>
            canAnnotate
              ? toggleAnnotating()
              : (requestEdits ?? stageEditsPrompt)(t("browser.file.requestEditsPrompt", { name: entry.name }))
          }
          className={cn(
            ANNOTATE_BUTTON,
            annotating &&
              "bg-primary/12 text-primary hover:bg-primary/18 hover:text-primary dark:bg-primary/20 dark:hover:bg-primary/25",
          )}
        >
          <HugeiconsIcon
            icon={CursorRectangleSelection02Icon}
            strokeWidth={1.75}
            className={ANNOTATE_GLYPH}
          />
        </IconButton>
        {kind === "html" ? (
          <Tooltip>
            <TooltipTrigger asChild={true}>
              <button
                type="button"
                aria-label={consoleLabel}
                aria-pressed={htmlPreview && view.consoleOpen}
                onClick={toggleConsole}
                className={cn(
                  "flex h-8 min-w-8 shrink-0 cursor-pointer items-center justify-center gap-1 rounded-md px-1.5 text-foreground transition-colors hover:bg-[color-mix(in_oklab,var(--foreground)_calc(6%*var(--contrast-wash-gain,1)),transparent)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
                  htmlPreview &&
                    view.consoleOpen &&
                    "bg-[color-mix(in_oklab,var(--foreground)_calc(8%*var(--contrast-wash-gain,1)),transparent)]",
                )}
              >
                <HugeiconsIcon icon={ComputerTerminal01Icon} strokeWidth={1.75} className="size-4.5" />
                {errorBadge}
              </button>
            </TooltipTrigger>
            <TooltipContent side="top" className="tooltip-compact">
              {consoleLabel}
            </TooltipContent>
          </Tooltip>
        ) : null}
        {kind ? (
          <IconButton
            label={copied ? t("browser.file.copied") : t("browser.file.copy")}
            disabled={!blob}
            onClick={copyContents}
            className={cn(NAV_BUTTON, "hidden @[34rem]:flex")}
          >
            <HugeiconsIcon icon={copied ? Tick02Icon : Copy01Icon} strokeWidth={1.75} className="size-4.5" />
          </IconButton>
        ) : null}
        <IconButton
          label={t("browser.download")}
          disabled={!download}
          onClick={() => download && void saveBrowserDownload(download)}
          className={NAV_BUTTON}
        >
          <HugeiconsIcon icon={Download01Icon} strokeWidth={1.75} className="size-4.5" />
        </IconButton>
        <PanelMenu tab={tab}>
          <div className="flex items-start gap-3 px-3 py-2 text-sm">
            <KindIcon name={entry.name} contentType={entry.contentType} className="mt-0.5 size-4.5" mono={true} />
            <span className="min-w-0 break-words">{entry.name}</span>
          </div>
          <DropdownMenuSeparator />
          <DropdownMenuSub>
            <DropdownMenuSubTrigger disabled={!blob}>
              <HugeiconsIcon icon={ArrowUpRight01Icon} strokeWidth={1.75} className="size-4" />
              {t("browser.file.openIn")}
            </DropdownMenuSubTrigger>
            <DropdownMenuSubContent className="browser-menu min-w-52 rounded-[20px] p-1.5">
              <DropdownMenuItem onSelect={openInNewChat}>
                <HugeiconsIcon icon={BubbleChatAddIcon} strokeWidth={1.75} className="size-4.5" />
                {t("browser.file.newChat")}
              </DropdownMenuItem>
              {/* A blob URL can't be handed to another app from the desktop app. */}
              {isTauri || !tabType ? null : (
                <DropdownMenuItem onSelect={openInBrowser}>
                  <HugeiconsIcon icon={InternetIcon} strokeWidth={1.75} className="size-4.5" />
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
                <HugeiconsIcon icon={ComputerTerminal01Icon} strokeWidth={1.75} className="size-4" />
                {consoleLabel}
                {errorBadge ? <span className="ml-auto">{errorBadge}</span> : null}
              </DropdownMenuItem>
            </>
          ) : null}
          {kind ? (
            <DropdownMenuItem disabled={!blob} onSelect={copyContents}>
              <HugeiconsIcon icon={Copy01Icon} strokeWidth={1.75} className="size-4" />
              {t("browser.file.copy")}
            </DropdownMenuItem>
          ) : null}
          {showsSource ? (
            <DropdownMenuCheckboxItem
              checked={view.wrap || kind === "text"}
              disabled={kind === "text"}
              onCheckedChange={(wrap) => setView({ wrap })}
              onSelect={(event) => event.preventDefault()}
            >
              <HugeiconsIcon icon={TextWrapIcon} strokeWidth={1.75} className="size-4" />
              {t("browser.file.wrap")}
            </DropdownMenuCheckboxItem>
          ) : null}
        </PanelMenu>
      </div>
    </>
  );
}

function FloatingFileToolbar({
  tab,
  entry,
}: { tab: BrowserTab; entry: Extract<BrowserEntry, { kind: "file" }> }) {
  const t = useT();
  const navigate = useNavigate();
  const requestEdits = useBrowserStore((state) => state.requestEdits);
  const canAnnotate = useBrowserStore((state) => state.sendAnnotations !== null);
  const annotating = useBrowserStore((state) => state.annotateTabId === tab.id);
  const view = useBrowserStore((state) => state.fileViews[tab.id]) ?? DEFAULT_FILE_VIEW;
  const [copied, setCopied] = useState(false);
  const download = tabDownload(tab);
  const blob = download?.blob;
  const kind = textFileKind(entry.name, entry.contentType, entry.plainText);
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
              "flex h-9 min-w-24 max-w-[55%] shrink cursor-pointer items-center gap-2 rounded-full pl-3.5 pr-2.5 text-ui-13p5 text-foreground outline-none transition-colors focus-visible:ring-2 focus-visible:ring-ring",
            )}
          >
            <KindIcon
              name={entry.name}
              contentType={entry.contentType}
              className="size-4.5"
              mono={true}
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
          className="browser-menu w-80 max-w-[calc(100vw-2rem)] rounded-[20px] p-1.5 [&_[data-slot=dropdown-menu-separator]]:mx-3 [&_[data-slot=dropdown-menu-separator]]:my-1.5"
        >
          <div className="flex items-start gap-3 px-3 py-2 text-sm">
            <KindIcon
              name={entry.name}
              contentType={entry.contentType}
              className="mt-0.5 size-4.5"
              mono={true}
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
            <DropdownMenuSubContent className="browser-menu min-w-52 rounded-[20px] p-1.5">
              <DropdownMenuItem onSelect={openInNewChat}>
                <HugeiconsIcon
                  icon={BubbleChatAddIcon}
                  strokeWidth={1.75}
                  className="size-4.5"
                />
                {t("browser.file.newChat")}
              </DropdownMenuItem>
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
      {/* A true circle, glyph centred; the label is its tooltip. */}
      <IconButton
        label={t("browser.file.requestEdits")}
        // Without a chat to send marks to, stages a prompt naming the file.
        onClick={() =>
          canAnnotate
            ? toggleAnnotating()
            : (requestEdits ?? stageEditsPrompt)(
                t("browser.file.requestEditsPrompt", { name: entry.name }),
              )
        }
        className={cn(
          PILL,
          "size-9 rounded-full p-0 text-foreground",
          annotating && "text-primary hover:text-primary",
        )}
      >
        <HugeiconsIcon
          icon={CursorRectangleSelection02Icon}
          strokeWidth={1.75}
          className={ANNOTATE_GLYPH}
        />
      </IconButton>
      <span
        aria-hidden={true}
        className="min-w-0 flex-1 pointer-events-none!"
      />
      {hasSource ? (
        <div
          role="tablist"
          aria-label={t("browser.file.viewMode")}
          className={cn(PILL_SURFACE, "flex h-9 shrink-0 items-center gap-0.5 rounded-full p-0.5")}
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
        contentClassName="browser-menu"
        onChange={(value) =>
          useBrowserStore
            .getState()
            .setZoom(tab.id, value === "fit" ? 1 : value)
        }
        className={cn(PILL, "mr-0 hidden h-9 pr-2.5 @[28rem]:flex")}
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

const SPLIT_PILL = cn(PILL, "flex h-9 shrink-0 items-center rounded-full");
const SPLIT_PART =
  "flex h-full cursor-pointer items-center rounded-full text-ui-13p5 text-foreground outline-none transition-colors hover:bg-[color-mix(in_oklab,var(--foreground)_6%,transparent)] focus-visible:ring-2 focus-visible:ring-ring disabled:pointer-events-none disabled:opacity-50 data-[state=open]:bg-[color-mix(in_oklab,var(--foreground)_8%,transparent)]";

function SplitChevron() {
  return (
    <HugeiconsIcon
      icon={ChevronDownStandardIcon}
      strokeWidth={1.75}
      className="size-4 shrink-0 text-muted-foreground"
    />
  );
}

/** Video file bar: the name, then Copy and Open, each with its options. */
function VideoFileToolbar({
  tab,
  entry,
}: { tab: BrowserTab; entry: Extract<BrowserEntry, { kind: "file" }> }) {
  const t = useT();
  const navigate = useNavigate();
  const download = tabDownload(tab);
  const blob = download?.blob;
  const { goBack, goForward } = useBrowserStore.getState();
  // A blob URL can't be handed to another app from the desktop app.
  const tabType = isTauri ? null : browserTabType(entry.name, entry.contentType || blob?.type || "");
  const canOpenTab = tabType !== null;
  const frameCopies = canCopyVideoFrame();
  const openInBrowser = () => {
    if (!blob || !tabType) return;
    const url = URL.createObjectURL(new Blob([blob], { type: tabType }));
    window.open(url, "_blank", "noopener,noreferrer");
    window.setTimeout(() => URL.revokeObjectURL(url), 60_000);
  };
  const openInNewChat = () => {
    if (!blob) return;
    startLibraryChat(navigate, {
      files: [new File([blob], entry.name, { type: entry.contentType || blob.type })],
    });
  };
  const save = () => download && void saveBrowserDownload(download);
  const copyFrame = () => {
    const video = tabVideo(tab.id);
    if (!video) return;
    void copyVideoFrame(video).then((ok) =>
      ok ? toast.success(t("browser.video.frameCopied")) : toast.error(t("browser.video.copyFrameFailed")),
    );
  };
  const copyName = () =>
    void copyToClipboard(entry.name).then((ok) => ok && toast.success(t("browser.video.nameCopied")));
  const extension = /\.([a-z0-9]+)$/i.exec(entry.name)?.[1]?.toUpperCase();
  const meta = [extension, blob ? formatBytes(blob.size) : null].filter(Boolean).join(" · ");
  return (
    <>
      {/* As the panel narrows, back/forward and then Copy hide before the name does. */}
      <div className="hidden shrink-0 items-center gap-0.5 @[34rem]:flex">
        <IconButton
          label={t("browser.back")}
          disabled={tab.index === 0}
          onClick={() => goBack(tab.id)}
          className={NAV_BUTTON}
        >
          <ArrowLeft strokeWidth={NAV_STROKE} className={NAV_ICON} />
        </IconButton>
        <IconButton
          label={t("browser.forward")}
          disabled={tab.index >= tab.history.length - 1}
          onClick={() => goForward(tab.id)}
          className={NAV_BUTTON}
        >
          <ArrowRight strokeWidth={NAV_STROKE} className={NAV_ICON} />
        </IconButton>
      </div>
      <div
        title={entry.name}
        className={cn(PILL_SURFACE, "flex h-9 min-w-24 flex-1 items-center gap-2 rounded-full px-3.5")}
      >
        <KindIcon name={entry.name} contentType={entry.contentType} className="size-4.5" />
        <span className="min-w-0 truncate text-ui-13p5 text-foreground">{entry.name}</span>
        {meta ? (
          <span className="hidden shrink-0 text-ui-12 text-muted-foreground @[30rem]:inline">{meta}</span>
        ) : null}
      </div>
      {/* Without Copy frame (desktop app) the name is all there is to copy. */}
      <div className={cn(SPLIT_PILL, "hidden @[26rem]:flex")}>
        <Tooltip>
          <TooltipTrigger asChild={true}>
            <button
              type="button"
              aria-label={frameCopies ? t("browser.video.copyFrame") : t("browser.video.copyName")}
              onClick={frameCopies ? copyFrame : copyName}
              className={cn(SPLIT_PART, frameCopies ? "pl-2.5 pr-1" : "px-2.5")}
            >
              <HugeiconsIcon icon={Copy01Icon} strokeWidth={1.75} className="size-4.5" />
            </button>
          </TooltipTrigger>
          <TooltipContent side="bottom" className="tooltip-compact">
            {frameCopies ? t("browser.video.copyFrame") : t("browser.video.copyName")}
          </TooltipContent>
        </Tooltip>
        {frameCopies ? (
          <DropdownMenu>
            <DropdownMenuTrigger asChild={true}>
              <button type="button" aria-label={t("browser.video.copyOptions")} className={cn(SPLIT_PART, "pr-2 pl-1")}>
                <SplitChevron />
              </button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end" sideOffset={6} className="browser-menu min-w-52 rounded-[20px] p-1.5">
              <DropdownMenuItem onSelect={copyFrame}>
                <HugeiconsIcon icon={Copy01Icon} strokeWidth={1.75} className="size-4" />
                {t("browser.video.copyFrame")}
              </DropdownMenuItem>
              <DropdownMenuItem onSelect={copyName}>
                <HugeiconsIcon icon={TextWrapIcon} strokeWidth={1.75} className="size-4" />
                {t("browser.video.copyName")}
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        ) : null}
      </div>
      <div className={SPLIT_PILL}>
        <button
          type="button"
          disabled={!blob}
          onClick={canOpenTab ? openInBrowser : save}
          title={canOpenTab ? t("browser.file.newBrowserTab") : t("browser.video.saveAs")}
          className={cn(SPLIT_PART, "gap-1.5 pr-1.5 pl-3")}
        >
          <HugeiconsIcon
            icon={canOpenTab ? ArrowUpRight01Icon : Download01Icon}
            strokeWidth={1.75}
            className="size-4.5"
          />
          {canOpenTab ? t("browser.video.open") : t("browser.video.saveAs")}
        </button>
        <DropdownMenu>
          <DropdownMenuTrigger asChild={true}>
            <button
              type="button"
              disabled={!blob}
              aria-label={t("browser.video.openOptions")}
              className={cn(SPLIT_PART, "pr-2.5 pl-1")}
            >
              <SplitChevron />
            </button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end" sideOffset={6} className="browser-menu min-w-56 rounded-[20px] p-1.5">
            {canOpenTab ? (
              <DropdownMenuItem onSelect={openInBrowser}>
                <HugeiconsIcon icon={InternetIcon} strokeWidth={1.75} className="size-4.5" />
                {t("browser.file.newBrowserTab")}
              </DropdownMenuItem>
            ) : null}
            <DropdownMenuItem onSelect={openInNewChat}>
              <HugeiconsIcon icon={BubbleChatAddIcon} strokeWidth={1.75} className="size-4.5" />
              {t("browser.file.newChat")}
            </DropdownMenuItem>
            <DropdownMenuSeparator />
            <DropdownMenuItem onSelect={save}>
              <HugeiconsIcon icon={Download01Icon} strokeWidth={1.75} className="size-4.5" />
              {t("browser.video.saveAs")}
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </div>
      <PanelMenu tab={tab} />
    </>
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

const MAX_MOUNTED_TABS = 4;
// Hidden web pages run scripts on the chat's thread, so only the newest stays live.
const MAX_HIDDEN_WEB_PAGES = 1;

function liveTabIds(mounted: readonly string[], tabs: BrowserTab[], activeTabId: string | null): Set<string> {
  const byId = new Map(tabs.map((tab) => [tab.id, tab]));
  const live = new Set<string>();
  let hiddenWeb = 0;
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

/** The chat's in-app browser, memoized so chat renders skip it. `active`: its chat is shown. */
export const BrowserPanel = memo(function BrowserPanel({ active = true }: { active?: boolean }) {
  const t = useT();
  const tabs = useBrowserStore((state) => state.tabs);
  const activeTabId = useBrowserStore((state) => state.activeTabId);
  const device = useBrowserStore((state) => state.device);
  const annotateTabId = useBrowserStore((state) => state.annotateTabId);
  const tabTitle = useTabTitle();
  const [pageElement, setPageElement] = useState<HTMLDivElement | null>(null);
  const sectionRef = useRef<HTMLElement>(null);
  // Zoom keys, Ctrl+wheel and the View menu zoom the page, not the interface, while focus or the pointer is here.
  useEffect(() => {
    if (!active) return;
    return registerZoomScope({
      contains: (element) => sectionRef.current?.contains(element) ?? false,
      zoom: (direction) => {
        const tabId = useBrowserStore.getState().activeTabId;
        if (tabId) zoomTab(tabId, direction);
      },
    });
  }, [active]);
  // Studio's find bar searches the page while focus is in here, or when switched to it.
  useEffect(() => {
    if (!active) return;
    return registerBrowserFind((node) => sectionRef.current?.contains(node) ?? false);
  }, [active]);
  const native = useNativeBrowser((state) => state.enabled);
  useEffect(() => (native ? startNativeViews() : undefined), [native]);
  const activeTab = tabs.find((tab) => tab.id === activeTabId);
  const activeEntry = activeTab ? currentEntry(activeTab) : null;
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
  const fileTab = activeEntry?.kind === "file";
  const floatingFileControls = activeEntry?.kind === "file" && !usesBrowserChrome(activeEntry);
  const documentShown = fileTab || Boolean(activeTab?.documentType);
  const deviceWidth =
    device !== "off" && activeEntry?.kind === "web"
      ? DEVICE_WIDTHS[device]
      : null;
  const pageAnnotating = activeTab && annotateTabId === activeTab.id ? annotateMode(activeTab, native) : null;

  return (
    <section
      ref={sectionRef}
      aria-label={t("browser.title")}
      // The chat's find skips the browser's chrome; the page is searched as its own target.
      {...{ [FIND_SKIP_ATTRIBUTE]: "" }}
      onKeyDown={(event) => {
        if (
          (event.metaKey || event.ctrlKey) &&
          !event.shiftKey &&
          !event.altKey &&
          event.key.toLowerCase() === "d"
        ) {
          event.preventDefault();
          useBrowserStore.getState().bookmarkPage();
        }
      }}
      className={cn(
        "relative flex h-full min-h-0 flex-col overflow-hidden bg-muted pt-[var(--studio-content-top-inset,0px)]",
        documentShown
          ? "[--browser-surface:color-mix(in_oklab,var(--muted)_55%,var(--card))]"
          : "[--browser-surface:var(--card)]",
      )}
    >
      <TabStrip tabs={tabs} activeTabId={activeTabId} active={active} />
      <div className="relative flex min-h-0 flex-1 flex-col overflow-hidden bg-[var(--browser-surface)]">
        <div
          className={cn(
            "browser-chrome @container flex shrink-0 items-center gap-2 px-2.5",
            floatingFileControls
              ? "browser-file-toolbar pointer-events-none absolute inset-x-0 top-0 z-20 py-2 *:pointer-events-auto"
              : // 1px less at the bottom for the page's top border.
                "pt-2 pb-[calc(var(--spacing)*2-1px)]",
          )}
        >
          {activeTab && activeEntry?.kind === "file" ? (
            floatingFileControls ? (
              <FloatingFileToolbar tab={activeTab} entry={activeEntry} />
            ) : isVideoEntry(activeEntry) ? (
              <VideoFileToolbar tab={activeTab} entry={activeEntry} />
            ) : (
              <BrowserFileToolbar tab={activeTab} entry={activeEntry} />
            )
          ) : (
            <WebToolbar tab={activeTab} />
          )}
        </div>
        {floatingFileControls || (activeEntry?.kind === "file" && isVideoEntry(activeEntry)) ? null : (
          <BookmarksBar tab={activeTab} />
        )}
        {deviceWidth ? <DeviceBar /> : null}
        <div
          ref={setPageElement}
          data-browser-page=""
          className={cn(
            "relative min-h-0 flex-1 overflow-hidden",
            floatingFileControls ? "browser-file-page" : "border-t border-border/70",
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
            data-browser-page=""
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
            // Refreshed bytes are a new document: the marks were on the old one.
            <AnnotateLayer
              key={`${activeTab.id}:${activeEntry.fileId}`}
              page={pageElement}
              fileName={activeEntry.name}
            />
          ) : null}
          {activeTab && activeEntry && (pageAnnotating === "frame" || pageAnnotating === "native") ? (
            // A new page, a reload or a replacement starts over: its marks were the last page's.
            <WebAnnotateLayer
              key={`${activeTab.id}:${entryKey(activeEntry)}:${activeTab.reloadKey}:${pageAnnotating}`}
              tabId={activeTab.id}
              title={activeTab.title}
              url={webAddress(activeTab) ?? ""}
              page={pageElement}
              native={pageAnnotating === "native"}
            />
          ) : null}
          {activeTab && activeEntry && pageAnnotating === "dom" && pageElement ? (
            <AnnotateLayer
              key={`${activeTab.id}:${entryKey(activeEntry)}:${activeTab.reloadKey}`}
              page={pageElement}
              fileName={tabTitle(activeTab, activeEntry)}
              url={webAddress(activeTab) ?? undefined}
            />
          ) : null}
        </div>
      </div>
    </section>
  );
});
