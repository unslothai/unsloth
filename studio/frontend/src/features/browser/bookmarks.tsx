// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Input } from "@/components/ui/input";
import { Popover, PopoverAnchor, PopoverContent } from "@/components/ui/popover";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { useT } from "@/i18n";
import { toast } from "@/lib/toast";
import { cn } from "@/lib/utils";
import {
  ArrowRightDoubleIcon,
  Delete02Icon,
  Folder01Icon,
  PencilEdit02Icon,
  StarIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { type ReactNode, type RefObject, useEffect, useId, useMemo, useRef, useState } from "react";
import { hostOf } from "./address";
import {
  type Bookmark,
  type BookmarkFolder,
  useBookmarkFor,
  useBrowserBookmarksStore,
} from "./bookmarks-store";
import { LinkContextMenu, MenuRow } from "./link-context-menu";
import { useBrowserPrefsStore } from "./prefs-store";
import { type BrowserTab, currentEntry, useBrowserStore } from "./store";
import { SiteFavicon } from "./site-favicon";

const FOLDERS: Record<BookmarkFolder, "browser.bookmarks.toolbar" | "browser.bookmarks.other"> = {
  toolbar: "browser.bookmarks.toolbar",
  other: "browser.bookmarks.other",
};

const HOVER_WASH =
  "hover:bg-[color-mix(in_oklab,var(--foreground)_calc(6%*var(--contrast-wash-gain,1)),transparent)] data-[state=open]:bg-[color-mix(in_oklab,var(--foreground)_calc(8%*var(--contrast-wash-gain,1)),transparent)]";

const ICON_PX = 32;

/** A tab's icon (a data: URL) as the small PNG a bookmark keeps, so it shows without a fetch. */
function bookmarkIcon(src: string): Promise<string | null> {
  return new Promise((resolve) => {
    const image = new Image();
    image.onload = () => {
      try {
        const canvas = document.createElement("canvas");
        canvas.width = ICON_PX;
        canvas.height = ICON_PX;
        const context = canvas.getContext("2d");
        if (!context) {
          resolve(null);
          return;
        }
        // Fitted, so a wide icon keeps its shape. An SVG with no size of its own fills the square.
        const width = image.naturalWidth || ICON_PX;
        const height = image.naturalHeight || ICON_PX;
        const scale = ICON_PX / Math.max(width, height);
        const drawn = { width: width * scale, height: height * scale };
        context.drawImage(image, (ICON_PX - drawn.width) / 2, (ICON_PX - drawn.height) / 2, drawn.width, drawn.height);
        resolve(canvas.toDataURL("image/png"));
      } catch {
        resolve(null);
      }
    };
    image.onerror = () => resolve(null);
    image.src = src;
  });
}

function saveBookmarkIcon(url: string, favicon: string | null): void {
  if (!favicon?.startsWith("data:image/")) return;
  const bookmark = useBrowserBookmarksStore.getState().bookmarks.find((candidate) => candidate.url === url);
  if (!bookmark) return;
  void bookmarkIcon(favicon).then((icon) => {
    if (icon) useBrowserBookmarksStore.getState().setBookmarkIcon(bookmark.id, icon);
  });
}

function tabAddress(tab: BrowserTab): string | null {
  const entry = currentEntry(tab);
  return entry.kind === "web" ? (tab.displayUrl ?? entry.url) : null;
}

// A bookmarked page's icon, whenever a tab gets one (framed or native), so a changed icon follows.
useBrowserStore.subscribe((state, previous) => {
  if (state.tabs === previous.tabs) return;
  for (const tab of state.tabs) {
    if (!tab.favicon) continue;
    const before = previous.tabs.find((candidate) => candidate.id === tab.id);
    const url = tabAddress(tab);
    if (!url || (before?.favicon === tab.favicon && before && tabAddress(before) === url)) continue;
    saveBookmarkIcon(url, tab.favicon);
  }
});

export function bookmarkTitle(bookmark: Bookmark): string {
  return bookmark.title || hostOf(bookmark.url) || bookmark.url;
}

export function removeBookmarkWithUndo(bookmark: Bookmark, t: ReturnType<typeof useT>): void {
  const store = useBrowserBookmarksStore.getState();
  const index = store.bookmarks.findIndex((other) => other.id === bookmark.id);
  store.removeBookmark(bookmark.id);
  toast(t("browser.bookmarks.removed"), {
    action: {
      label: t("browser.bookmarks.undo"),
      onClick: () => useBrowserBookmarksStore.getState().restoreBookmark(bookmark, index),
    },
  });
}

function openBookmark(url: string, tabId: string | undefined, newTab: boolean): void {
  const store = useBrowserStore.getState();
  if (newTab || !tabId) store.openUrl(url, { newTab: true });
  else store.navigate(tabId, { url });
}

/** The bookmark editor: edits apply as it closes unless cancelled; cancelling a new bookmark removes it. */
function BookmarkEditor({
  bookmark,
  isNew,
  onClose,
}: {
  bookmark: Bookmark;
  isNew: boolean;
  onClose: () => void;
}) {
  const t = useT();
  const id = useId();
  const showEditor = useBrowserPrefsStore((state) => state.showBookmarkEditor);
  const [title, setTitle] = useState(bookmarkTitle(bookmark));
  const [folder, setFolder] = useState(bookmark.folder);
  const draft = useRef({ title, folder, discard: false });
  draft.current = { ...draft.current, title, folder };
  // Applies on unmount, so a click outside the panel keeps the edits as Firefox's does.
  useEffect(
    () => () => {
      const { title: name, folder: place, discard } = draft.current;
      if (discard) return;
      useBrowserBookmarksStore.getState().updateBookmark(bookmark.id, { title: name, folder: place });
    },
    [bookmark.id],
  );
  const dismiss = (remove: boolean) => {
    draft.current.discard = true;
    if (remove) {
      if (isNew) useBrowserBookmarksStore.getState().removeBookmark(bookmark.id);
      else removeBookmarkWithUndo(bookmark, t);
    }
    onClose();
  };
  return (
    <form
      className="flex flex-col gap-4"
      onSubmit={(event) => {
        event.preventDefault();
        onClose();
      }}
    >
      <div className="flex flex-col gap-3">
        <h2 className="text-center text-ui-14 font-semibold text-foreground">
          {t(isNew ? "browser.bookmarks.add" : "browser.bookmarks.edit")}
        </h2>
        <div className="h-px bg-border" />
      </div>
      <div className="flex flex-col gap-1.5">
        <label htmlFor={`${id}-name`} className="text-ui-13 text-foreground">
          {t("browser.bookmarks.name")}
        </label>
        <Input
          id={`${id}-name`}
          data-bookmark-name=""
          value={title}
          onChange={(event) => setTitle(event.target.value)}
          spellCheck={false}
          autoComplete="off"
        />
      </div>
      <div className="flex flex-col gap-1.5">
        <span id={`${id}-location`} className="text-ui-13 text-foreground">
          {t("browser.bookmarks.location")}
        </span>
        <Select value={folder} onValueChange={(value) => setFolder(value as BookmarkFolder)}>
          <SelectTrigger aria-labelledby={`${id}-location`} className="w-full">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {(Object.keys(FOLDERS) as BookmarkFolder[]).map((value) => (
              <SelectItem key={value} value={value}>
                <HugeiconsIcon
                  icon={value === "toolbar" ? StarIcon : Folder01Icon}
                  strokeWidth={1.75}
                  className="size-4 text-muted-foreground"
                />
                {t(FOLDERS[value])}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>
      <div className="flex items-center gap-2">
        <Checkbox
          id={`${id}-show-editor`}
          checked={showEditor}
          onCheckedChange={(checked) => useBrowserPrefsStore.getState().setShowBookmarkEditor(checked === true)}
        />
        <label htmlFor={`${id}-show-editor`} className="cursor-pointer text-ui-13 text-muted-foreground">
          {t("browser.bookmarks.showEditor")}
        </label>
      </div>
      <div className="flex justify-end gap-2">
        <Button type="button" variant="outline" size="sm" onClick={() => dismiss(true)}>
          {t(isNew ? "browser.bookmarks.cancel" : "browser.bookmarks.remove")}
        </Button>
        <Button type="submit" size="sm">
          {t("browser.bookmarks.save")}
        </Button>
      </div>
    </form>
  );
}

export function BookmarkEditPopover({
  bookmark,
  isNew = false,
  open,
  onOpenChange,
  align = "end",
  children,
}: {
  bookmark: Bookmark | undefined;
  isNew?: boolean;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  align?: "start" | "end";
  children: ReactNode;
}) {
  return (
    <Popover open={open && bookmark !== undefined} onOpenChange={onOpenChange}>
      {children}
      {bookmark ? (
        <PopoverContent
          align={align}
          sideOffset={8}
          className="w-80 rounded-[16px] p-4"
          onOpenAutoFocus={(event) => {
            const input = (event.currentTarget as HTMLElement | null)?.querySelector<HTMLInputElement>(
              "[data-bookmark-name]",
            );
            if (!input) return;
            event.preventDefault();
            input.focus();
            input.select();
          }}
        >
          <BookmarkEditor bookmark={bookmark} isNew={isNew} onClose={() => onOpenChange(false)} />
        </PopoverContent>
      ) : null}
    </Popover>
  );
}

let handledBookmarkSequence = 0;

export function BookmarkStar({
  url,
  title,
  favicon,
}: {
  url: string | null;
  title: string;
  favicon: string | null;
}) {
  const t = useT();
  const bookmark = useBookmarkFor(url);
  const sequence = useBrowserStore((state) => state.bookmarkSequence);
  const [editing, setEditing] = useState<{ isNew: boolean } | null>(null);
  const press = () => {
    if (!url) return;
    if (bookmark) {
      setEditing({ isNew: false });
      return;
    }
    const saved = useBrowserBookmarksStore.getState().addBookmark(url, title || hostOf(url));
    if (!saved) return;
    saveBookmarkIcon(url, favicon);
    if (useBrowserPrefsStore.getState().showBookmarkEditor) setEditing({ isNew: true });
    else toast.success(t("browser.bookmarks.saved"));
  };
  const pressRef = useRef(press);
  pressRef.current = press;
  useEffect(() => {
    // Each ⌘D once: the bar remounts per tab, and a later tab shouldn't answer an earlier press.
    if (sequence === handledBookmarkSequence) return;
    handledBookmarkSequence = sequence;
    pressRef.current();
  }, [sequence]);
  // A bookmark made before icons were kept, or whose tab had none yet, takes this one.
  const missingIcon = bookmark !== undefined && !bookmark.icon;
  useEffect(() => {
    if (missingIcon && url) saveBookmarkIcon(url, favicon);
  }, [missingIcon, url, favicon]);
  if (!url) return null;
  const label = t(bookmark ? "browser.bookmarks.edit" : "browser.bookmarks.bookmarkPage");
  return (
    <BookmarkEditPopover
      bookmark={bookmark}
      isNew={editing?.isNew}
      open={editing !== null}
      onOpenChange={(open) => !open && setEditing(null)}
    >
      <Tooltip>
        <TooltipTrigger asChild={true}>
          <PopoverAnchor asChild={true}>
            <button
              type="button"
              aria-label={label}
              aria-pressed={bookmark !== undefined}
              aria-expanded={editing !== null}
              onClick={() => (editing ? setEditing(null) : press())}
              className={cn(
                "flex size-7 shrink-0 cursor-pointer items-center justify-center rounded-md text-muted-foreground transition-colors hover:bg-[color-mix(in_oklab,var(--foreground)_calc(8%*var(--contrast-wash-gain,1)),transparent)] hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring aria-expanded:bg-[color-mix(in_oklab,var(--foreground)_calc(10%*var(--contrast-wash-gain,1)),transparent)]",
                bookmark && "text-primary hover:text-primary [&_path]:fill-current",
              )}
            >
              <HugeiconsIcon icon={StarIcon} strokeWidth={1.75} className="size-4.25" />
            </button>
          </PopoverAnchor>
        </TooltipTrigger>
        <TooltipContent side="bottom" className="tooltip-compact">
          {label}
        </TooltipContent>
      </Tooltip>
    </BookmarkEditPopover>
  );
}

function BookmarkItem({
  bookmark,
  tabId,
  hidden,
}: {
  bookmark: Bookmark;
  tabId: string | undefined;
  /** Cut off by the bar's end: kept in place so it can be measured, shown under » instead. */
  hidden: boolean;
}) {
  const t = useT();
  const [editing, setEditing] = useState(false);
  const editChosen = useRef(false);
  const title = bookmarkTitle(bookmark);
  return (
    <BookmarkEditPopover
      bookmark={bookmark}
      open={editing}
      onOpenChange={setEditing}
      align="start"
    >
      <LinkContextMenu
        url={bookmark.url}
        tabId={tabId}
        onCloseAutoFocus={(event) => {
          // Focus stays in the editor the menu opened.
          if (!editChosen.current) return;
          editChosen.current = false;
          event.preventDefault();
        }}
        extra={
          <>
            <MenuRow
              icon={PencilEdit02Icon}
              onSelect={() => {
                editChosen.current = true;
                setEditing(true);
              }}
            >
              {t("browser.bookmarks.edit")}
            </MenuRow>
            <MenuRow icon={Delete02Icon} onSelect={() => removeBookmarkWithUndo(bookmark, t)}>
              {t("browser.bookmarks.delete")}
            </MenuRow>
          </>
        }
      >
        <PopoverAnchor asChild={true}>
          <button
            type="button"
            data-bookmark-id={bookmark.id}
            title={`${title}\n${bookmark.url}`}
            tabIndex={hidden ? -1 : undefined}
            aria-hidden={hidden || undefined}
            onClick={(event) => openBookmark(bookmark.url, tabId, event.metaKey || event.ctrlKey)}
            onAuxClick={(event) => {
              if (event.button === 1) openBookmark(bookmark.url, tabId, true);
            }}
            className={cn(
              "flex h-7 min-w-0 max-w-44 shrink-0 cursor-pointer items-center gap-1.5 rounded-md px-2 text-ui-12p5 text-foreground transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
              HOVER_WASH,
              hidden && "invisible",
            )}
          >
            <SiteFavicon
              url={bookmark.url}
              icon={bookmark.icon}
              className="size-4 rounded-[3px]"
              fallbackClassName="size-4 text-muted-foreground"
            />
            <span className="min-w-0 truncate">{title}</span>
          </button>
        </PopoverAnchor>
      </LinkContextMenu>
    </BookmarkEditPopover>
  );
}

function BookmarkMenu({
  label,
  icon,
  bookmarks,
  tabId,
  children,
}: {
  label: string;
  icon: typeof StarIcon;
  bookmarks: Bookmark[];
  tabId: string | undefined;
  children?: ReactNode;
}) {
  const t = useT();
  return (
    <DropdownMenu>
      <Tooltip>
        <TooltipTrigger asChild={true}>
          <DropdownMenuTrigger asChild={true}>
            <button
              type="button"
              aria-label={label}
              className={cn(
                "flex h-7 shrink-0 cursor-pointer items-center gap-1.5 rounded-md px-1.5 text-ui-12p5 text-foreground transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
                HOVER_WASH,
              )}
            >
              <HugeiconsIcon icon={icon} strokeWidth={1.75} className="size-4 text-muted-foreground" />
              {children}
            </button>
          </DropdownMenuTrigger>
        </TooltipTrigger>
        <TooltipContent side="bottom" className="tooltip-compact">
          {label}
        </TooltipContent>
      </Tooltip>
      <DropdownMenuContent align="end" sideOffset={6} className="max-h-[min(--spacing(96),var(--radix-dropdown-menu-content-available-height))] overflow-y-auto w-64 rounded-[16px] p-1.5">
        {bookmarks.map((bookmark) => (
          <DropdownMenuItem
            key={bookmark.id}
            title={bookmark.url}
            onSelect={() => openBookmark(bookmark.url, tabId, false)}
          >
            <SiteFavicon
              url={bookmark.url}
              icon={bookmark.icon}
              className="size-4 rounded-[3px]"
              fallbackClassName="size-4 text-muted-foreground"
            />
            <span className="min-w-0 flex-1 truncate">{bookmarkTitle(bookmark)}</span>
          </DropdownMenuItem>
        ))}
        <DropdownMenuSeparator />
        <DropdownMenuItem onSelect={() => useBrowserStore.getState().openInternal("bookmarks")}>
          {t("browser.bookmarks.manage")}
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}

/** Bar items cut off by its end, measured on resize: an IntersectionObserver misread them as the panel opened and under CSS zoom. */
function useClippedItems(ids: string): [RefObject<HTMLDivElement | null>, Set<string>] {
  const ref = useRef<HTMLDivElement>(null);
  const [clipped, setClipped] = useState<Set<string>>(() => new Set());
  useEffect(() => {
    const root = ref.current;
    if (!root || !ids) {
      setClipped(new Set());
      return;
    }
    const measure = () => {
      const end = root.getBoundingClientRect().right + 0.5;
      const next = new Set<string>();
      for (const node of root.querySelectorAll<HTMLElement>("[data-bookmark-id]")) {
        const id = node.dataset.bookmarkId;
        if (id && node.getBoundingClientRect().right > end) next.add(id);
      }
      setClipped((previous) =>
        next.size === previous.size && [...next].every((id) => previous.has(id)) ? previous : next,
      );
    };
    const observer = new ResizeObserver(measure);
    observer.observe(root);
    for (const node of root.querySelectorAll("[data-bookmark-id]")) observer.observe(node);
    measure();
    return () => observer.disconnect();
  }, [ids]);
  return [ref, clipped];
}

export function BookmarksBar({ tab }: { tab: BrowserTab | undefined }) {
  const t = useT();
  const mode = useBrowserPrefsStore((state) => state.bookmarksToolbar);
  const bookmarks = useBrowserBookmarksStore((state) => state.bookmarks);
  const toolbar = useMemo(() => bookmarks.filter((bookmark) => bookmark.folder === "toolbar"), [bookmarks]);
  const other = useMemo(() => bookmarks.filter((bookmark) => bookmark.folder === "other"), [bookmarks]);
  const [listRef, clipped] = useClippedItems(toolbar.map((bookmark) => bookmark.id).join(" "));
  const newTab = tab ? currentEntry(tab).kind === "newtab" : false;
  const shown = mode === "always" || (mode === "newtab" && newTab && bookmarks.length > 0);
  if (!shown) return null;
  const overflow = toolbar.filter((bookmark) => clipped.has(bookmark.id));
  return (
    <div
      role="toolbar"
      aria-label={t("browser.bookmarks.toolbar")}
      className="browser-chrome @container mt-0.5 flex h-9 shrink-0 items-center gap-1 px-2.5 pb-1.5"
    >
      <div ref={listRef} className="flex min-w-0 flex-1 items-center gap-0.5 overflow-hidden">
        {toolbar.length === 0 ? (
          <span className="truncate px-2 text-ui-12p5 text-muted-foreground">
            {t("browser.bookmarks.toolbarEmpty")}
          </span>
        ) : (
          toolbar.map((bookmark) => (
            <BookmarkItem key={bookmark.id} bookmark={bookmark} tabId={tab?.id} hidden={clipped.has(bookmark.id)} />
          ))
        )}
      </div>
      {overflow.length > 0 ? (
        <BookmarkMenu
          label={t("browser.bookmarks.more")}
          icon={ArrowRightDoubleIcon}
          bookmarks={overflow}
          tabId={tab?.id}
        />
      ) : null}
      {other.length > 0 ? (
        <>
          <span aria-hidden={true} className="mx-0.5 h-4 w-px shrink-0 bg-border" />
          <BookmarkMenu label={t("browser.bookmarks.other")} icon={Folder01Icon} bookmarks={other} tabId={tab?.id}>
            <span className="hidden @[32rem]:inline">{t("browser.bookmarks.other")}</span>
          </BookmarkMenu>
        </>
      ) : null}
    </div>
  );
}
