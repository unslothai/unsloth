// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A pinned page's sidebar row: icon, name, unpin and the tab's menu. Clicking shows its tab,
// opening one if needed. The sidebar places and drags it like a chat (`rowProps`).

import {
  ContextMenu,
  ContextMenuContent,
  ContextMenuTrigger,
} from "@/components/ui/context-menu";
import { NonModalDropdownMenu } from "@/components/ui/non-modal-dropdown-menu";
import { SidebarMenuButton, SidebarMenuItem } from "@/components/ui/sidebar";
import { useT } from "@/i18n";
import { openExternalLink } from "@/lib/open-link";
import { cn } from "@/lib/utils";
import { MoreHorizontalIcon, PinOffIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useNavigate, useRouterState } from "@tanstack/react-router";
import { type HTMLAttributes, useRef, useState } from "react";
import { browserPanelAvailable } from "./panel-availability";
import { type PinnedPage, usePinnedPagesStore } from "./pinned-pages-store";
import { SiteFavicon } from "./site-favicon";
import { MAX_TAB_TITLE_CHARS, useBrowserStore } from "./store";
import {
  CONTEXT_TAB_MENU,
  DROPDOWN_TAB_MENU,
  TabMenuItems,
  focusRenameField,
  renameTabTo,
  unpinPage,
  usePinnedTab,
} from "./tab-menu";

const MENU = "unsloth-plus-menu sidebar-row-menu sidebar-menu w-60";
// As a chat row's: the pin and the kebab show on hover, and while the menu is open.
const ACTION =
  "sidebar-row-action sidebar-touch-reveal group-hover/recent-item:opacity-100 group-hover/recent-item:pointer-events-auto focus-visible:opacity-100 focus-visible:pointer-events-auto group-has-[.sidebar-row-action[data-state=open]]/recent-item:opacity-100 group-has-[.sidebar-row-action[data-state=open]]/recent-item:pointer-events-auto";

/** Every pinned page, in pin order. */
export function usePinnedPages(): PinnedPage[] {
  return usePinnedPagesStore((state) => state.pages);
}

export function PinnedPageRow({
  page,
  className,
  rowProps,
}: {
  page: PinnedPage;
  className?: string;
  /** Sidebar drag and drop props, as on a chat row. */
  rowProps?: HTMLAttributes<HTMLLIElement> & Record<`data-${string}`, string>;
}) {
  const t = useT();
  const navigate = useNavigate();
  const onChat = useRouterState({ select: (state) => state.location.pathname.startsWith("/chat") });
  const tab = usePinnedTab(page.id);
  const active = useBrowserStore((state) => state.open && tab !== undefined && state.activeTabId === tab.id);
  const [renaming, setRenaming] = useState(false);
  // Rename moves focus to the field; the closing menu mustn't take it back to its trigger.
  const keepFocus = useRef(false);
  const title = tab?.customTitle || page.title || page.url;

  const open = () => {
    const show = () => useBrowserStore.getState().openPinned(page.id, page.url, page.title);
    if (onChat) {
      if (browserPanelAvailable()) show();
      else openExternalLink(page.url);
      return;
    }
    // The panel lives beside a chat: go to one first, once it is on screen.
    void navigate({ to: "/chat" }).then(() => requestAnimationFrame(() => requestAnimationFrame(show)));
  };
  const startRename = () => {
    keepFocus.current = true;
    setRenaming(true);
  };
  const keepMenuFocus = (event: Event) => {
    if (!keepFocus.current) return;
    keepFocus.current = false;
    event.preventDefault();
    focusRenameField(`[data-pinned-rename="${CSS.escape(page.id)}"]`);
  };

  if (renaming) {
    return (
      <SidebarMenuItem className="group/recent-item relative">
        <RenameField
          id={page.id}
          title={title}
          label={t("browser.tabMenu.renameLabel")}
          onDone={(value) => {
            setRenaming(false);
            if (value !== null && value.trim() !== title) renameTabTo(tab, page, value);
          }}
        />
      </SidebarMenuItem>
    );
  }

  return (
    <ContextMenu>
      <ContextMenuTrigger asChild={true}>
        <SidebarMenuItem
          {...rowProps}
          className={cn("group/recent-item relative", className)}
          data-pinned-page={page.id}
        >
          <SidebarMenuButton
            isActive={active}
            title={page.url}
            onClick={open}
            onDoubleClick={(event) => {
              event.preventDefault();
              setRenaming(true);
            }}
            className="sidebar-nav-btn h-[calc(30px*var(--ui-space-scale,1))] cursor-pointer gap-[calc(8.5px*var(--ui-space-scale,1))] rounded-full py-0 pl-3 pr-4 text-ui-14p5 font-medium leading-ui-19 tracking-nav group-hover/recent-item:pr-16 group-has-[.sidebar-row-action:focus-visible]/recent-item:pr-16 group-has-[.sidebar-row-action[data-state=open]]/recent-item:pr-16 [@media(pointer:coarse)]:pr-16"
          >
            <SiteFavicon
              url={page.url}
              icon={tab?.favicon ?? undefined}
              className="size-4 rounded-[3px]"
              fallbackClassName="size-icon"
            />
            <span className="truncate">{title}</span>
          </SidebarMenuButton>
          <button
            type="button"
            onClick={(event) => {
              event.stopPropagation();
              unpinPage(page.id);
            }}
            aria-label={t("browser.tabMenu.unpin")}
            className={cn(ACTION, "is-unpin-action")}
          >
            <span className="sidebar-row-action-glyph">
              <HugeiconsIcon icon={PinOffIcon} strokeWidth={1.75} className="size-icon" />
            </span>
          </button>
          <NonModalDropdownMenu
            side="bottom"
            align="start"
            sideOffset={0}
            className={MENU}
            onCloseAutoFocus={keepMenuFocus}
            trigger={(triggerRef) => (
              <button
                ref={triggerRef}
                type="button"
                onClick={(event) => event.stopPropagation()}
                aria-label={t("browser.tabMenu.pageOptions")}
                className={ACTION}
              >
                <span className="sidebar-row-action-glyph">
                  <HugeiconsIcon icon={MoreHorizontalIcon} strokeWidth={1.75} className="size-icon" />
                </span>
              </button>
            )}
          >
            <TabMenuItems P={DROPDOWN_TAB_MENU} tab={tab} pinned={page} onRename={startRename} />
          </NonModalDropdownMenu>
        </SidebarMenuItem>
      </ContextMenuTrigger>
      <ContextMenuContent className={MENU} onCloseAutoFocus={keepMenuFocus}>
        <TabMenuItems P={CONTEXT_TAB_MENU} tab={tab} pinned={page} onRename={startRename} />
      </ContextMenuContent>
    </ContextMenu>
  );
}

function RenameField({
  id,
  title,
  label,
  onDone,
}: {
  id: string;
  title: string;
  label: string;
  onDone: (value: string | null) => void;
}) {
  const done = useRef(false);
  const finish = (value: string | null) => {
    if (done.current) return;
    done.current = true;
    onDone(value);
  };
  return (
    <input
      // biome-ignore lint/a11y/noAutofocus: opened from Rename, to type the name
      autoFocus={true}
      defaultValue={title}
      maxLength={MAX_TAB_TITLE_CHARS}
      aria-label={label}
      data-pinned-rename={id}
      onFocus={(event) => event.currentTarget.select()}
      onBlur={(event) => finish(event.currentTarget.value)}
      onKeyDown={(event) => {
        if (event.key === "Enter") finish(event.currentTarget.value);
        else if (event.key === "Escape") finish(null);
      }}
      className="h-[calc(30px*var(--ui-space-scale,1))] w-full border-0 bg-transparent py-0 pl-3 pr-4 text-ui-14p5 font-medium leading-ui-19 tracking-nav text-foreground outline-none"
    />
  );
}
