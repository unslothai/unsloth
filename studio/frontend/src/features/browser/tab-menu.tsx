// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A tab's menu, shared by its right-click in the tab strip and by a pinned page in the sidebar
// (its 3-dot menu and its right-click), which offer the same actions.

import {
  ContextMenuItem,
  ContextMenuSeparator,
  ContextMenuSub,
  ContextMenuSubContent,
  ContextMenuSubTrigger,
} from "@/components/ui/context-menu";
import {
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuSub,
  DropdownMenuSubContent,
  DropdownMenuSubTrigger,
} from "@/components/ui/dropdown-menu";
import { useChatRuntimeStore } from "@/features/chat";
import { resetToNewChat } from "@/features/library";
import { useT } from "@/i18n";
import { copyToClipboard } from "@/lib/copy-to-clipboard";
import { ForkIcon } from "@/lib/fork-icon";
import { openExternalLink } from "@/lib/open-link";
import { toast } from "@/lib/toast";
import {
  Add01Icon,
  BubbleChatAddIcon,
  BubbleChatTemporaryIcon,
  Cancel01Icon,
  Copy01Icon,
  Link01Icon,
  LinkSquare02Icon,
  PencilEdit02Icon,
  PinIcon,
  PinOffIcon,
  Refresh01Icon,
  VolumeHighIcon,
  VolumeMute02Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon, type IconSvgElement } from "@hugeicons/react";
import { useNavigate } from "@tanstack/react-router";
import { type ComponentType, Fragment, type ReactNode } from "react";
import { hostOf } from "./address";
import { callNative, useNativeBrowser } from "./native-support";
import { nativeAction, hasNativeView } from "./native-view";
import { type PinnedPage, usePinnedPagesStore } from "./pinned-pages-store";
import { type BrowserTab, currentEntry, useBrowserStore } from "./store";

const ICON = "size-icon";

export interface TabMenuParts {
  Item: ComponentType<{
    children?: ReactNode;
    variant?: "default" | "destructive";
    onSelect?: (event: Event) => void;
  }>;
  Separator: ComponentType<Record<string, never>>;
  Sub: ComponentType<{ children?: ReactNode }>;
  SubTrigger: ComponentType<{ children?: ReactNode }>;
  SubContent: ComponentType<{ children?: ReactNode; className?: string }>;
}

export const CONTEXT_TAB_MENU: TabMenuParts = {
  Item: ContextMenuItem,
  Separator: ContextMenuSeparator,
  Sub: ContextMenuSub,
  SubTrigger: ContextMenuSubTrigger,
  SubContent: ContextMenuSubContent,
};

export const DROPDOWN_TAB_MENU: TabMenuParts = {
  Item: DropdownMenuItem,
  Separator: DropdownMenuSeparator,
  Sub: DropdownMenuSub,
  SubTrigger: DropdownMenuSubTrigger,
  SubContent: DropdownMenuSubContent,
};

/** The address a tab shows, for a web page; null for files, new tabs and internal pages. */
export function tabAddress(tab: BrowserTab | undefined): string | null {
  const entry = tab ? currentEntry(tab) : null;
  return entry?.kind === "web" ? (tab?.displayUrl ?? entry.url) : null;
}

/** The tab showing a pinned page, if one is open. */
export function usePinnedTab(pinnedId: string | undefined): BrowserTab | undefined {
  return useBrowserStore((state) =>
    pinnedId ? state.tabs.find((tab) => tab.pinnedId === pinnedId) : undefined,
  );
}

function isNativePage(tab: BrowserTab): boolean {
  return currentEntry(tab).kind === "web" && hasNativeView(tab.id);
}

export function reloadTab(tab: BrowserTab): void {
  if (isNativePage(tab)) nativeAction(tab.id, "reload");
  else useBrowserStore.getState().reload(tab.id);
}

export function setTabMuted(tab: BrowserTab, muted: boolean): void {
  useBrowserStore.getState().setMuted(tab.id, muted);
  // The desktop app keeps it for the tab's native view, open now or later.
  if (useNativeBrowser.getState().enabled) {
    void callNative("browser_view_mute", { tabId: tab.id, muted }).catch(() => undefined);
  }
}

export function pinTab(tab: BrowserTab, title: string): void {
  const url = tabAddress(tab);
  const page = url ? usePinnedPagesStore.getState().pin(url, title) : null;
  if (page) useBrowserStore.getState().setTabPinned(tab.id, page.id);
}

export function unpinPage(pinnedId: string): void {
  usePinnedPagesStore.getState().unpin(pinnedId);
  const store = useBrowserStore.getState();
  for (const tab of store.tabs) if (tab.pinnedId === pinnedId) store.setTabPinned(tab.id, null);
}

/** Names the tab, and the sidebar pin it shows; empty gives both back their page's title. */
export function renameTabTo(tab: BrowserTab | undefined, pinned: PinnedPage | undefined, title: string): void {
  if (tab) useBrowserStore.getState().renameTab(tab.id, title);
  const pinnedId = pinned?.id ?? tab?.pinnedId;
  if (pinnedId) usePinnedPagesStore.getState().rename(pinnedId, title.trim() || tab?.title || "");
}

/** Focus a rename field once a menu has let go of focus. */
export function focusRenameField(selector: string): void {
  requestAnimationFrame(() => {
    const field = document.querySelector<HTMLInputElement>(selector);
    field?.focus();
    field?.select();
  });
}

type Navigate = ReturnType<typeof useNavigate>;

/** A new chat (temporary or not) with a copy of the tab, or the page, open beside it. */
function forkToChat(navigate: Navigate, source: { tab?: BrowserTab; url: string }, temporary: boolean): void {
  resetToNewChat();
  if (temporary) useChatRuntimeStore.getState().setIncognito(true);
  void navigate({ to: "/chat", search: { new: crypto.randomUUID() } }).then(() => {
    // After the switch has closed the panel for the chat that was open.
    requestAnimationFrame(() =>
      requestAnimationFrame(() => {
        const store = useBrowserStore.getState();
        if (source.tab && store.tabs.some((tab) => tab.id === source.tab?.id)) store.duplicateTab(source.tab.id);
        else store.openUrl(source.url, { newTab: true });
      }),
    );
  });
}

function Row({
  P,
  icon,
  children,
  onSelect,
}: {
  P: TabMenuParts;
  icon: IconSvgElement;
  children: ReactNode;
  onSelect: () => void;
}) {
  return (
    <P.Item onSelect={onSelect}>
      <HugeiconsIcon icon={icon} strokeWidth={1.75} className={ICON} />
      <span>{children}</span>
    </P.Item>
  );
}

/** Menu rows. `strip` adds tab strip actions; `pinned` acts on the pinned page's open tab. Rows that
 *  don't apply are omitted, not disabled. */
export function TabMenuItems({
  P,
  tab,
  pinned,
  strip = false,
  onRename,
}: {
  P: TabMenuParts;
  tab?: BrowserTab;
  pinned?: PinnedPage;
  strip?: boolean;
  onRename: () => void;
}) {
  const t = useT();
  const navigate = useNavigate();
  const tabs = useBrowserStore((state) => state.tabs);
  const url = (tab && tabAddress(tab)) ?? pinned?.url ?? null;
  const pinnedId = pinned?.id ?? tab?.pinnedId ?? null;
  const index = tab ? tabs.findIndex((other) => other.id === tab.id) : -1;
  const store = useBrowserStore.getState();
  const kind = tab ? currentEntry(tab).kind : null;

  // Empty groups drop out with their separators.
  const groups: ReactNode[][] = [
    [
      pinnedId ? (
        <Row key="unpin" P={P} icon={PinOffIcon} onSelect={() => unpinPage(pinnedId)}>
          {t("browser.tabMenu.unpin")}
        </Row>
      ) : tab && url ? (
        <Row key="pin" P={P} icon={PinIcon} onSelect={() => pinTab(tab, tab.customTitle || tab.title || hostOf(url))}>
          {t("browser.tabMenu.pin")}
        </Row>
      ) : null,
    ],
    [
      strip && tab ? (
        <Row key="newTabRight" P={P} icon={Add01Icon} onSelect={() => store.newTabAfter(tab.id)}>
          {t("browser.tabMenu.newTabRight")}
        </Row>
      ) : null,
      tab && kind !== "newtab" ? (
        <Row key="reload" P={P} icon={Refresh01Icon} onSelect={() => reloadTab(tab)}>
          {t("browser.reload")}
        </Row>
      ) : null,
      tab || url ? (
        <Row
          key="duplicate"
          P={P}
          icon={Copy01Icon}
          onSelect={() => (tab ? store.duplicateTab(tab.id) : url && store.openUrl(url, { newTab: true }))}
        >
          {t("browser.tabMenu.duplicate")}
        </Row>
      ) : null,
      url ? (
        <P.Sub key="fork">
          <P.SubTrigger>
            <HugeiconsIcon icon={ForkIcon} strokeWidth={1.75} className={ICON} />
            <span>{t("browser.tabMenu.fork")}</span>
          </P.SubTrigger>
          <P.SubContent className="browser-menu unsloth-plus-menu sidebar-row-menu w-48">
            <Row P={P} icon={BubbleChatAddIcon} onSelect={() => forkToChat(navigate, { tab, url }, false)}>
              {t("browser.tabMenu.forkNewChat")}
            </Row>
            <Row P={P} icon={BubbleChatTemporaryIcon} onSelect={() => forkToChat(navigate, { tab, url }, true)}>
              {t("browser.tabMenu.forkTemporaryChat")}
            </Row>
          </P.SubContent>
        </P.Sub>
      ) : null,
      url ? (
        <Row
          key="copyUrl"
          P={P}
          icon={Link01Icon}
          onSelect={() => void copyToClipboard(url).then((ok) => ok && toast.success(t("browser.linkCopied")))}
        >
          {t("browser.tabMenu.copyUrl")}
        </Row>
      ) : null,
      url ? (
        <Row key="openExternal" P={P} icon={LinkSquare02Icon} onSelect={() => openExternalLink(url)}>
          {t("browser.tabMenu.openExternal")}
        </Row>
      ) : null,
    ],
    [
      tab || pinned ? (
        <Row key="rename" P={P} icon={PencilEdit02Icon} onSelect={onRename}>
          {t("browser.tabMenu.rename")}
        </Row>
      ) : null,
      tab && kind === "web" ? (
        <Row
          key="mute"
          P={P}
          icon={tab.muted ? VolumeHighIcon : VolumeMute02Icon}
          onSelect={() => setTabMuted(tab, !tab.muted)}
        >
          {t(tab.muted ? "browser.tabMenu.unmute" : "browser.tabMenu.mute")}
        </Row>
      ) : null,
    ],
    [
      tab ? (
        <Row key="close" P={P} icon={Cancel01Icon} onSelect={() => store.closeTab(tab.id)}>
          {t("browser.tabMenu.close")}
        </Row>
      ) : null,
      strip && tab && tabs.length > 1 ? (
        <Row key="closeOthers" P={P} icon={Cancel01Icon} onSelect={() => store.closeOtherTabs(tab.id)}>
          {t("browser.tabMenu.closeOthers")}
        </Row>
      ) : null,
      strip && tab && index >= 0 && index < tabs.length - 1 ? (
        <Row key="closeRight" P={P} icon={Cancel01Icon} onSelect={() => store.closeTabsToRight(tab.id)}>
          {t("browser.tabMenu.closeRight")}
        </Row>
      ) : null,
    ],
  ];
  const shown = groups.map((group) => group.filter(Boolean)).filter((group) => group.length > 0);
  return (
    <>
      {shown.map((group, at) => (
        // biome-ignore lint/suspicious/noArrayIndexKey: fixed groups in a fixed order
        <Fragment key={at}>
          {at > 0 ? <P.Separator /> : null}
          {group}
        </Fragment>
      ))}
    </>
  );
}
