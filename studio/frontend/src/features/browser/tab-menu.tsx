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
  VolumeHighIcon,
  VolumeMute02Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon, type IconSvgElement } from "@hugeicons/react";
import { useNavigate } from "@tanstack/react-router";
import { GitBranchIcon, RefreshCw } from "lucide-react";
import type { ComponentType, ReactNode } from "react";
import { hostOf } from "./address";
import { callNative, useNativeBrowser } from "./native-support";
import { nativeAction, hasNativeView } from "./native-view";
import { type PinnedPage, usePinnedPagesStore } from "./pinned-pages-store";
import { type BrowserTab, currentEntry, useBrowserStore } from "./store";

const ICON = "size-icon";

export interface TabMenuParts {
  Item: ComponentType<{
    children?: ReactNode;
    disabled?: boolean;
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
  disabled,
  onSelect,
}: {
  P: TabMenuParts;
  icon: IconSvgElement;
  children: ReactNode;
  disabled?: boolean;
  onSelect: () => void;
}) {
  return (
    <P.Item disabled={disabled} onSelect={onSelect}>
      <HugeiconsIcon icon={icon} strokeWidth={1.75} className={ICON} />
      <span>{children}</span>
    </P.Item>
  );
}

/**
 * The menu's rows. From the tab strip (`tab`, `strip`) it offers what acts on the strip too (New tab
 * to the right, closing others); from the sidebar (`pinned`) it acts on the pinned page's tab when
 * one is open, and opens the page for what needs one.
 */
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
  const web = tab ? currentEntry(tab).kind === "web" : false;
  return (
    <>
      {pinnedId ? (
        <Row P={P} icon={PinOffIcon} onSelect={() => unpinPage(pinnedId)}>
          {t("browser.tabMenu.unpin")}
        </Row>
      ) : (
        <Row
          P={P}
          icon={PinIcon}
          disabled={!tab || !url}
          onSelect={() => tab && pinTab(tab, tab.customTitle || tab.title || (url ? hostOf(url) : ""))}
        >
          {t("browser.tabMenu.pin")}
        </Row>
      )}
      <P.Separator />
      {strip && tab ? (
        <Row P={P} icon={Add01Icon} onSelect={() => store.newTabAfter(tab.id)}>
          {t("browser.tabMenu.newTabRight")}
        </Row>
      ) : null}
      <P.Item disabled={!tab} onSelect={() => tab && reloadTab(tab)}>
        <RefreshCw strokeWidth={1.75} className={ICON} />
        <span>{t("browser.reload")}</span>
      </P.Item>
      <Row
        P={P}
        icon={Copy01Icon}
        disabled={!tab && !url}
        onSelect={() => (tab ? store.duplicateTab(tab.id) : url && store.openUrl(url, { newTab: true }))}
      >
        {t("browser.tabMenu.duplicate")}
      </Row>
      <P.Sub>
        <P.SubTrigger>
          <GitBranchIcon strokeWidth={1.75} className={ICON} />
          <span>{t("browser.tabMenu.fork")}</span>
        </P.SubTrigger>
        <P.SubContent className="unsloth-plus-menu sidebar-row-menu w-48">
          <Row
            P={P}
            icon={BubbleChatAddIcon}
            disabled={!url}
            onSelect={() => url && forkToChat(navigate, { tab, url }, false)}
          >
            {t("browser.tabMenu.forkNewChat")}
          </Row>
          <Row
            P={P}
            icon={BubbleChatTemporaryIcon}
            disabled={!url}
            onSelect={() => url && forkToChat(navigate, { tab, url }, true)}
          >
            {t("browser.tabMenu.forkTemporaryChat")}
          </Row>
        </P.SubContent>
      </P.Sub>
      <Row
        P={P}
        icon={Link01Icon}
        disabled={!url}
        onSelect={() =>
          url && void copyToClipboard(url).then((ok) => ok && toast.success(t("browser.linkCopied")))
        }
      >
        {t("browser.tabMenu.copyUrl")}
      </Row>
      <Row P={P} icon={LinkSquare02Icon} disabled={!url} onSelect={() => url && openExternalLink(url)}>
        {t("browser.tabMenu.openExternal")}
      </Row>
      <P.Separator />
      <Row P={P} icon={PencilEdit02Icon} disabled={!tab && !pinned} onSelect={onRename}>
        {t("browser.tabMenu.rename")}
      </Row>
      <Row
        P={P}
        icon={tab?.muted ? VolumeHighIcon : VolumeMute02Icon}
        disabled={!tab || !web}
        onSelect={() => tab && setTabMuted(tab, !tab.muted)}
      >
        {t(tab?.muted ? "browser.tabMenu.unmute" : "browser.tabMenu.mute")}
      </Row>
      <P.Separator />
      <Row P={P} icon={Cancel01Icon} disabled={!tab} onSelect={() => tab && store.closeTab(tab.id)}>
        {t("browser.tabMenu.close")}
      </Row>
      {strip && tab ? (
        <>
          <Row P={P} icon={Cancel01Icon} disabled={tabs.length < 2} onSelect={() => store.closeOtherTabs(tab.id)}>
            {t("browser.tabMenu.closeOthers")}
          </Row>
          <Row
            P={P}
            icon={Cancel01Icon}
            disabled={index < 0 || index >= tabs.length - 1}
            onSelect={() => store.closeTabsToRight(tab.id)}
          >
            {t("browser.tabMenu.closeRight")}
          </Row>
        </>
      ) : null}
    </>
  );
}

