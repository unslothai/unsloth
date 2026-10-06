// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  ContextMenu,
  ContextMenuContent,
  ContextMenuItem,
  ContextMenuSeparator,
  ContextMenuTrigger,
} from "@/components/ui/context-menu";
import { useT } from "@/i18n";
import { copyToClipboard } from "@/lib/copy-to-clipboard";
import { openExternalLink } from "@/lib/open-link";
import { toast } from "@/lib/toast";
import {
  ArrowUpRight01Icon,
  Copy01Icon,
  LinkSquare02Icon,
  PlusSignIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon, type IconSvgElement } from "@hugeicons/react";
import type { ReactElement, ReactNode } from "react";
import { useBrowserStore } from "./store";

export const CONTEXT_MENU = "browser-menu unsloth-plus-menu sidebar-row-menu w-56";
const MENU_ICON = "size-icon";

export function MenuRow({
  icon,
  children,
  onSelect,
  destructive,
}: {
  icon: IconSvgElement;
  children: ReactNode;
  onSelect: () => void;
  destructive?: boolean;
}) {
  return (
    <ContextMenuItem
      onSelect={onSelect}
      variant={destructive ? "destructive" : "default"}
    >
      <HugeiconsIcon icon={icon} strokeWidth={1.75} className={MENU_ICON} />
      {children}
    </ContextMenuItem>
  );
}

/** The right-click menu of anything that stands for a page; with no `url` (a file from chat), only `extra`. */
export function LinkContextMenu({
  url,
  tabId,
  extra,
  onCloseAutoFocus,
  children,
}: {
  url: string | null;
  tabId?: string;
  extra?: ReactNode;
  /** Lets a row that opens something keep focus there as the menu closes. */
  onCloseAutoFocus?: (event: Event) => void;
  children: ReactElement;
}) {
  const t = useT();
  const { navigate, openUrl } = useBrowserStore.getState();
  return (
    <ContextMenu>
      <ContextMenuTrigger asChild={true}>{children}</ContextMenuTrigger>
      <ContextMenuContent className={CONTEXT_MENU} onCloseAutoFocus={onCloseAutoFocus}>
        {url ? (
          <>
            {tabId ? (
              <MenuRow
                icon={ArrowUpRight01Icon}
                onSelect={() => navigate(tabId, { url })}
              >
                {t("browser.suggestedMenu.open")}
              </MenuRow>
            ) : null}
            <MenuRow
              icon={PlusSignIcon}
              onSelect={() => openUrl(url, { newTab: true })}
            >
              {t("browser.suggestedMenu.openInNewTab")}
            </MenuRow>
            <MenuRow
              icon={LinkSquare02Icon}
              onSelect={() => openExternalLink(url)}
            >
              {t("browser.openExternal")}
            </MenuRow>
            <ContextMenuSeparator />
            <MenuRow
              icon={Copy01Icon}
              onSelect={() =>
                void copyToClipboard(url).then(
                  (ok) => ok && toast.success(t("browser.linkCopied")),
                )
              }
            >
              {t("browser.copyLink")}
            </MenuRow>
            {extra ? <ContextMenuSeparator /> : null}
          </>
        ) : null}
        {extra}
      </ContextMenuContent>
    </ContextMenu>
  );
}
