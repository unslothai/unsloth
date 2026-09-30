// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

const EDGE_OFFSET = 12;
const MOBILE_EDGE_OFFSET = 16;
const HEADER_TOP_OFFSET = 52;
const DESKTOP_TITLEBAR_HEIGHT = 34;

export const CHAT_SETTINGS_INSET_VAR = "--studio-chat-settings-inset";
// Only the widest card scales with --ui-space-scale; the download panel and rail gutters do not.
const CORNER_CARD_MAX_WIDTH = 448;
const DOWNLOAD_PANEL_WIDTH = 400;
const CORNER_CARD_GUTTERS = 44;

export function chatSettingsInsetMinColumn(uiSpaceScale = 1): number {
  return (
    Math.max(CORNER_CARD_MAX_WIDTH * uiSpaceScale, DOWNLOAD_PANEL_WIDTH) +
    CORNER_CARD_GUTTERS
  );
}

const HEADER_ROUTES = new Set(["/chat", "/images", "/video", "/audio"]);

export type ToastOffset = {
  top: number;
  right: number;
};

export type ToastOffsets = {
  default: ToastOffset;
  mobile: ToastOffset;
};

export function getToastOffsets(
  pathname: string,
  isDesktopApp: boolean,
  usesCustomTitlebar: boolean,
  /** The page header grows with the UI font size; the titlebar band does not. */
  uiSpaceScale = 1,
): ToastOffsets {
  const hasPageHeader =
    HEADER_ROUTES.has(pathname) || pathname.startsWith("/chat/");
  const titlebarOffset =
    isDesktopApp && (!hasPageHeader || usesCustomTitlebar)
      ? DESKTOP_TITLEBAR_HEIGHT
      : 0;
  const headerTopOffset = Math.round(HEADER_TOP_OFFSET * uiSpaceScale);
  const defaultTopOffset = hasPageHeader ? headerTopOffset : EDGE_OFFSET;
  const mobileTopOffset = hasPageHeader ? headerTopOffset : MOBILE_EDGE_OFFSET;

  return {
    default: {
      top: defaultTopOffset + titlebarOffset,
      right: EDGE_OFFSET,
    },
    mobile: {
      top: mobileTopOffset + titlebarOffset,
      right: MOBILE_EDGE_OFFSET,
    },
  };
}

export function insetPastChatSettings(offset: ToastOffset): {
  top: number;
  right: string;
} {
  return {
    top: offset.top,
    right: `calc(${offset.right}px + var(${CHAT_SETTINGS_INSET_VAR}, 0px))`,
  };
}

type InsetPanel = {
  offsetWidth: number;
  parentElement: { clientWidth: number } | null;
};
type InsetObserver = new (
  callback: () => void,
) => {
  observe(target: object): void;
  disconnect(): void;
};

// Live width, not the stored one: a drag paints the panel before it commits.
export function watchChatSettingsInset(
  root: { style: Pick<CSSStyleDeclaration, "setProperty" | "removeProperty"> },
  panel: InsetPanel | null,
  fallbackWidth: number,
  uiSpaceScale = 1,
  Observer: InsetObserver = ResizeObserver,
): () => void {
  const row = panel?.parentElement ?? null;
  const minColumn = chatSettingsInsetMinColumn(uiSpaceScale);
  let applied: string | null = null;
  const apply = () => {
    const width = panel?.offsetWidth || fallbackWidth;
    const fits = !row || row.clientWidth - width >= minColumn;
    const next = fits ? `${width}px` : null;
    if (next === applied) return;
    applied = next;
    if (next) root.style.setProperty(CHAT_SETTINGS_INSET_VAR, next);
    else root.style.removeProperty(CHAT_SETTINGS_INSET_VAR);
  };
  apply();
  const observer = panel ? new Observer(apply) : null;
  if (row) observer?.observe(row);
  if (panel) observer?.observe(panel);
  return () => {
    observer?.disconnect();
    root.style.removeProperty(CHAT_SETTINGS_INSET_VAR);
  };
}
