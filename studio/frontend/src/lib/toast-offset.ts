// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

const EDGE_OFFSET = 12;
const MOBILE_EDGE_OFFSET = 16;
const HEADER_TOP_OFFSET = 52;
const DESKTOP_TITLEBAR_HEIGHT = 34;

// Width of the open Run settings panel, published on <html> by ChatSettingsPanel.
export const CHAT_SETTINGS_INSET_VAR = "--studio-chat-settings-inset";
// Widest corner card (448px) plus its gutters; a narrower chat column cannot hold it.
export const CHAT_SETTINGS_INSET_MIN_COLUMN = 480;

const HEADER_ROUTES = new Set(["/chat", "/images", "/video"]);

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
): ToastOffsets {
  const hasPageHeader =
    HEADER_ROUTES.has(pathname) || pathname.startsWith("/chat/");
  const titlebarOffset =
    isDesktopApp && (!hasPageHeader || usesCustomTitlebar)
      ? DESKTOP_TITLEBAR_HEIGHT
      : 0;
  const defaultTopOffset = hasPageHeader ? HEADER_TOP_OFFSET : EDGE_OFFSET;
  const mobileTopOffset = hasPageHeader
    ? HEADER_TOP_OFFSET
    : MOBILE_EDGE_OFFSET;

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
type InsetObserver = new (callback: () => void) => {
  observe(target: object): void;
  disconnect(): void;
};

// Publishes the panel's live width: a drag paints the panel before it commits the stored width.
export function watchChatSettingsInset(
  root: { style: Pick<CSSStyleDeclaration, "setProperty" | "removeProperty"> },
  panel: InsetPanel | null,
  fallbackWidth: number,
  Observer: InsetObserver = ResizeObserver,
): () => void {
  const row = panel?.parentElement ?? null;
  let applied: string | null = null;
  const apply = () => {
    const width = panel?.offsetWidth || fallbackWidth;
    const fits =
      !row || row.clientWidth - width >= CHAT_SETTINGS_INSET_MIN_COLUMN;
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
