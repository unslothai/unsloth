// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

const EDGE_OFFSET = 12;
const MOBILE_EDGE_OFFSET = 16;
const HEADER_TOP_OFFSET = 52;
const DESKTOP_TITLEBAR_HEIGHT = 34;

export const CHAT_SETTINGS_INSET_VAR = "--studio-chat-settings-inset";
// the desktop browser panel sets this while its native page spans the toast area and adjacent settings
export const BROWSER_PAGE_INSET_VAR = "--studio-browser-page-inset";
// only the widest card scales with --ui-space-scale; the download panel and rail gutters stay fixed
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
    right: `calc(${offset.right}px + max(var(${CHAT_SETTINGS_INSET_VAR}, 0px), var(${BROWSER_PAGE_INSET_VAR}, 0px)))`,
  };
}

type InsetStyle = Pick<
  CSSStyleDeclaration,
  "setProperty" | "removeProperty" | "getPropertyValue"
>;

const insetScopes = new Set<HTMLElement>();
const insetValues = new Map<string, string>();
const insetListeners = new Set<() => void>();

function notifyInsetChange(): void {
  for (const listener of insetListeners) listener();
}

function removeCornerInset(name: string): string {
  const previous = insetValues.get(name) ?? "";
  insetValues.delete(name);
  for (const scope of insetScopes) scope.style.removeProperty(name);
  if (previous) notifyInsetChange();
  return previous;
}

function setCornerInset(name: string, value: string | null): void {
  if (!value) {
    removeCornerInset(name);
    return;
  }
  const changed = insetValues.get(name) !== value;
  insetValues.set(name, value);
  for (const scope of insetScopes) scope.style.setProperty(name, value);
  if (changed) notifyInsetChange();
}

/** Fires after a corner inset changes: rail cards move with it without resizing, so no ResizeObserver sees it. */
export function subscribeCornerInsets(listener: () => void): () => void {
  insetListeners.add(listener);
  return () => {
    insetListeners.delete(listener);
  };
}

export const cornerInsets: { style: InsetStyle } = {
  style: {
    setProperty: setCornerInset,
    removeProperty: removeCornerInset,
    getPropertyValue: (name: string) => insetValues.get(name) ?? "",
  },
};

export function cornerInsetScope(
  element: HTMLElement | null,
): (() => void) | undefined {
  if (!element) return undefined;
  insetScopes.add(element);
  for (const [name, value] of insetValues) {
    element.style.setProperty(name, value);
  }
  return () => {
    insetScopes.delete(element);
  };
}

export const COMPOSER_CLEARANCE_VAR = "--studio-composer-clearance";
// The collapsed download button's column: 16px inset, 44px button, 16px of air.
const CORNER_RAIL_COLUMN = 76;
const composerClearances = new Map<object, number>();

function applyComposerClearance(): void {
  const lift = Math.max(0, ...composerClearances.values());
  cornerInsets.style.setProperty(COMPOSER_CLEARANCE_VAR, lift > 0 ? `${lift}px` : "");
}

/** Lifts the corner rail above `composer` while it reaches into the rail's column (#14146); compare panes each register. */
export function watchComposerClearance(
  composer: HTMLElement,
  uiSpaceScale = 1,
): () => void {
  const key = {};
  const apply = () => {
    const rect = composer.getBoundingClientRect();
    const intrudes =
      rect.height > 0 &&
      window.innerWidth - rect.right < CORNER_RAIL_COLUMN * uiSpaceScale;
    // The rail already keeps a 16px gutter; this leaves 8px between it and the composer.
    const lift = intrudes ? Math.round(window.innerHeight - rect.top - 8) : 0;
    if (lift > 0) composerClearances.set(key, lift);
    else composerClearances.delete(key);
    applyComposerClearance();
  };
  apply();
  const observer = new ResizeObserver(apply);
  observer.observe(composer);
  window.addEventListener("resize", apply);
  return () => {
    observer.disconnect();
    window.removeEventListener("resize", apply);
    composerClearances.delete(key);
    applyComposerClearance();
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
