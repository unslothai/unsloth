// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

const EDGE_OFFSET = 12;
const MOBILE_EDGE_OFFSET = 16;
const HEADER_TOP_OFFSET = 52;
// The custom titlebar's header controls end at 36px, the macOS ones at 42px.
const CUSTOM_TITLEBAR_HEADER_TOP_OFFSET = 46;
const DESKTOP_TITLEBAR_HEIGHT = 34;
const CUSTOM_TITLEBAR_HEIGHT = 42;

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
  /** The page header grows with the UI font size; the titlebar band does not. */
  uiSpaceScale = 1,
  usesCustomTitlebar = false,
): ToastOffsets {
  const hasPageHeader =
    HEADER_ROUTES.has(pathname) || pathname.startsWith("/chat/");
  // Page headers sit in the titlebar band on every desktop titlebar, so only a page
  // without one has the band to clear.
  const titlebarOffset =
    isDesktopApp && !hasPageHeader
      ? usesCustomTitlebar
        ? CUSTOM_TITLEBAR_HEIGHT
        : DESKTOP_TITLEBAR_HEIGHT
      : 0;
  const headerTopOffset = Math.round(
    (isDesktopApp && usesCustomTitlebar
      ? CUSTOM_TITLEBAR_HEADER_TOP_OFFSET
      : HEADER_TOP_OFFSET) * uiSpaceScale,
  );
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
