// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

const EDGE_OFFSET = 12;
const MOBILE_EDGE_OFFSET = 16;
const HEADER_TOP_OFFSET = 52;
const DESKTOP_TITLEBAR_HEIGHT = 34;

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
): ToastOffsets {
  const hasPageHeader =
    HEADER_ROUTES.has(pathname) || pathname.startsWith("/chat/");
  // Page headers sit in the titlebar band on every desktop titlebar, with their controls
  // at the same height, so only a page without one has the band to clear.
  const titlebarOffset =
    isDesktopApp && !hasPageHeader ? DESKTOP_TITLEBAR_HEIGHT : 0;
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
