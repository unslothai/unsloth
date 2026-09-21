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
