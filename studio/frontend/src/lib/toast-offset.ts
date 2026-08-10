// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

const EDGE_OFFSET = 12;
const MOBILE_EDGE_OFFSET = 16;
const CHAT_TOP_OFFSET = 52;
const DESKTOP_TITLEBAR_HEIGHT = 34;

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
): ToastOffsets {
  const isChatRoute = pathname === "/chat" || pathname.startsWith("/chat/");
  const titlebarOffset = isDesktopApp ? DESKTOP_TITLEBAR_HEIGHT : 0;

  return {
    default: {
      top: (isChatRoute ? CHAT_TOP_OFFSET : EDGE_OFFSET) + titlebarOffset,
      right: EDGE_OFFSET,
    },
    mobile: {
      top:
        (isChatRoute ? CHAT_TOP_OFFSET : MOBILE_EDGE_OFFSET) + titlebarOffset,
      right: MOBILE_EDGE_OFFSET,
    },
  };
}
