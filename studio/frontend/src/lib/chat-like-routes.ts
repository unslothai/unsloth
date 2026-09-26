// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Each renders its own full-height shell, whose page header sits in the titlebar band.
const CHAT_LIKE_ROUTES = new Set(["/chat", "/images", "/video", "/audio"]);

export function isChatLikeRoute(pathname: string): boolean {
  return CHAT_LIKE_ROUTES.has(pathname);
}
