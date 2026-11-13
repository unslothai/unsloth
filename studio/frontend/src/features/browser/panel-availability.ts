// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useSyncExternalStore } from "react";

// Whether a chat is on screen to show the browser panel beside it; the chat sets it.
let panelAvailable = false;
const listeners = new Set<() => void>();

export function setBrowserPanelAvailable(available: boolean): void {
  if (panelAvailable === available) return;
  panelAvailable = available;
  for (const listener of listeners) listener();
}

export function browserPanelAvailable(): boolean {
  return panelAvailable;
}

// The chat whose browser is on screen, as the browser store names it; the store sets it.
let shownChat = "new";

export function setShownBrowserChat(chat: string): void {
  shownChat = chat;
}

/** A new chat's run reads this when it starts, so its actions stay in that chat after a switch. */
export function shownBrowserChat(): string {
  return shownChat;
}

const subscribe = (listener: () => void) => {
  listeners.add(listener);
  return () => void listeners.delete(listener);
};

export function useBrowserPanelAvailable(): boolean {
  return useSyncExternalStore(subscribe, browserPanelAvailable, browserPanelAvailable);
}
