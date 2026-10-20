// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { isTauri } from "@/lib/api-base";
import { create } from "zustand";
import { currentEntry, setNativeWebHistory, useBrowserStore } from "./store";

/** Whether pages open in native views: in the desktop app where they can use its checking proxy
 *  (not macOS 13 and earlier), the proxied frame otherwise. */
export const useNativeBrowser = create(() => ({ enabled: false }));

export async function callNative<T = void>(command: string, args?: Record<string, unknown>): Promise<T> {
  const { invoke } = await import("@tauri-apps/api/core");
  return invoke<T>(command, args);
}

const localOrigins = new Map<string, string>();

export async function grantLocalOrigin(tabId: string, url: string): Promise<void> {
  await callNative("browser_view_grant", { tabId, url });
  if (!useBrowserStore.getState().tabs.some((tab) => tab.id === tabId)) {
    await callNative("browser_view_close", { tabId, forgetLocal: true });
    return;
  }
  localOrigins.set(tabId, new URL(url).origin);
  watchLocalTabs();
}

export function localOriginGranted(tabId: string, url: string): boolean {
  try {
    return localOrigins.get(tabId) === new URL(url).origin;
  } catch {
    return false;
  }
}

let closeLocalView = (tabId: string): void => void callNative("browser_view_close", { tabId, forgetLocal: true }).catch(() => undefined);

/** The views keep their own bookkeeping, so they close a view whose tab left its local origin. */
export function setLocalViewCloser(close: (tabId: string) => void): void {
  closeLocalView = close;
}

let watching = false;

// Subscribed on the first grant, not at module scope: the store is reached through the chat import cycle.
function watchLocalTabs(): void {
  if (watching) return;
  watching = true;
  useBrowserStore.subscribe((state) => {
    for (const tabId of localOrigins.keys()) {
      const tab = state.tabs.find((tab) => tab.id === tabId);
      const entry = tab && currentEntry(tab);
      if (entry?.kind === "web" && localOriginGranted(tabId, entry.url)) continue;
      localOrigins.delete(tabId);
      if (!tab) void callNative("browser_view_close", { tabId, forgetLocal: true }).catch(() => undefined);
      // A public page drops the local grant too.
      else if (entry?.kind !== "web") closeLocalView(tabId);
    }
  });
}

// Here rather than with the views, which load with the panel: navigation needs it before then.
if (isTauri) {
  void callNative<boolean>("browser_view_supported")
    .then((supported) => {
      if (!supported) return;
      setNativeWebHistory(true);
      useNativeBrowser.setState({ enabled: true });
    })
    .catch(() => undefined);
}

import { onNativeViewsClosed } from "@/lib/native-browser-clear";
onNativeViewsClosed(() => localOrigins.clear());
export { clearNativeBrowsingData, nativeClearing, onNativeViewsClosed } from "@/lib/native-browser-clear";
