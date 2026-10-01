// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { isTauri } from "@/lib/api-base";
import { create } from "zustand";
import { setNativeWebHistory } from "./store";

/** Whether pages open in native views: in the desktop app where they can use its checking proxy
 *  (not macOS 13 and earlier), the proxied frame otherwise. */
export const useNativeBrowser = create(() => ({ enabled: false }));

export async function callNative<T = void>(command: string, args?: Record<string, unknown>): Promise<T> {
  const { invoke } = await import("@tauri-apps/api/core");
  return invoke<T>(command, args);
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

let clearing = false;
const closedListeners = new Set<() => void>();

/** True while a clear runs: a page shown meanwhile could write the cleared data back. */
export function nativeClearing(): boolean {
  return clearing;
}

/** Called once a clear has closed every native page. */
export function onNativeViewsClosed(listener: () => void): void {
  closedListeners.add(listener);
}

/** Clear the native pages' own cookies, storage and cache, closing the pages for the clear. */
export async function clearNativeBrowsingData(): Promise<void> {
  if (!useNativeBrowser.getState().enabled) return;
  clearing = true;
  try {
    await callNative("browser_view_clear_data", { closeViews: true });
  } finally {
    clearing = false;
    for (const listener of closedListeners) listener();
  }
}
