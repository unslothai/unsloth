// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { isTauri } from "./api-base.ts";

let clearing = false;
const closedListeners = new Set<() => void>();

/** A page shown during a clear could write the cleared data back. */
export function nativeClearing(): boolean {
  return clearing;
}

export function onNativeViewsClosed(listener: () => void): void {
  closedListeners.add(listener);
}

/** Clear data and account switches both come here, so neither races a page reopening. */
export async function clearNativeBrowsingData(): Promise<void> {
  if (!isTauri) return;
  const { invoke } = await import("@tauri-apps/api/core");
  // A failed probe fails the clear: skipping it would keep the previous account's cookies.
  if (!(await invoke<boolean>("browser_view_supported"))) return;
  clearing = true;
  try {
    await invoke("browser_view_clear_data", { closeViews: true });
  } finally {
    clearing = false;
    for (const listener of closedListeners) listener();
  }
}
