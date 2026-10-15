// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { isTauri } from "@/lib/api-base";
import { create } from "zustand";
import { setNativeWebHistory } from "./store";

/** Native views need the desktop app's checking proxy (not macOS 13 and earlier). */
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

export { clearNativeBrowsingData, nativeClearing, onNativeViewsClosed } from "@/lib/native-browser-clear";
