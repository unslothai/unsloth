// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Pushes backend-persisted pill settings to the Rust layer on startup so the
// hotkey works before the settings UI is ever opened.

import { useEffect } from "react";
import { isTauri } from "@/lib/api-base";
import { fetchPillSettings, syncNativePillConfig } from "./api";
import { isMacPlatform } from "@/lib/pill-native";

// server-port re-announcements can arrive in bursts; one sync per window.
let lastSyncStartedAt = 0;

async function syncConfigToNative(): Promise<void> {
  if (Date.now() - lastSyncStartedAt < 30_000) return;
  lastSyncStartedAt = Date.now();
  try {
    await syncNativePillConfig(await fetchPillSettings());
  } catch {
    // Backend not up yet or signed out; the next sync trigger retries.
  }
}

export function PillConfigSync(): null {
  useEffect(() => {
    if (!isTauri || !isMacPlatform()) return;
    let disposed = false;
    let unlisten: (() => void) | undefined;

    // Give auth/backend startup a moment, then sync; also re-sync when the
    // backend port is (re)announced.
    const timer = setTimeout(() => void syncConfigToNative(), 3000);
    void import("@tauri-apps/api/event")
      .then(({ listen }) =>
        listen("server-port", () => {
          setTimeout(() => void syncConfigToNative(), 2000);
        }),
      )
      .then((cleanup) => {
        if (disposed) cleanup();
        else unlisten = cleanup;
      })
      .catch(() => undefined);

    return () => {
      disposed = true;
      clearTimeout(timer);
      unlisten?.();
    };
  }, []);

  return null;
}
