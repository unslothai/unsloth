// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Pushes backend-persisted pill settings to the Rust layer on startup so the
// hotkey works before the settings UI is ever opened.

import { useEffect } from "react";
import { isTauri } from "@/lib/api-base";
import { fetchPillSettings, syncNativePillConfig } from "./api";
import { isMacPlatform } from "@/lib/pill-native";

// server-port re-announcements can arrive in bursts; one sync per window. The
// throttle keys on the last SUCCESS, not the last attempt: a failed startup
// sync used to bank 30s of silence and swallow the retry the port triggers,
// leaving the hotkey stale for the window's lifetime.
let lastSyncSucceededAt = 0;
let syncInFlight = false;

async function syncConfigToNative(): Promise<void> {
  if (syncInFlight) return;
  if (lastSyncSucceededAt && Date.now() - lastSyncSucceededAt < 30_000) return;
  syncInFlight = true;
  try {
    await syncNativePillConfig(await fetchPillSettings());
    lastSyncSucceededAt = Date.now();
  } catch {
    // Backend not up yet or signed out; the next sync trigger really does retry.
  } finally {
    syncInFlight = false;
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
