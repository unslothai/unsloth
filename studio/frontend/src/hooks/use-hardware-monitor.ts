// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useCallback, useSyncExternalStore } from "react";

// Whether the sidebar VRAM/RAM monitor is shown. Off by default (opt-in via
// Settings > Appearance); when off the sidebar also stops polling /api/system.
const MONITOR_KEY = "hardware_monitor_enabled";

function loadEnabled(): boolean {
  if (typeof window === "undefined") return false;
  try {
    const raw = window.localStorage.getItem(MONITOR_KEY);
    if (raw === null) return false;
    return raw === "true";
  } catch {
    return false;
  }
}

let enabledValue = loadEnabled();
const listeners = new Set<() => void>();

function subscribe(cb: () => void) {
  listeners.add(cb);
  if (typeof window === "undefined") {
    return () => listeners.delete(cb);
  }
  const onStorage = (e: StorageEvent) => {
    if (e.key === MONITOR_KEY || e.key === null) {
      enabledValue = loadEnabled();
      cb();
    }
  };
  window.addEventListener("storage", onStorage);
  return () => {
    listeners.delete(cb);
    window.removeEventListener("storage", onStorage);
  };
}

function setEnabledGlobal(next: boolean) {
  enabledValue = next;
  try {
    window.localStorage.setItem(MONITOR_KEY, String(next));
  } catch {}
  listeners.forEach((cb) => cb());
}

export function useHardwareMonitor() {
  const enabled = useSyncExternalStore(
    subscribe,
    () => enabledValue,
    () => false,
  );

  const setEnabled = useCallback((value: boolean) => setEnabledGlobal(value), []);
  const toggle = useCallback(() => setEnabledGlobal(!enabledValue), []);

  return { enabled, setEnabled, toggle };
}
