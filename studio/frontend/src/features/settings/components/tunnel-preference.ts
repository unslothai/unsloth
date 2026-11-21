// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

const USE_TUNNEL_KEY = "unsloth_api_use_tunnel";

let currentUseTunnelPref: boolean | null = null;
const useTunnelListeners = new Set<() => void>();

export function readUseTunnelPref(): boolean {
  if (currentUseTunnelPref !== null) {
    return currentUseTunnelPref;
  }
  if (typeof window === "undefined") {
    return true;
  }
  try {
    return window.localStorage.getItem(USE_TUNNEL_KEY) !== "false";
  } catch {
    return true;
  }
}

export function subscribeUseTunnelPref(listener: () => void): () => void {
  useTunnelListeners.add(listener);
  return () => {
    useTunnelListeners.delete(listener);
  };
}

export function writeUseTunnelPref(value: boolean): void {
  currentUseTunnelPref = value;
  if (typeof window !== "undefined") {
    try {
      window.localStorage.setItem(USE_TUNNEL_KEY, value ? "true" : "false");
    } catch {
      // keep the in-memory value when persistent storage is unavailable.
    }
  }
  for (const listener of useTunnelListeners) {
    listener();
  }
}
