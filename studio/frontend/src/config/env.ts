// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { apiUrl } from "@/lib/api-base";
import { create } from "zustand";

export const env = {
  MODE: import.meta.env.MODE,
  DEV: import.meta.env.DEV,
  PROD: import.meta.env.PROD,
  BASE_URL: import.meta.env.BASE_URL,
} as const;

// ── Platform / device type ──────────────────────────────────

export type DeviceType = "mac" | "windows" | "linux" | string;

interface PlatformState {
  deviceType: DeviceType;
  chatOnly: boolean;
  fetched: boolean;
  isChatOnly: () => boolean;
}

export const usePlatformStore = create<PlatformState>()((_, get) => ({
  deviceType: "linux",
  chatOnly: false,
  fetched: false,
  isChatOnly: () => get().chatOnly,
}));

// Client-side platform detection as fallback when backend isn't ready yet.
function detectLocalPlatform(): DeviceType {
  const ua = navigator.userAgent.toLowerCase();
  if (ua.includes("mac")) return "mac";
  if (ua.includes("win")) return "windows";
  return "linux";
}

export async function fetchDeviceType(): Promise<DeviceType> {
  const { fetched } = usePlatformStore.getState();
  if (fetched) return usePlatformStore.getState().deviceType;

  try {
    const res = await fetch(apiUrl("/api/health"));
    if (res.ok) {
      const data = (await res.json()) as { device_type?: string; chat_only?: boolean };
      const deviceType = data.device_type ?? detectLocalPlatform();
      const chatOnly = data.chat_only ?? deviceType === "mac";
      usePlatformStore.setState({ deviceType, chatOnly, fetched: true });
      return deviceType;
    }
  } catch {
    // Backend not ready — use client-side detection so chat-only guard
    // still works on initial load (important for macOS).
    const deviceType = detectLocalPlatform();
    const chatOnly = deviceType === "mac";
    usePlatformStore.setState({ deviceType, chatOnly, fetched: true });
    return deviceType;
  }

  return usePlatformStore.getState().deviceType;
}
