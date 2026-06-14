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

// Platform / device type

export type DeviceType = "mac" | "windows" | "linux" | string;

interface PlatformState {
  deviceType: DeviceType;
  chatOnly: boolean;
  // Live tunnel URL from /api/health (authed); null if none. Ephemeral -- read live.
  cloudflareUrl: string | null;
  // Direct host:port base from /api/health (authed); the non-tunnel API base.
  serverUrl: string | null;
  fetched: boolean;
  isChatOnly: () => boolean;
}

// Client-side fallback when backend isn't ready yet.
function detectLocalPlatform(): DeviceType {
  if (typeof navigator === "undefined") return "linux";
  const platform = navigator.platform.toLowerCase();
  const ua = navigator.userAgent.toLowerCase();
  if (platform.includes("mac") || ua.includes("mac")) return "mac";
  if (platform.includes("win") || ua.includes("win")) return "windows";
  return "linux";
}

const localDeviceType = detectLocalPlatform();

export const usePlatformStore = create<PlatformState>()((_, get) => ({
  deviceType: localDeviceType,
  chatOnly: localDeviceType === "mac",
  cloudflareUrl: null,
  serverUrl: null,
  fetched: false,
  isChatOnly: () => get().chatOnly,
}));

// `force` re-reads /api/health even if cached: the tunnel URL may arrive after
// the first health call, so consumers can refresh to pick it up.
export async function fetchDeviceType(options?: {
  force?: boolean;
}): Promise<DeviceType> {
  const { fetched } = usePlatformStore.getState();
  if (fetched && !options?.force) return usePlatformStore.getState().deviceType;

  try {
    // /api/health only reports the server's device_type to authed callers.
    // Read the token from storage directly: importing features/auth here
    // would be an import cycle (auth/session imports this store).
    const token =
      typeof window === "undefined"
        ? null
        : localStorage.getItem("unsloth_auth_token");
    const res = await fetch(apiUrl("/api/health"), {
      headers: token ? { Authorization: `Bearer ${token}` } : undefined,
    });
    if (res.ok) {
      const data = (await res.json()) as {
        device_type?: string;
        chat_only?: boolean;
        cloudflare_url?: string | null;
        server_url?: string | null;
      };
      const deviceType = data.device_type ?? detectLocalPlatform();
      const chatOnly = data.chat_only ?? false;
      // Cache only a server-reported platform. Unauthenticated responses fall
      // back to the browser platform, which can differ from the host (WSL,
      // SSH); keeping fetched=false retries once a token exists.
      usePlatformStore.setState({
        deviceType,
        chatOnly,
        cloudflareUrl: data.cloudflare_url ?? null,
        serverUrl: data.server_url ?? null,
        fetched: data.device_type !== undefined,
      });
      return deviceType;
    }
  } catch {
    // Backend not ready: use client-side detection so chat-only guard works
    // on initial load (important for macOS). Keep fetched=false so a later
    // call retries against the backend.
    const deviceType = detectLocalPlatform();
    const chatOnly = deviceType === "mac";
    usePlatformStore.setState({ deviceType, chatOnly, fetched: false });
    return deviceType;
  }

  return usePlatformStore.getState().deviceType;
}
