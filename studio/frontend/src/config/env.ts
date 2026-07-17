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
  // Why chatOnly is set (null when training is enabled), from /api/health.
  // e.g. "mlx_unavailable" on Apple Silicon -> the UI explains the greyed-out
  // Train/Export instead of silently disabling them.
  chatOnlyReason: string | null;
  // From /api/health (authed): live tunnel URL, direct (non-tunnel) base, and
  // whether the server was launched with --secure.
  cloudflareUrl: string | null;
  serverUrl: string | null;
  secure: boolean;
  fetched: boolean;
  isChatOnly: () => boolean;
}

// Browser-side detection. Used as a fallback when the backend isn't ready,
// and directly for client-keyboard concerns (shortcut labels), where the
// server-reported deviceType would be wrong for remote/tunnel sessions.
export function detectLocalPlatform(): DeviceType {
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
  chatOnlyReason: null,
  cloudflareUrl: null,
  serverUrl: null,
  secure: false,
  fetched: false,
  isChatOnly: () => get().chatOnly,
}));

// Once an authoritative (server-reported) platform has been fetched, a
// non-forced response must not overwrite it. The post-render fetchDeviceType()
// in main.tsx runs before auth is ready and can resolve after the authed
// root-route/provider fetches; such a late write would reset deviceType,
// cloudflareUrl/serverUrl/secure, and fetched, whether it is a browser fallback
// (unauthenticated) or an earlier authenticated request that landed after a
// later forced refresh. Forced refreshes are explicit re-reads, so they still write.
function shouldKeepAuthoritativePlatform(force?: boolean): boolean {
  return !force && usePlatformStore.getState().fetched;
}

// `force` re-reads /api/health even if cached, to pick up a late-arriving tunnel URL.
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
        chat_only_reason?: string | null;
        cloudflare_url?: string | null;
        server_url?: string | null;
        secure?: boolean;
      };
      // Once the store holds an authoritative (server-reported) platform, a
      // non-forced response must not overwrite it. It may be an unauthenticated
      // fallback, or an earlier authenticated request that resolved after a
      // later forced refresh already picked up device_type and the tunnel
      // fields; writing either would reset device type or null the tunnel
      // fields. Forced refreshes are explicit re-reads, so they still write.
      if (shouldKeepAuthoritativePlatform(options?.force)) {
        return usePlatformStore.getState().deviceType;
      }
      const deviceType = data.device_type ?? detectLocalPlatform();
      const chatOnly = data.chat_only ?? false;
      const chatOnlyReason = data.chat_only_reason ?? null;
      // Cache only a server-reported platform. Unauthenticated responses fall
      // back to the browser platform, which can differ from the host (WSL,
      // SSH); keeping fetched=false retries once a token exists.
      usePlatformStore.setState({
        deviceType,
        chatOnly,
        chatOnlyReason,
        cloudflareUrl: data.cloudflare_url ?? null,
        serverUrl: data.server_url ?? null,
        secure: data.secure ?? false,
        fetched: data.device_type !== undefined,
      });
      return deviceType;
    }
  } catch {
    // Backend not ready: use client-side detection so chat-only guard works
    // on initial load (important for macOS). Keep fetched=false so a later
    // call retries against the backend. But a late non-forced failure must not
    // wipe an authoritative platform that already resolved.
    if (shouldKeepAuthoritativePlatform(options?.force)) {
      return usePlatformStore.getState().deviceType;
    }
    const deviceType = detectLocalPlatform();
    const chatOnly = deviceType === "mac";
    usePlatformStore.setState({ deviceType, chatOnly, fetched: false });
    return deviceType;
  }

  return usePlatformStore.getState().deviceType;
}
