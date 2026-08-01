// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { apiUrl } from "@/lib/api-base";
import {
  isDetectionDeferred,
  isProvisionalVerdict,
  resolveVerdict,
} from "@/config/hardware-verdict";
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

// How long fetchDeviceType waits out a backend that is still detecting, and how often
// it re-reads. Sized from the warm's torch import (~1-2s cold), plus headroom.
const HARDWARE_DETECT_WAIT_MS = 5000;
const HARDWARE_DETECT_POLL_MS = 200;
// The bounded wait above is spent at most once per page load: see fetchDeviceType.
let hardwareWaitSpent = false;

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
    // Re-read while the backend is still measuring: chat_only is its pre-detection
    // default until then, and __root.tsx's beforeLoad acts on what this returns,
    // sending a GPU host to /chat with Train hidden. The window is only the torch
    // import, so a bounded re-read lands the measurement without stalling boot.
    const headers = token ? { Authorization: `Bearer ${token}` } : undefined;
    // Once per load: a slow host leaves the reply provisional and `fetched` false, so
    // without this every later navigation would spend the whole window on the same answer.
    const deadline = hardwareWaitSpent
      ? 0
      : Date.now() + HARDWARE_DETECT_WAIT_MS;
    let res = await fetch(apiUrl("/api/health"), { headers });
    while (res.ok && Date.now() < deadline) {
      hardwareWaitSpent = true;
      const peek = (await res.clone().json()) as {
        hardware_detecting?: boolean;
        hardware_detection_deferred?: boolean;
      };
      // Deferred is not "in progress": nothing will settle, so do not wait.
      if (!isProvisionalVerdict(peek) || isDetectionDeferred(peek)) break;
      await new Promise((resolve) => setTimeout(resolve, HARDWARE_DETECT_POLL_MS));
      res = await fetch(apiUrl("/api/health"), { headers });
    }
    if (res.ok) {
      const data = (await res.json()) as {
        device_type?: string;
        chat_only?: boolean;
        chat_only_reason?: string | null;
        hardware_detecting?: boolean;
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
      const previous = usePlatformStore.getState();
      // A provisional reply omits device_type, so a forced refresh during the warm would
      // fall back to the browser platform and relabel a WSL, SSH or remote host as local,
      // changing model filtering, paths and install commands. Keep the server's answer.
      const keepPlatform = data.device_type === undefined && previous.fetched;
      const deviceType =
        data.device_type ?? (keepPlatform ? previous.deviceType : detectLocalPlatform());
      // A still-provisional reply keeps the stored verdict: see resolveVerdict.
      const { chatOnly, chatOnlyReason } = resolveVerdict(data, previous);
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
        fetched: data.device_type !== undefined || keepPlatform,
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
