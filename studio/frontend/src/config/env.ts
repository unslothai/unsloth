// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { apiUrl } from "@/lib/api-base";
import { setHfEndpoints } from "@/lib/hf-endpoint";
import {
  isDetectionDeferred,
  isProvisionalVerdict,
  resolveVerdict,
} from "@/config/hardware-verdict";
import { create } from "zustand";

// Backend routes, so same origin: the page CSP already allows them whatever endpoint is saved later.
const MODELSCOPE_HUB_PATH = "/api/hub/modelscope";

function backendUrl(path: string): string {
  return new URL(apiUrl(path), window.location.href).href;
}

export const env = {
  MODE: import.meta.env.MODE,
  DEV: import.meta.env.DEV,
  PROD: import.meta.env.PROD,
  BASE_URL: import.meta.env.BASE_URL,
} as const;

export type DeviceType = "mac" | "windows" | "linux" | string;

export type FileManager = "finder" | "explorer" | "files" | null;

interface PlatformState {
  deviceType: DeviceType;
  // Unified memory: an over-committed load takes the machine down. Narrower than deviceType
  // "mac" (Intel Macs spill to RAM). Mirrors the backend's is_apple_silicon gate.
  appleSilicon: boolean;
  fileManager: FileManager | undefined;
  chatOnly: boolean;
  // /api/health reason for chatOnly; null while training is enabled.
  chatOnlyReason: string | null;
  // MLX gate covers mlx, mlx-lm, and mlx-vlm together; record the blocker when known.
  chatOnlyDetail: string | null;
  // authenticated /api/health values for cloudflareUrl, serverUrl, and secure.
  cloudflareUrl: string | null;
  serverUrl: string | null;
  secure: boolean;
  lanUrls: string[];
  fetched: boolean;
  // torch-warm kill switch defers detection to first use; the sidebar polls.
  detectionDeferred: boolean;
  isChatOnly: () => boolean;
  // true until /api/health returns a verdict; guesses cannot gate features.
  capabilitiesUnknown: () => boolean;
}

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
  appleSilicon: false,
  fileManager: undefined,
  // user-agent guess for redirects; capability gates await the server.
  chatOnly: localDeviceType === "mac",
  chatOnlyReason: null,
  chatOnlyDetail: null,
  cloudflareUrl: null,
  serverUrl: null,
  secure: false,
  lanUrls: [],
  fetched: false,
  detectionDeferred: false,
  isChatOnly: () => get().chatOnly,
  // deferred detection is settled because no verdict will arrive this session.
  capabilitiesUnknown: () => {
    const state = get();
    return !state.fetched && !state.detectionDeferred;
  },
}));

// Never let a non-forced response overwrite an authoritative platform: the early unauthed fetch
// in main.tsx can resolve late. Forced refreshes still write.
function shouldKeepAuthoritativePlatform(force?: boolean): boolean {
  return !force && usePlatformStore.getState().fetched;
}

// Wait for a still-detecting backend, sized from the warm torch import (~1-2s) plus headroom.
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
    // device_type is only reported to authed callers. Read the token directly: importing
    // features/auth would cycle.
    const token =
      typeof window === "undefined"
        ? null
        : localStorage.getItem("unsloth_auth_token");
    // Re-read while the backend is still detecting: chat_only is its pre-detection default and
    // __root.tsx's beforeLoad acts on it.
    const headers = token ? { Authorization: `Bearer ${token}` } : undefined;
    // Wait only when authed (else beforeLoad stalls /login), and only once. Claim the latch after.
    const spendWait = Boolean(token) && !hardwareWaitSpent;
    const deadline = spendWait ? Date.now() + HARDWARE_DETECT_WAIT_MS : 0;
    let tokenRejected = false;
    let res = await fetch(apiUrl("/api/health"), { headers });
    while (res.ok && Date.now() < deadline) {
      const peek = (await res.clone().json()) as {
        hardware_detecting?: boolean;
        hardware_detection_deferred?: boolean;
        version?: string;
      };
      // Deferred is not "in progress": nothing will settle, so do not wait.
      if (!isProvisionalVerdict(peek) || isDetectionDeferred(peek)) break;
      // A rejected token gets the unauthed body: no `version`, so the wait could only time out.
      if (peek.version === undefined) {
        tokenRejected = true;
        break;
      }
      await new Promise((resolve) => setTimeout(resolve, HARDWARE_DETECT_POLL_MS));
      res = await fetch(apiUrl("/api/health"), { headers });
    }
    // Not spent on a rejected token, so a later sign-in in this page load still gets the wait.
    if (spendWait && !tokenRejected) hardwareWaitSpent = true;
    if (res.ok) {
      const data = (await res.json()) as {
        device_type?: string;
        apple_silicon?: boolean;
        file_manager?: FileManager;
        chat_only?: boolean;
        chat_only_reason?: string | null;
        hardware_detecting?: boolean;
        cloudflare_url?: string | null;
        server_url?: string | null;
        secure?: boolean;
        hf_endpoint?: string;
        hf_datasets_server?: string;
        hub_source?: string;
        hub_proxy?: string | null;
        datasets_server_proxy?: string | null;
      };
      // Before the authoritative-platform guard: hub_proxy is unauthed and idempotent, and a mirror
      // whose first authoritative reply already landed would otherwise never route Hub calls.
      const hubProxy = typeof data.hub_proxy === "string" ? data.hub_proxy : null;
      const datasetsProxy =
        typeof data.datasets_server_proxy === "string" ? data.datasets_server_proxy : null;
      setHfEndpoints(
        data.hub_source === "modelscope"
          ? backendUrl(MODELSCOPE_HUB_PATH)
          : hubProxy
            ? backendUrl(hubProxy)
            : data.hf_endpoint,
        datasetsProxy ? backendUrl(datasetsProxy) : data.hf_datasets_server,
        data.hub_source,
        { endpoint: hubProxy !== null, datasetsServer: datasetsProxy !== null },
      );
      if (shouldKeepAuthoritativePlatform(options?.force)) {
        return usePlatformStore.getState().deviceType;
      }
      const previous = usePlatformStore.getState();
      // A provisional reply omits device_type; keep the server's answer rather than relabel a remote host.
      const keepPlatform = data.device_type === undefined && previous.fetched;
      const deviceType =
        data.device_type ?? (keepPlatform ? previous.deviceType : detectLocalPlatform());
      // Kept on the same terms as device_type; absent means false (correct on Intel Macs).
      const appleSilicon =
        data.apple_silicon ?? (keepPlatform ? previous.appleSilicon : false);
      const fileManager =
        data.device_type !== undefined
          ? data.file_manager
          : keepPlatform
            ? previous.fileManager
            : undefined;
      // A still-provisional reply keeps the stored verdict: see resolveVerdict.
      const { chatOnly, chatOnlyReason, chatOnlyDetail } = resolveVerdict(
        data,
        previous,
      );
      // Cache only a server-reported platform; fetched=false retries once a token exists.
      usePlatformStore.setState({
        deviceType,
        appleSilicon,
        fileManager,
        chatOnly,
        chatOnlyReason,
        chatOnlyDetail,
        cloudflareUrl: data.cloudflare_url ?? null,
        serverUrl: data.server_url ?? null,
        secure: data.secure ?? false,
        fetched: data.device_type !== undefined || keepPlatform,
        detectionDeferred: isDetectionDeferred(data),
      });
      return deviceType;
    }
  } catch {
    // Backend not ready: use client detection, keep fetched=false, but never wipe an authoritative one.
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
