// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { apiUrl } from "@/lib/api-base";
import { HUB_HF_TOKEN_HEADER } from "./hub-token-header";

// AUTH_TOKEN_KEY from features/auth/session, read directly from localStorage
// so this network-layer module stays importable without the auth/store stack.
const AUTH_TOKEN_STORAGE_KEY = "unsloth_auth_token";

function getSessionToken(): string | null {
  if (typeof localStorage === "undefined") {
    return null;
  }
  return localStorage.getItem(AUTH_TOKEN_STORAGE_KEY);
}

export function isBrowserOffline(): boolean {
  return isNavigatorOffline();
}

const NETWORK_STATUS_EVENT = "unsloth-network-status";
const REMOTE_OFFLINE_TTL_MS = 30_000;
// Discovery and repository pages are served by the main Hugging Face origin.
// Keep optional services such as datasets-server separate so an outage there
// cannot make the whole Hub appear offline.
const HUGGING_FACE_ORIGIN = "https://huggingface.co";
const noopUnsubscribe = () => undefined;

type RemoteNetworkScope = string | readonly string[];

const remoteOfflineUntilByOrigin = new Map<string, number>();

function isNavigatorOffline(): boolean {
  return typeof navigator !== "undefined" && navigator.onLine === false;
}

function emitNetworkStatusChange(): void {
  if (typeof window === "undefined") {
    return;
  }
  window.dispatchEvent(new Event(NETWORK_STATUS_EVENT));
}

export function getBrowserOfflineRetryDelayMs(): number {
  // Keyed off the empirical remote-offline TTL, not navigator.onLine, so
  // recovery doesn't stall on platforms where navigator.onLine is stuck false.
  return Math.max(
    0,
    getRemoteOfflineUntil(HUGGING_FACE_ORIGIN) - Date.now(),
  );
}

function normalizeScope(scope: RemoteNetworkScope): readonly string[] {
  return typeof scope === "string" ? [scope] : scope;
}

function getRemoteOfflineUntil(scope: RemoteNetworkScope): number {
  const now = Date.now();
  let until = 0;
  for (const origin of normalizeScope(scope)) {
    const value = remoteOfflineUntilByOrigin.get(origin) ?? 0;
    if (value <= now) {
      remoteOfflineUntilByOrigin.delete(origin);
      continue;
    }
    until = Math.max(until, value);
  }
  return until;
}

export function isRemoteNetworkOffline(
  scope: RemoteNetworkScope = HUGGING_FACE_ORIGIN,
): boolean {
  return getRemoteOfflineUntil(scope) > Date.now();
}

export function isHuggingFaceOffline(): boolean {
  // navigator.onLine is advisory only (false-reports offline on WSL2 / some
  // WebKitGTK/Tauri webviews). The authoritative signal is the empirical
  // remote-offline TTL, set when a real fetch fails and cleared on next success;
  // navigator's online/offline events still drive re-evaluation.
  return isRemoteNetworkOffline(HUGGING_FACE_ORIGIN);
}

export function markRemoteNetworkOnline(origin?: string): void {
  if (origin === undefined) {
    if (remoteOfflineUntilByOrigin.size === 0) {
      return;
    }
    remoteOfflineUntilByOrigin.clear();
    emitNetworkStatusChange();
    return;
  }
  if (!remoteOfflineUntilByOrigin.delete(origin)) {
    return;
  }
  emitNetworkStatusChange();
}

export function markRemoteNetworkOffline(
  originOrTtl: string | number = HUGGING_FACE_ORIGIN,
  ttlMs = REMOTE_OFFLINE_TTL_MS,
): void {
  const origin =
    typeof originOrTtl === "string"
      ? originOrTtl
      : HUGGING_FACE_ORIGIN;
  const ttl = typeof originOrTtl === "number" ? originOrTtl : ttlMs;
  const nextUntil = Date.now() + ttl;
  if (nextUntil <= (remoteOfflineUntilByOrigin.get(origin) ?? 0)) {
    return;
  }
  remoteOfflineUntilByOrigin.set(origin, nextUntil);
  emitNetworkStatusChange();
}

export function subscribeNetworkStatus(listener: () => void): () => void {
  if (typeof window === "undefined") {
    return noopUnsubscribe;
  }
  window.addEventListener("online", listener);
  window.addEventListener("offline", listener);
  window.addEventListener(NETWORK_STATUS_EVENT, listener);
  return () => {
    window.removeEventListener("online", listener);
    window.removeEventListener("offline", listener);
    window.removeEventListener(NETWORK_STATUS_EVENT, listener);
  };
}

function isAbortError(error: unknown): boolean {
  return error instanceof DOMException && error.name === "AbortError";
}

function isNetworkFetchError(error: unknown): boolean {
  if (isAbortError(error)) {
    return false;
  }
  return error instanceof TypeError;
}

function rawUrlFromFetchInput(input: Parameters<typeof fetch>[0]): string {
  return typeof input === "string"
    ? input
    : input instanceof URL
      ? input.toString()
      : input.url;
}

function originFromFetchInput(
  input: Parameters<typeof fetch>[0],
): string | null {
  try {
    const base =
      typeof window !== "undefined" ? window.location.href : "http://localhost";
    return new URL(rawUrlFromFetchInput(input), base).origin;
  } catch {
    return null;
  }
}

class RequestTimeoutError extends Error {
  constructor() {
    super("Request timed out");
    this.name = "RequestTimeoutError";
  }
}

/** Timeout/abort wiring shared by the direct and proxied fetch paths. */
async function runFetchWithTimeout(
  input: Parameters<typeof fetch>[0],
  init: Parameters<typeof fetch>[1],
  timeoutMs: number,
): Promise<Response> {
  const parentSignal = init?.signal;
  const controller = new AbortController();
  let timedOut = false;
  const timeout = setTimeout(() => {
    timedOut = true;
    controller.abort();
  }, timeoutMs);
  const abortFromParent = () => controller.abort();

  if (parentSignal?.aborted) {
    abortFromParent();
  } else {
    parentSignal?.addEventListener("abort", abortFromParent, { once: true });
  }

  try {
    return await fetch(input, { ...init, signal: controller.signal });
  } catch (error) {
    if (timedOut) {
      throw new RequestTimeoutError();
    }
    throw error;
  } finally {
    clearTimeout(timeout);
    parentSignal?.removeEventListener("abort", abortFromParent);
  }
}

// Browser-level privacy tooling (DNS filtering, TLS-inspecting firewalls,
// tracker blockers) can block direct fetches to huggingface.co while the
// backend still has connectivity. When a direct hub fetch fails, retry it
// through the backend's read-only /api/hub/hf-proxy passthrough before
// declaring the hub offline.
const HF_PROXY_PATH = "/api/hub/hf-proxy";
const PROXYABLE_ORIGINS: ReadonlySet<string> = new Set([
  HUGGING_FACE_ORIGIN,
  "https://datasets-server.huggingface.co",
]);
// After a successful fallback, go proxy-first for a while so every listing
// page doesn't pay for a doomed direct attempt plus its timeout.
const PROXY_PREFER_TTL_MS = 10 * 60_000;

let preferProxyUntil = 0;

export function __resetHfProxyPreferenceForTests(): void {
  preferProxyUntil = 0;
}

function isProxyableRequest(
  input: Parameters<typeof fetch>[0],
  init: Parameters<typeof fetch>[1],
): boolean {
  const method = (
    init?.method ??
    (typeof input === "object" && "method" in input ? input.method : "GET")
  ).toUpperCase();
  return method === "GET";
}

/**
 * Re-issue a Hugging Face request through the backend passthrough. The HF
 * bearer token (if any) moves to the internal token header; Authorization is
 * replaced with the studio session token the backend routes require.
 */
function fetchViaBackendProxy(
  input: Parameters<typeof fetch>[0],
  init: Parameters<typeof fetch>[1],
  timeoutMs: number,
): Promise<Response> {
  const sourceHeaders = init?.headers
    ? new Headers(init.headers)
    : input instanceof Request
      ? input.headers
      : undefined;
  const headers = new Headers();
  const hfAuthorization = sourceHeaders?.get("authorization");
  if (hfAuthorization?.toLowerCase().startsWith("bearer ")) {
    headers.set(HUB_HF_TOKEN_HEADER, hfAuthorization.slice("bearer ".length));
  }
  const sessionToken = getSessionToken();
  if (sessionToken) {
    headers.set("Authorization", `Bearer ${sessionToken}`);
  }
  const proxyUrl = apiUrl(
    `${HF_PROXY_PATH}?url=${encodeURIComponent(rawUrlFromFetchInput(input))}`,
  );
  return runFetchWithTimeout(
    proxyUrl,
    { method: "GET", headers, signal: init?.signal },
    timeoutMs,
  );
}

export async function fetchWithTimeout(
  input: Parameters<typeof fetch>[0],
  init: Parameters<typeof fetch>[1] = {},
  timeoutMs = 15_000,
): Promise<Response> {
  const origin = originFromFetchInput(input);
  const canProxy =
    origin !== null &&
    PROXYABLE_ORIGINS.has(origin) &&
    isProxyableRequest(input, init);

  if (canProxy && preferProxyUntil > Date.now()) {
    try {
      const response = await fetchViaBackendProxy(input, init, timeoutMs);
      markRemoteNetworkOnline(origin);
      return response;
    } catch (error) {
      if (init.signal?.aborted) {
        throw error;
      }
      // proxy stopped working; retry direct below and re-arm only on a
      // successful future fallback.
      preferProxyUntil = 0;
    }
  }

  try {
    const response = await runFetchWithTimeout(input, init, timeoutMs);
    if (origin) {
      markRemoteNetworkOnline(origin);
    }
    return response;
  } catch (error) {
    const timedOut = error instanceof RequestTimeoutError;
    if (canProxy && (timedOut || isNetworkFetchError(error))) {
      try {
        const response = await fetchViaBackendProxy(input, init, timeoutMs);
        preferProxyUntil = Date.now() + PROXY_PREFER_TTL_MS;
        markRemoteNetworkOnline(origin);
        return response;
      } catch {
        // both paths failed; report the direct failure below.
      }
    }
    if (timedOut) {
      throw error;
    }
    if (origin && isNetworkFetchError(error)) {
      markRemoteNetworkOffline(origin);
    }
    throw error;
  }
}
