// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { getHfEndpoint } from "@/lib/hf-endpoint";
import { fetchHub } from "@/lib/hub-fetch";

const NETWORK_STATUS_EVENT = "unsloth-network-status";
const REMOTE_OFFLINE_TTL_MS = 30_000;
const HUGGING_FACE_ORIGIN = "https://huggingface.co";
const noopUnsubscribe = () => undefined;

/** Follows HF_ENDPOINT, since the backoff maps key on the request origin. */
function defaultHubOrigin(): string {
  try {
    return new URL(getHfEndpoint()).origin;
  } catch {
    return HUGGING_FACE_ORIGIN;
  }
}

type RemoteNetworkScope = string | readonly string[];

/**
 * Browsers collapse CORS, DNS, TLS and outages into one TypeError, so "network-opaque" is what
 * we can prove. "auth-rejected" never backs the origin off.
 */
export type HubFailureKind =
  | "aborted"
  | "timeout"
  | "browser-offline"
  | "network-opaque"
  | "auth-rejected"
  | "unknown";

export interface HubFailure {
  kind: HubFailureKind;
  /** Already sanitised: never contains a full URL or a token. */
  message: string;
  /** Origin only, never the full request URL (which carries the search query). */
  origin: string | null;
  status?: number;
  retryable: boolean;
}

export class HubFetchError extends Error {
  readonly failure: HubFailure;

  constructor(failure: HubFailure, options?: { cause?: unknown }) {
    super(failure.message, options);
    this.name = "HubFetchError";
    this.failure = failure;
  }
}

export function isHubFetchError(error: unknown): error is HubFetchError {
  return error instanceof HubFetchError;
}

const remoteOfflineUntilByOrigin = new Map<string, number>();
// Cleared only by a success, so the cause outlives the backoff window.
const lastFailureByOrigin = new Map<string, HubFailure>();

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
  // Uses the empirical TTL, not navigator.onLine; the earliest live window, not the latest.
  return Math.max(
    0,
    getEarliestRemoteOfflineUntil() - Date.now(),
  );
}

function normalizeScope(scope: RemoteNetworkScope): readonly string[] {
  return typeof scope === "string" ? [scope] : scope;
}

function offlineUntil(origin: string): number {
  const value = remoteOfflineUntilByOrigin.get(origin) ?? 0;
  if (value <= Date.now()) {
    remoteOfflineUntilByOrigin.delete(origin);
    return 0;
  }
  return value;
}

function getRemoteOfflineUntil(scope: RemoteNetworkScope): number {
  let until = 0;
  for (const origin of normalizeScope(scope)) {
    until = Math.max(until, offlineUntil(origin));
  }
  return until;
}

function getEarliestRemoteOfflineUntil(): number {
  const now = Date.now();
  let until = 0;
  for (const [origin, value] of remoteOfflineUntilByOrigin) {
    if (value <= now) {
      remoteOfflineUntilByOrigin.delete(origin);
      continue;
    }
    if (until === 0 || value < until) {
      until = value;
    }
  }
  return until;
}

export function isRemoteNetworkOffline(
  scope: RemoteNetworkScope = defaultHubOrigin(),
): boolean {
  return getRemoteOfflineUntil(scope) > Date.now();
}

export function isHuggingFaceOffline(): boolean {
  // navigator.onLine is advisory (false offline on WSL2 / some WebKitGTK webviews).
  return isRemoteNetworkOffline(defaultHubOrigin());
}

/** A lapsed backoff is "probing"; only a success promotes to "available". */
export type HubPhase = "available" | "probing" | "unavailable";

export function getHubPhase(origin: string = defaultHubOrigin()): HubPhase {
  if (!lastFailureByOrigin.has(origin)) {
    return "available";
  }
  return isRemoteNetworkOffline(origin) ? "unavailable" : "probing";
}

export function getLastHubFailure(
  origin: string = defaultHubOrigin(),
): HubFailure | null {
  return lastFailureByOrigin.get(origin) ?? null;
}

export function markRemoteNetworkOnline(origin?: string): void {
  if (origin === undefined) {
    if (
      remoteOfflineUntilByOrigin.size === 0 &&
      lastFailureByOrigin.size === 0
    ) {
      return;
    }
    remoteOfflineUntilByOrigin.clear();
    lastFailureByOrigin.clear();
    emitNetworkStatusChange();
    return;
  }
  const hadWindow = remoteOfflineUntilByOrigin.delete(origin);
  const hadFailure = lastFailureByOrigin.delete(origin);
  if (!hadWindow && !hadFailure) {
    return;
  }
  emitNetworkStatusChange();
}

export function markRemoteNetworkOffline(
  originOrTtl: string | number = defaultHubOrigin(),
  ttlMs = REMOTE_OFFLINE_TTL_MS,
  failure?: HubFailure,
): void {
  const origin =
    typeof originOrTtl === "string" ? originOrTtl : defaultHubOrigin();
  const ttl = typeof originOrTtl === "number" ? originOrTtl : ttlMs;
  const nextUntil = Date.now() + ttl;
  const previousUntil = remoteOfflineUntilByOrigin.get(origin) ?? 0;
  // The cause must describe the window in force; a first cause is always taken.
  const takesWindow = nextUntil > previousUntil;
  const records =
    failure !== undefined && (takesWindow || !lastFailureByOrigin.has(origin));
  const failureChanged =
    records && lastFailureByOrigin.get(origin)?.kind !== failure?.kind;
  if (records && failure !== undefined) {
    lastFailureByOrigin.set(origin, failure);
  }
  if (!takesWindow) {
    if (failureChanged) {
      emitNetworkStatusChange();
    }
    return;
  }
  remoteOfflineUntilByOrigin.set(origin, nextUntil);
  emitNetworkStatusChange();
}

export function clearRemoteBackoff(
  origin: string = defaultHubOrigin(),
): void {
  if (!remoteOfflineUntilByOrigin.delete(origin)) {
    return;
  }
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

function originFromFetchInput(
  input: Parameters<typeof fetch>[0],
): string | null {
  try {
    const raw =
      typeof input === "string"
        ? input
        : input instanceof URL
          ? input.toString()
          : input.url;
    const base =
      typeof window !== "undefined" ? window.location.href : "http://localhost";
    return new URL(raw, base).origin;
  } catch {
    return null;
  }
}

function hostLabel(origin: string | null): string {
  if (!origin) {
    return "Hugging Face";
  }
  try {
    return new URL(origin).host;
  } catch {
    return origin;
  }
}

/** Drops the request URL (search query, internal hostnames); only the host survives. */
export function classifyFetchFailure(
  error: unknown,
  origin: string | null,
  options: { timedOut?: boolean } = {},
): HubFailure {
  const host = hostLabel(origin);
  if (options.timedOut) {
    return {
      kind: "timeout",
      message: `The request to ${host} timed out.`,
      origin,
      retryable: true,
    };
  }
  if (isAbortError(error)) {
    return {
      kind: "aborted",
      message: "The request was cancelled.",
      origin,
      retryable: true,
    };
  }
  if (isNetworkFetchError(error)) {
    if (isNavigatorOffline()) {
      return {
        kind: "browser-offline",
        message: "This browser reports no network connection.",
        origin,
        retryable: true,
      };
    }
    return {
      kind: "network-opaque",
      message: `Unable to reach ${host}. Check your network connection.`,
      origin,
      retryable: true,
    };
  }
  return {
    kind: "unknown",
    message: `The request to ${host} failed.`,
    origin,
    retryable: true,
  };
}

// 403 is excluded on purpose: it means a gated/private repo, not a rejected token.
const HUB_TOKEN_REJECTED_RE =
  /invalid credentials|invalid (?:user )?(?:access )?token|oauth token verification failed|token (?:has )?(?:expired|been revoked)/i;

/** Null for anything but a refused token, including 403, 404, 429 and 5xx. */
export function hubAuthFailure(
  error: { status?: number | null; message?: string | null },
  origin: string | null = defaultHubOrigin(),
): HubFailure | null {
  const rejected =
    error.status === 401 ||
    (error.status == null && HUB_TOKEN_REJECTED_RE.test(error.message ?? ""));
  if (!rejected) {
    return null;
  }
  return {
    kind: "auth-rejected",
    message: `${hostLabel(origin)} refused the saved Hugging Face token. It may have expired or been revoked. Update or clear it in Settings, then try again.`,
    origin,
    status: 401,
    retryable: true,
  };
}

/** Strips the SDK's "URL: ... Request ID: ..." trailer, which leaks the query and host. */
export function sanitizeHubErrorMessage(message: string): string {
  if (!message) return message;
  const cleaned = message.replace(/\.?\s*URL:\s*\S+(\.\s*Request ID:\s*\S+)?\.?\s*$/, "");
  return cleaned.trim() || message;
}

export async function fetchWithTimeout(
  input: Parameters<typeof fetch>[0],
  init: Parameters<typeof fetch>[1] = {},
  timeoutMs = 15_000,
): Promise<Response> {
  const parentSignal = init.signal;
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

  const origin = originFromFetchInput(input);

  try {
    const response = await fetchHub(input, {
      ...init,
      signal: controller.signal,
    });
    if (origin) {
      markRemoteNetworkOnline(origin);
    }
    return response;
  } catch (error) {
    // A superseded query must not blacklist the Hub or overwrite a diagnosis.
    if (parentSignal?.aborted && !timedOut) {
      throw error;
    }
    const failure = classifyFetchFailure(error, origin, { timedOut });
    // Connectivity failures only: a slow optional asset timing out must not take the origin down.
    if (origin && !timedOut && isNetworkFetchError(error)) {
      markRemoteNetworkOffline(origin, REMOTE_OFFLINE_TTL_MS, failure);
    }
    throw new HubFetchError(failure, { cause: error });
  } finally {
    clearTimeout(timeout);
    parentSignal?.removeEventListener("abort", abortFromParent);
  }
}
