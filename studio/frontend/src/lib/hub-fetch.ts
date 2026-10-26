// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** The adapter and relays take the Unsloth session; a relay forwards the HF token in
 * `X-HF-Authorization`. */

import {
  getHfDatasetsServerBase,
  getHfEndpoint,
  getHubSource,
  isModelScopeHubUrl,
  isProxiedHubUrl,
  refreshHubSession,
} from "@/lib/hf-endpoint";
import {
  clearHfTokenRejected,
  hfTokenRejectionMark,
  isHfTokenRejected,
  noteHfTokenRejected,
} from "@/lib/hf-token-rejection";

// Set by the relay on the endpoint's own answers, whose 401 is about the HF token.
const UPSTREAM_HEADER = "X-Hub-Upstream";

function requestUrl(input: Parameters<typeof fetch>[0]): string {
  return typeof input === "string"
    ? input
    : input instanceof URL
      ? input.href
      : input.url;
}

/** As fetch does, `init.headers` replaces a Request input's own. */
function requestHeaders(input: Parameters<typeof fetch>[0], init: RequestInit): Headers {
  if (init.headers !== undefined) return new Headers(init.headers);
  return new Headers(
    typeof Request !== "undefined" && input instanceof Request ? input.headers : undefined,
  );
}

function takesSession(url: string): boolean {
  return isModelScopeHubUrl(url) || isProxiedHubUrl(url);
}

function withHubAuth(
  input: Parameters<typeof fetch>[0],
  init: RequestInit,
): RequestInit {
  const url = requestUrl(input);
  if (!takesSession(url)) return init;
  const headers = requestHeaders(input, init);
  const hfToken = headers.get("Authorization");
  if (hfToken && isProxiedHubUrl(url)) headers.set("X-HF-Authorization", hfToken);
  let token: string | null = null;
  try {
    token = localStorage.getItem("unsloth_auth_token");
  } catch {
    token = null;
  }
  if (token) headers.set("Authorization", `Bearer ${token}`);
  else headers.delete("Authorization");
  return { ...init, headers };
}

/** ModelScope's relay drops the HF token, so it is excluded. */
function sentHfToken(
  input: Parameters<typeof fetch>[0],
  init: RequestInit,
  url: string,
): string | null {
  if (isModelScopeHubUrl(url)) return null;
  const header = requestHeaders(input, init).get("Authorization");
  const token = header?.replace(/^Bearer\s+/i, "").trim();
  return token || null;
}

/** Explicit headers, so a Request input's Authorization cannot come back later. */
function withoutHfToken(input: Parameters<typeof fetch>[0], init: RequestInit): RequestInit {
  const headers = requestHeaders(input, init);
  headers.delete("Authorization");
  return { ...init, headers };
}

function isRetryableRead(input: Parameters<typeof fetch>[0], init: RequestInit): boolean {
  const method = (
    init.method ?? (typeof input === "object" && "method" in input ? input.method : "GET")
  ).toUpperCase();
  return method === "GET" || method === "HEAD";
}

async function fetchWithSession(
  input: Parameters<typeof fetch>[0],
  init: RequestInit,
  url: string,
): Promise<Response> {
  let response = await fetch(input, withHubAuth(input, init));
  if (
    response.status === 401 &&
    !response.headers.has(UPSTREAM_HEADER) &&
    takesSession(url) &&
    (await refreshHubSession())
  ) {
    response = await fetch(input, withHubAuth(input, init));
  }
  return response;
}

function isHubRefusal(response: Response, url: string): boolean {
  if (response.status !== 401) return false;
  return isProxiedHubUrl(url) ? response.headers.has(UPSTREAM_HEADER) : !takesSession(url);
}

/** A 403/404 still means the token was read; a refused token gets 401. */
function tokenAccepted(response: Response, url: string): boolean {
  if (isProxiedHubUrl(url) && !response.headers.has(UPSTREAM_HEADER) && !response.ok) return false;
  return response.ok || response.status === 403 || response.status === 404;
}

/** Per endpoint or datasets server, so one Hub's refusal never skips the token on another. */
export function hubRejectionScope(url?: string): string {
  const endpoint = getHfEndpoint();
  let target = endpoint;
  if (url !== undefined) {
    // Most specific first: a datasets server can live under the model endpoint.
    target =
      [endpoint, getHfDatasetsServerBase()]
        .sort((a, b) => b.length - a.length)
        .find((base) => url === base || url.startsWith(`${base.replace(/\/+$/, "")}/`)) ??
      urlOrigin(url);
  }
  return `${getHubSource()}|${target}`;
}

function urlOrigin(url: string): string {
  try {
    return new URL(url, globalThis.location?.href).origin;
  } catch {
    return url;
  }
}

/** Optional: its network error must not replace the first answer. */
async function probe(
  input: Parameters<typeof fetch>[0],
  init: RequestInit,
  url: string,
): Promise<Response | null> {
  try {
    return await fetchWithSession(input, init, url);
  } catch (error) {
    // The caller's abort or timeout is the answer, not a failed probe.
    const signal = init.signal ?? (input instanceof Request ? input.signal : undefined);
    if (signal?.aborted || (error instanceof Error && error.name === "AbortError")) {
      throw error;
    }
    return null;
  }
}

export async function fetchHub(
  input: Parameters<typeof fetch>[0],
  init: RequestInit = {},
): Promise<Response> {
  const url = requestUrl(input);
  const hfToken = sentHfToken(input, init, url);
  const retryable = hfToken !== null && isRetryableRead(input, init);
  // Already refused this session, so ask anonymously.
  const scope = hubRejectionScope(url);
  const skipToken = retryable && isHfTokenRejected(hfToken, scope);
  const started = hfTokenRejectionMark();
  let response = await fetchWithSession(input, skipToken ? withoutHfToken(input, init) : init, url);
  if (skipToken && !response.ok && [401, 403, 404].includes(response.status)) {
    // Retry the token in case the refusal was transient.
    const withToken = await probe(input, init, url);
    if (withToken) {
      // Keep the token's answer: an anonymous 404 would be cached as a missing repo.
      void response.body?.cancel().catch(() => undefined);
      if (tokenAccepted(withToken, url)) clearHfTokenRejected(scope);
      response = withToken;
    }
  } else if (retryable && !skipToken && tokenAccepted(response, url)) {
    clearHfTokenRejected(scope);
  } else if (retryable && !skipToken && isHubRefusal(response, url)) {
    // A valid token gets 404 for a hidden repo, so this 401 is a refused token.
    const anonymous = await probe(input, withoutHfToken(input, init), url);
    if (anonymous?.ok) {
      void response.body?.cancel().catch(() => undefined);
      noteHfTokenRejected(hfToken, scope, started);
      response = anonymous;
    } else {
      void anonymous?.body?.cancel().catch(() => undefined);
    }
  }
  // The relay's own 502 means it could not reach the endpoint.
  if (response.status === 502 && !response.headers.has(UPSTREAM_HEADER) && isProxiedHubUrl(url)) {
    throw new TypeError("The Hub endpoint could not be reached.");
  }
  return response;
}

export const hubFetch: typeof fetch = (input, init) =>
  fetchHub(input, init ?? {});
