// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Hub fetch: the adapter and relays take the Unsloth session; a relay forwards the HF token in `X-HF-Authorization`. */

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

// Set by the relay on the endpoint's own answers, whose 401 is about the Hugging Face token.
const UPSTREAM_HEADER = "X-Hub-Upstream";

function requestUrl(input: Parameters<typeof fetch>[0]): string {
  return typeof input === "string"
    ? input
    : input instanceof URL
      ? input.href
      : input.url;
}

/** The headers this call sends: as fetch does, `init.headers` replaces a Request input's own. */
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

/** The Hugging Face token this request would send the Hub, or null. ModelScope is excluded:
 * its relay takes the Unsloth session and drops the HF token. */
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

/** Explicit headers, so a Request input's own Authorization cannot come back later. */
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

/** A 401 that came from the Hub itself: direct, or the relay passing on the endpoint's answer. */
function isHubRefusal(response: Response, url: string): boolean {
  if (response.status !== 401) return false;
  return isProxiedHubUrl(url) ? response.headers.has(UPSTREAM_HEADER) : !takesSession(url);
}

/** The Hub read the token and answered: a success, or a 403/404 for the resource (a token the
 * Hub refuses gets 401). Through the relay only the endpoint's own answer counts. */
function tokenAccepted(response: Response, url: string): boolean {
  if (isProxiedHubUrl(url) && !response.headers.has(UPSTREAM_HEADER) && !response.ok) return false;
  return response.ok || response.status === 403 || response.status === 404;
}

/** Which Hub a token refusal belongs to: the endpoint or datasets server the request went to
 * (its origin for any other URL), so a refusal by one never skips the token on another. */
export function hubRejectionScope(url?: string): string {
  const endpoint = getHfEndpoint();
  let target = endpoint;
  if (url !== undefined) {
    // The most specific base first: a datasets server can live under the model endpoint.
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

/** A second, optional attempt: a network error or timeout there must not replace the answer
 * the first request already got. */
async function probe(
  input: Parameters<typeof fetch>[0],
  init: RequestInit,
  url: string,
): Promise<Response | null> {
  try {
    return await fetchWithSession(input, init, url);
  } catch {
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
  // Already refused this session: every read with it would 401 again, so ask anonymously.
  const scope = hubRejectionScope(url);
  const skipToken = retryable && isHfTokenRejected(hfToken, scope);
  const started = hfTokenRejectionMark();
  let response = await fetchWithSession(input, skipToken ? withoutHfToken(input, init) : init, url);
  if (skipToken && !response.ok && [401, 403, 404].includes(response.status)) {
    // What anonymous access cannot read may still be the token's to read: the refusal could
    // have been the Hub's verifier briefly failing. A token that answers again is cleared.
    const withToken = await probe(input, init, url);
    if (withToken) {
      // Still refused, the token's own answer (a 401) is the result, as on the first refusal:
      // an anonymous 404 for a private repo would be cached as the repo being missing.
      void response.body?.cancel().catch(() => undefined);
      if (tokenAccepted(withToken, url)) clearHfTokenRejected(scope);
      response = withToken;
    }
  } else if (retryable && !skipToken && tokenAccepted(response, url)) {
    // Past the recheck window the token went out again and was accepted: that Hub no
    // longer refuses it.
    clearHfTokenRejected(scope);
  } else if (retryable && !skipToken && isHubRefusal(response, url)) {
    // A token the Hub accepts gets 404 for a repo it cannot see, so this 401 is the token
    // being refused. Public data still answers without it; anything else keeps the original.
    const anonymous = await probe(input, withoutHfToken(input, init), url);
    if (anonymous?.ok) {
      void response.body?.cancel().catch(() => undefined);
      noteHfTokenRejected(hfToken, scope, started);
      response = anonymous;
    } else {
      void anonymous?.body?.cancel().catch(() => undefined);
    }
  }
  // The relay's own 502 means it could not reach the endpoint: fail as a direct fetch would.
  if (response.status === 502 && !response.headers.has(UPSTREAM_HEADER) && isProxiedHubUrl(url)) {
    throw new TypeError("The Hub endpoint could not be reached.");
  }
  return response;
}

export const hubFetch: typeof fetch = (input, init) =>
  fetchHub(input, init ?? {});
