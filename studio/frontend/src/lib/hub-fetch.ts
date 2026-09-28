// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Hub fetch: the adapter and relays take the Unsloth session; a relay forwards the HF token in `X-HF-Authorization`. */

import {
  isModelScopeHubUrl,
  isProxiedHubUrl,
  refreshHubSession,
} from "@/lib/hf-endpoint";
import {
  clearHfTokenRejected,
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

function takesSession(url: string): boolean {
  return isModelScopeHubUrl(url) || isProxiedHubUrl(url);
}

function withHubAuth(
  input: Parameters<typeof fetch>[0],
  init: RequestInit,
): RequestInit {
  const url = requestUrl(input);
  if (!takesSession(url)) return init;
  const headers = new Headers(init.headers);
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
function sentHfToken(url: string, init: RequestInit): string | null {
  if (isModelScopeHubUrl(url)) return null;
  const header = new Headers(init.headers).get("Authorization");
  const token = header?.replace(/^Bearer\s+/i, "").trim();
  return token || null;
}

function withoutHfToken(init: RequestInit): RequestInit {
  const headers = new Headers(init.headers);
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

export async function fetchHub(
  input: Parameters<typeof fetch>[0],
  init: RequestInit = {},
): Promise<Response> {
  const url = requestUrl(input);
  const hfToken = sentHfToken(url, init);
  const retryable = hfToken !== null && isRetryableRead(input, init);
  // Already refused this session: every read with it would 401 again, so ask anonymously.
  const skipToken = retryable && isHfTokenRejected(hfToken);
  let response = await fetchWithSession(input, skipToken ? withoutHfToken(init) : init, url);
  if (skipToken && !response.ok && [401, 403, 404].includes(response.status)) {
    // What anonymous access cannot read may still be the token's to read: the refusal could
    // have been the Hub's verifier briefly failing. A token that answers again is cleared.
    const withToken = await fetchWithSession(input, init, url);
    if (withToken.ok) {
      void response.body?.cancel().catch(() => undefined);
      clearHfTokenRejected();
      response = withToken;
    } else {
      void withToken.body?.cancel().catch(() => undefined);
    }
  } else if (retryable && !skipToken && isHubRefusal(response, url)) {
    // A token the Hub accepts gets 404 for a repo it cannot see, so this 401 is the token
    // being refused. Public data still answers without it; anything else keeps the original.
    const anonymous = await fetchWithSession(input, withoutHfToken(init), url);
    if (anonymous.ok) {
      void response.body?.cancel().catch(() => undefined);
      noteHfTokenRejected(hfToken);
      response = anonymous;
    } else {
      void anonymous.body?.cancel().catch(() => undefined);
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
