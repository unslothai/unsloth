// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * `fetch` for Hub and datasets-server calls. The backend's ModelScope adapter and its
 * relays to a custom endpoint take the Unsloth session; a relay passes the Hugging Face
 * token on in `X-HF-Authorization`, and the adapter never sees it. Reads the session from
 * storage, as config/env.ts does: importing features/auth would pull in more than the
 * bare-node tests can load.
 */

import {
  isModelScopeHubUrl,
  isProxiedHubUrl,
  refreshHubSession,
} from "@/lib/hf-endpoint";

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

// The session token lives an hour and is otherwise refreshed only by authFetch.
export async function fetchHub(
  input: Parameters<typeof fetch>[0],
  init: RequestInit = {},
): Promise<Response> {
  const response = await fetch(input, withHubAuth(input, init));
  if (
    response.status !== 401 ||
    response.headers.has(UPSTREAM_HEADER) ||
    !takesSession(requestUrl(input)) ||
    !(await refreshHubSession())
  ) {
    return response;
  }
  return fetch(input, withHubAuth(input, init));
}

/** `fetch` for Hub SDK calls that take no timeout. */
export const hubFetch: typeof fetch = (input, init) =>
  fetchHub(input, init ?? {});
