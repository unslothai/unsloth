// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { fetchWithTimeout, isHubFetchError } from "./network";
import { HUB_HF_TOKEN_HEADER } from "./hub-token-header";

export type HubResource = "models" | "datasets";

export const PROXY_PREFIX = "/api/hub/discovery/";
export const DEFAULT_HUB_ENDPOINT = "https://huggingface.co";

// Only transport failures justify a retry. A real HTTP status proves the direct
// request reached the Hub, so re-sending it would just double the traffic;
// those never reach here, since fetchWithTimeout resolves with the Response.
const FALLBACK_KINDS = new Set(["network-opaque", "csp-blocked", "timeout"]);

type FetchLike = (
  input: Parameters<typeof fetch>[0],
  init: Parameters<typeof fetch>[1],
) => Promise<Response>;

export interface HubTransportDeps {
  /** Direct cross-origin request to the Hub. */
  direct?: FetchLike;
  /** Same-origin request to the Studio backend (attaches the Studio session). */
  backend?: FetchLike;
  /** True when the configured Hub endpoint is a mirror the browser can't reach. */
  proxyFirst?: () => boolean;
}

// Lazy: @/features/auth/api reads import.meta.env, so a static import would
// make this module unimportable outside vite.
async function defaultBackend(
  input: Parameters<typeof fetch>[0],
  init: Parameters<typeof fetch>[1],
): Promise<Response> {
  const { authFetch } = await import("@/features/auth/api");
  return authFetch(input as RequestInfo, init);
}

function defaultProxyFirst(): boolean {
  return false;
}

function urlOf(input: Parameters<typeof fetch>[0]): string {
  return typeof input === "string"
    ? input
    : input instanceof URL
      ? input.toString()
      : input.url;
}

export function isProxyUrl(raw: string): boolean {
  return raw.startsWith(PROXY_PREFIX);
}

/**
 * Retarget a Hub request at the Studio backend. The SDK puts the HF token in
 * Authorization, but on our API that slot is the Studio session, so the token
 * moves to the internal header and authFetch supplies the bearer.
 */
export function toProxyRequest(
  raw: string,
  resource: HubResource,
  init: Parameters<typeof fetch>[1],
): { url: string; init: Parameters<typeof fetch>[1] } {
  const queryAt = raw.indexOf("?");
  const search = isProxyUrl(raw)
    ? queryAt === -1
      ? ""
      : raw.slice(queryAt)
    : new URL(raw, DEFAULT_HUB_ENDPOINT).search;

  const headers = new Headers(init?.headers);
  const hfAuth = headers.get("Authorization");
  headers.delete("Authorization");
  const bearer = hfAuth?.replace(/^Bearer\s+/i, "").trim();
  if (bearer) {
    headers.set(HUB_HF_TOKEN_HEADER, bearer);
  }
  headers.set("Accept", "application/json");

  return {
    url: `${PROXY_PREFIX}${resource}${search}`,
    init: { ...init, headers },
  };
}

/**
 * A fetch for the Hub SDK: direct first, falling back to the same-origin backend
 * when the browser cannot make the request but the server can.
 *
 * Affinity is per-instance, so a fallen-back iterator keeps its later pages on
 * the proxy instead of re-failing the direct route every page.
 */
export function createHubTransport(
  resource: HubResource,
  deps: HubTransportDeps = {},
): typeof fetch {
  const direct =
    deps.direct ?? ((input, init) => fetchWithTimeout(input, init));
  const backend = deps.backend ?? defaultBackend;
  const proxyFirst = deps.proxyFirst ?? defaultProxyFirst;
  let useProxy = false;

  return async (input, init) => {
    const raw = urlOf(input);

    // The backend hands back a relative next-page link, so a proxy URL here
    // means the SDK is already paginating through the fallback.
    if (isProxyUrl(raw)) {
      const req = toProxyRequest(raw, resource, init);
      return backend(req.url, req.init);
    }

    if (useProxy || proxyFirst()) {
      const req = toProxyRequest(raw, resource, init);
      return backend(req.url, req.init);
    }

    try {
      return await direct(input, init);
    } catch (error) {
      if (!isHubFetchError(error) || !FALLBACK_KINDS.has(error.failure.kind)) {
        throw error;
      }
      // Aborts and HTTP statuses are excluded above: the browser genuinely
      // could not make the request, but the server may still manage it.
      useProxy = true;
      const req = toProxyRequest(raw, resource, init);
      return backend(req.url, req.init);
    }
  };
}
