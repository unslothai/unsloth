// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  fetchWithTimeout,
  type HubFailure,
  isHubFetchError,
  markRemoteNetworkOffline,
  setHubProxyServing,
} from "./network";
import { HUB_HF_TOKEN_HEADER } from "./hub-token-header";

export type HubResource = "models" | "datasets";

export const PROXY_PREFIX = "/api/hub/discovery/";
const INFO_PREFIX = "/api/hub/discovery-info/";
export const DEFAULT_HUB_ENDPOINT = "https://huggingface.co";

// Only transport failures justify a retry. A real HTTP status proves the direct
// request reached the Hub, so re-sending it would just double the traffic;
// those never reach here, since fetchWithTimeout resolves with the Response.
// browser-offline is included because navigator.onLine is advisory and reads
// false on WSL2 and some Tauri/WebKitGTK webviews (see network.ts); letting it
// veto the fallback would make the one untrusted signal authoritative.
const FALLBACK_KINDS = new Set([
  "network-opaque",
  "csp-blocked",
  "timeout",
  "browser-offline",
]);

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

function hubUrlOf(raw: string): URL | null {
  try {
    return new URL(raw, DEFAULT_HUB_ENDPOINT);
  } catch {
    return null;
  }
}

/**
 * The backend's next-page link is absolute, because @huggingface/hub's
 * parseLinkHeader only matches an http(s) target, so match on the path rather
 * than the whole string.
 */
export function isProxyUrl(raw: string): boolean {
  return hubUrlOf(raw)?.pathname.startsWith(PROXY_PREFIX) ?? false;
}

/**
 * The proxy serves the listing endpoints and carries no path of its own, so a
 * request that means something by its path (modelInfo's
 * /api/models/{repo}/revision/HEAD) must never be retargeted at it: the SDK
 * would parse a listing array as one repo and cache a model with no id.
 */
function isListingUrl(raw: string, resource: HubResource): boolean {
  return hubUrlOf(raw)?.pathname === `/api/${resource}`;
}

/**
 * modelInfo's /api/{resource}/{repo}/revision/{rev}. It needs the path-preserving
 * route: sending it to the listing proxy loses the repo, and leaving it direct
 * would put a mirror user's repo id and token on the public Hub.
 */
function infoTargetOf(
  raw: string,
  resource: HubResource,
): { repo: string; revision: string } | null {
  const path = hubUrlOf(raw)?.pathname;
  const m = path?.match(
    new RegExp(`^/api/${resource}/(.+)/revision/([^/]+)$`),
  );
  if (!m) return null;
  return {
    repo: decodeURIComponent(m[1]),
    revision: decodeURIComponent(m[2]),
  };
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
  const search = hubUrlOf(raw)?.search ?? "";

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

function toInfoRequest(
  target: { repo: string; revision: string },
  raw: string,
  resource: HubResource,
  init: Parameters<typeof fetch>[1],
): { url: string; init: Parameters<typeof fetch>[1] } {
  const req = toProxyRequest(raw, resource, init);
  const params = new URLSearchParams(hubUrlOf(raw)?.search ?? "");
  const out = new URLSearchParams();
  out.set("repo", target.repo);
  out.set("revision", target.revision);
  for (const v of params.getAll("expand")) out.append("expand", v);
  return { url: `${INFO_PREFIX}${resource}?${out}`, init: req.init };
}

/**
 * The fetch every modelInfo caller should use: with a mirror configured it goes
 * through the path-preserving backend route rather than the public Hub.
 */
export function createModelInfoFetch(
  resource: HubResource = "models",
): typeof fetch {
  return createHubTransport(resource, { proxyFirst: hubProxyFirstRef });
}

// Set by the hub feature at import time; kept indirect so this module does not
// pull in @/config/env, which is unimportable outside vite.
let hubProxyFirstRef: () => boolean = defaultProxyFirst;

export function setHubProxyFirst(fn: () => boolean): void {
  hubProxyFirstRef = fn;
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
    deps.direct ??
    ((input, init) =>
      fetchWithTimeout(input, init, undefined, { recordFailure: false }));
  const backend = deps.backend ?? defaultBackend;
  const proxyFirst = deps.proxyFirst ?? defaultProxyFirst;
  let useProxy = false;

  // A page the backend served proves the Hub's content is reachable even though
  // this browser could not fetch it. fetchWithTimeout has already recorded the
  // direct failure, and nothing else clears it, so without this the catalog
  // keeps calling the Hub unavailable while fallback results are on screen: it
  // refuses to paginate and renders an empty result set as a connection error.
  const viaBackend = async (
    raw: string,
    init: Parameters<typeof fetch>[1],
    directFailure?: HubFailure,
  ): Promise<Response> => {
    const req = toProxyRequest(raw, resource, init);
    const response = await backend(req.url, req.init);
    // Only that the backend can serve us. The origin's own state is left alone
    // so the direct clients stay suppressed instead of failing and re-marking.
    setHubProxyServing(response.ok);
    if (!response.ok && directFailure) {
      // authFetch resolves on a non-2xx, so without this the saved direct
      // failure is dropped and the panel loses the classified cause.
      const origin = hubUrlOf(raw)?.origin;
      if (origin) {
        markRemoteNetworkOffline(origin, undefined, directFailure);
      }
    }
    return response;
  };

  return async (input, init) => {
    const raw = urlOf(input);

    // The backend hands back a next-page link pointing at itself, so a proxy
    // URL here means the SDK is already paginating through the fallback.
    if (isProxyUrl(raw)) {
      return viaBackend(raw, init);
    }

    if (!isListingUrl(raw, resource)) {
      const info = infoTargetOf(raw, resource);
      if (info && (useProxy || proxyFirst())) {
        const req = toInfoRequest(info, raw, resource, init);
        return backend(req.url, req.init);
      }
      // Anything else keeps its path and stays on the direct route.
      return direct(input, init);
    }

    if (useProxy || proxyFirst()) {
      return viaBackend(raw, init);
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
      try {
        return await viaBackend(raw, init, error.failure);
      } catch (proxyError) {
        // Both routes are gone, so now it is true. Recording it earlier would
        // have aborted the attempt above.
        const origin = hubUrlOf(raw)?.origin;
        if (origin) {
          markRemoteNetworkOffline(origin, undefined, error.failure);
        }
        throw proxyError;
      }
    }
  };
}
