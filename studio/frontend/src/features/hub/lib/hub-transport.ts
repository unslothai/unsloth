// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  fetchWithTimeout,
  type HubFailure,
  type HubService,
  isHubFetchError,
  markRemoteNetworkOffline,
  setDirectHubBlocked,
  setHubProxyServing,
} from "./network";
import {
  HUB_HF_TOKEN_HEADER,
  HUB_PROXY_MARKER_HEADER,
} from "./hub-token-header";

export type HubResource = "models" | "datasets";

export const PROXY_PREFIX = "/api/hub/discovery/";
const INFO_PREFIX = "/api/hub/discovery-info/";
export const DEFAULT_HUB_ENDPOINT = "https://huggingface.co";

// Only transport failures justify a retry: a real HTTP status proves the request
// reached the Hub (and resolves as a Response anyway). browser-offline counts
// because navigator.onLine is advisory and reads false on WSL2 and some
// Tauri/WebKitGTK webviews (see network.ts); it must not veto the fallback.
const FALLBACK_KINDS = new Set([
  "network-opaque",
  "csp-blocked",
  "timeout",
  "browser-offline",
]);

// The service tag is decided here, not by the caller: only this module knows
// which URLs are the discovery listing and which are a repo-path lookup, and a
// lookup tagged "discovery" would retire the catalog's diagnosis on success.
type FetchLike = (
  input: Parameters<typeof fetch>[0],
  init: Parameters<typeof fetch>[1],
  service?: HubService,
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

function wasAborted(
  init: Parameters<typeof fetch>[1],
  error: unknown,
): boolean {
  if (init?.signal?.aborted) return true;
  return error instanceof DOMException && error.name === "AbortError";
}

function urlOf(input: Parameters<typeof fetch>[0]): string {
  return typeof input === "string"
    ? input
    : input instanceof URL
      ? input.toString()
      : input.url;
}

/**
 * Where to file a diagnosis. The saved failure names the Hub; `raw` may be the
 * backend's own next-page link, whose origin is Studio's, and keying a Hub
 * failure there leaves the Hub looking available with no cause on screen.
 */
function failureOrigin(failure: HubFailure, raw: string): string {
  return failure.origin ?? hubUrlOf(raw)?.origin ?? DEFAULT_HUB_ENDPOINT;
}

/**
 * A stand-in cause for the proxy-first (mirror) path, which never attempts the
 * direct route and so has no saved failure to carry. Without one, nothing is
 * ever recorded on a mirror: no backoff, no cause on screen, and the phase stays
 * "available" over a dead feed. `origin` is left null so the panel says
 * "Hugging Face" rather than naming the operator's internal mirror.
 */
/**
 * Whether the status the browser received is the Hub's own answer. The backend
 * remaps an upstream 401/403 to 424, so a 424 is the Hub and a 401/403 is
 * Studio's session; it collapses an upstream 5xx to 502, and any other 5xx is
 * Studio itself or a proxy in front of it. Only the rest came from the Hub.
 */
function isHubAnswer(status?: number): boolean {
  if (status === undefined) return false;
  if (status === 424) return true;
  return status < 500 && status !== 401 && status !== 403;
}

function proxyOnlyFailure(status?: number): HubFailure {
  if (status === 424) {
    return {
      kind: "http",
      message: "Hugging Face rejected the access token.",
      origin: null,
      retryable: false,
    };
  }
  if (status === 401 || status === 403) {
    return {
      kind: "http",
      message: "This Studio session is no longer signed in.",
      origin: null,
      retryable: false,
    };
  }
  if (!isHubAnswer(status)) {
    return {
      kind: "network-opaque",
      message: "The server could not reach Hugging Face.",
      origin: null,
      retryable: true,
    };
  }
  return {
    kind: "http",
    message:
      status === 429
        ? "Hugging Face is rate limiting this deployment. Try again shortly."
        : `Hugging Face returned ${status} to the server.`,
    origin: null,
    status,
    retryable: true,
  };
}

/** A status the Hub itself answered this browser with, so it is the diagnosis. */
function directHttpFailure(status: number, origin: string): HubFailure {
  return {
    kind: "http",
    message:
      status === 429
        ? "Hugging Face is rate limiting this browser. Try again shortly."
        : `Hugging Face returned ${status}.`,
    origin,
    status,
    retryable: status !== 401 && status !== 403,
  };
}

function hubUrlOf(raw: string): URL | null {
  try {
    return new URL(raw, DEFAULT_HUB_ENDPOINT);
  } catch {
    return null;
  }
}

/**
 * Match on the path: the backend's next-page link is absolute because
 * @huggingface/hub's parseLinkHeader only matches an http(s) target.
 */
export function isProxyUrl(raw: string): boolean {
  return hubUrlOf(raw)?.pathname.startsWith(PROXY_PREFIX) ?? false;
}

/**
 * The proxy serves listings and carries no path of its own, so a path-bearing
 * request must never be retargeted at it: the SDK would parse a listing array
 * as one repo and cache a model with no id.
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
 * when the browser cannot make the request but the server can. Affinity is
 * per-instance, so a fallen-back iterator keeps later pages on the proxy.
 */
export function createHubTransport(
  resource: HubResource,
  deps: HubTransportDeps = {},
): typeof fetch {
  const direct =
    deps.direct ??
    ((input, init, service) =>
      fetchWithTimeout(input, init, undefined, {
        recordFailure: false,
        service,
      }));
  const backend = deps.backend ?? defaultBackend;
  const proxyFirst = deps.proxyFirst ?? defaultProxyFirst;
  let useProxy = false;
  // Set when the backend turns out to predate the proxy routes.
  let proxyUnavailable = false;
  // Held for the transport's proxy lifetime: later pages have no failure of
  // their own, and the direct one was deliberately never recorded.
  let savedDirectFailure: HubFailure | undefined;

  // A page the backend served proves the Hub's content is reachable even though
  // this browser could not fetch it. Nothing else clears the recorded direct
  // failure, so without this the catalog keeps calling the Hub unavailable while
  // fallback results are on screen, and refuses to paginate.
  const viaBackend = async (
    raw: string,
    init: Parameters<typeof fetch>[1],
    directFailure?: HubFailure,
  ): Promise<Response> => {
    const req = toProxyRequest(raw, resource, init);
    let response: Response;
    try {
      response = await backend(req.url, req.init);
    } catch (error) {
      // A rejection leaves the flag stale, which would keep reporting the Hub
      // available off a proxy that is gone. Once affinity is on this is the
      // only path a later page takes, so the diagnosis is recorded here too.
      setHubProxyServing(false);
      if (!wasAborted(init, error)) {
        const failure = directFailure ?? proxyOnlyFailure();
        const origin = directFailure
          ? failureOrigin(directFailure, raw)
          : DEFAULT_HUB_ENDPOINT;
        markRemoteNetworkOffline(origin, undefined, failure, "discovery");
      }
      throw error;
    }
    // Only that the backend can serve us. The origin's own state is left alone:
    // marking it here would re-render consumers mid-flight, and the flag already
    // suppresses the direct clients (see isDirectHubOffline).
    setHubProxyServing(response.ok);
    if (!response.ok) {
      if (
        response.status === 404 &&
        response.headers.get(HUB_PROXY_MARKER_HEADER) === null
      ) {
        // The SPA catch-all answered: this backend has no proxy route. That is
        // a deployment fact, not a Hub outage, so stop offering the fallback
        // for this transport rather than blaming Hugging Face for it.
        proxyUnavailable = true;
        useProxy = false;
        setHubProxyServing(false);
        return response;
      }
      // authFetch resolves on a non-2xx, so without this the cause is dropped
      // and the panel falls back to its generic wording. The Hub's own answer
      // outranks the saved direct failure: it names something the browser could
      // not see. Anything else is a fact about our own stack, so there the
      // direct cause stays as the more specific of the two.
      const failure =
        isHubAnswer(response.status) || !directFailure
          ? proxyOnlyFailure(response.status)
          : directFailure;
      const origin = directFailure
        ? failureOrigin(directFailure, raw)
        : DEFAULT_HUB_ENDPOINT;
      markRemoteNetworkOffline(origin, undefined, failure, "discovery");
    }
    return response;
  };

  // A repo lookup the backend served proves only that this browser cannot reach
  // the origin, which is what the direct clients need to know. It says nothing
  // about the catalog feed, so it must not touch the feed's flag: that one
  // forces the phase to "available" and blanks the panel, and a detail-pane
  // success would then erase why the catalog failed. Promote only, since a
  // missing repo answers 404 and that is not the browser recovering.
  const viaInfoBackend = async (
    info: { repo: string; revision: string },
    raw: string,
    init: Parameters<typeof fetch>[1],
  ): Promise<Response> => {
    const req = toInfoRequest(info, raw, resource, init);
    const response = await backend(req.url, req.init);
    if (
      response.status === 404 &&
      response.headers.get(HUB_PROXY_MARKER_HEADER) === null
    ) {
      // Same version skew the listing handles: an unmarked 404 is the SPA
      // catch-all, not a missing repo, so stop offering the route rather than
      // reporting the repo gone and retrying a proxy that is not there.
      proxyUnavailable = true;
      useProxy = false;
      return response;
    }
    if (response.ok) {
      setDirectHubBlocked(true);
    }
    return response;
  };

  return async (input, init) => {
    const raw = urlOf(input);

    // The backend hands back a next-page link pointing at itself, so a proxy
    // URL here means the SDK is already paginating through the fallback.
    if (isProxyUrl(raw)) {
      return viaBackend(raw, init, savedDirectFailure);
    }

    if (!isListingUrl(raw, resource)) {
      const info = infoTargetOf(raw, resource);
      if (info && ((useProxy && !proxyUnavailable) || proxyFirst())) {
        return viaInfoBackend(info, raw, init);
      }
      if (!info) {
        // Not a route the backend can serve, so direct is the only option.
        return direct(input, init, "other");
      }
      try {
        const response = await direct(input, init, "other");
        // The browser reached the Hub, so whatever blocked it has lifted. This
        // is the only clear on the info path: nothing else demotes it, and the
        // flag has no expiry, so one transient timeout would otherwise keep the
        // README, avatar and quant clients on cached data for the session.
        setDirectHubBlocked(false);
        return response;
      } catch (error) {
        // The same fallback the listing gets. Without it a blocked browser shows
        // a working feed with no metadata on any deep link, pinned publisher or
        // priority card, because these transports never run a listing and so
        // never pick up the affinity that would have routed them.
        if (
          proxyUnavailable ||
          !isHubFetchError(error) ||
          !FALLBACK_KINDS.has(error.failure.kind)
        ) {
          throw error;
        }
        useProxy = true;
        return viaInfoBackend(info, raw, init);
      }
    }

    // proxyFirst is not gated on proxyUnavailable: on a mirror the direct route
    // would put a private repo id and token on the public Hub, so a backend
    // without the route means the feed is unavailable, not that we go direct.
    if ((useProxy && !proxyUnavailable) || proxyFirst()) {
      return viaBackend(raw, init, savedDirectFailure);
    }

    try {
      const response = await direct(input, init, "discovery");
      // The browser fetched the listing itself, so the backend is not serving
      // the feed; a stale flag would force "available" and hide the next failure.
      setHubProxyServing(false);
      setDirectHubBlocked(false);
      if (!response.ok) {
        // The SDK turns this into an ApiError, and fetchWithTimeout has already
        // cleared the origin on the resolved response, so nothing else records
        // it and the status never reaches the panel. The ttl is 0 on purpose: a
        // status proves the origin IS reachable, so backing off would put the
        // README, avatar and quant clients into offline mode and stall Load
        // more over a bad query or a stale token. The cause is recorded, the
        // phase stays "probing", and this is not a fallback trigger.
        const origin = hubUrlOf(raw)?.origin ?? DEFAULT_HUB_ENDPOINT;
        // No backoff is opened or retired here. fetchWithTimeout already
        // cleared this feed's window on the response, and one opened after it by
        // another request is newer evidence than ours: clearing that would
        // defeat the backoff from the auto path and resume direct README and
        // avatar fetches against an origin that just failed for them.
        markRemoteNetworkOffline(
          origin,
          0,
          directHttpFailure(response.status, origin),
          "discovery",
        );
      }
      return response;
    } catch (error) {
      if (
        proxyUnavailable ||
        !isHubFetchError(error) ||
        !FALLBACK_KINDS.has(error.failure.kind)
      ) {
        throw error;
      }
      // Aborts and HTTP statuses are excluded above: the browser genuinely
      // could not make the request, but the server may still manage it.
      useProxy = true;
      savedDirectFailure = error.failure;
      try {
        return await viaBackend(raw, init, error.failure);
      } catch (proxyError) {
        // Both routes are gone, so now it is true; recording it earlier would
        // have aborted the attempt above. An abort is not a proxy failure, and
        // a superseded query must not back off its own replacement.
        const origin = hubUrlOf(raw)?.origin;
        if (origin && !wasAborted(init, proxyError)) {
          markRemoteNetworkOffline(origin, undefined, error.failure, "discovery");
        }
        throw proxyError;
      }
    }
  };
}
