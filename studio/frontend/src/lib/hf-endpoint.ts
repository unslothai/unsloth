// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Centralised HuggingFace endpoint for the frontend.
 *
 * The endpoint values live in module-level variables, so every
 * `getHfEndpoint()` / `getHfDatasetsServerBase()` call is a single property
 * read. `config/env.ts` pushes them in with `setHfEndpoints()` when
 * `/api/health` answers.
 *
 * This module deliberately imports nothing. `features/hub/lib/network.ts` is a
 * leaf utility that the unit tests import directly under bare
 * `node --experimental-strip-types`; reaching back into `config/env.ts` from
 * here would drag `import.meta.env` and a Zustand store into that graph, which
 * bare Node cannot evaluate. The dependency runs env.ts -> here, never back.
 *
 * Storage semantics: everything here is in-memory, so on every page load the
 * frontend re-fetches `/api/health` and the backend's live `HF_ENDPOINT` env
 * var wins.
 */

export const DEFAULT_HF_ENDPOINT = "https://huggingface.co";
export const DEFAULT_DATASETS_SERVER = "https://datasets-server.huggingface.co";

let _endpoint = DEFAULT_HF_ENDPOINT;
let _datasetsServer = DEFAULT_DATASETS_SERVER;

/**
 * Mirror the backend's `hf_endpoint_url()` normalisation: add a scheme when one
 * is missing and drop trailing slashes, so consumers that build
 * `` `${getHfEndpoint()}/api/models` `` never emit a doubled slash. Anything
 * that is not a plain http(s) origin is rejected rather than propagated -- the
 * value is pasted into URLs and compared against `new URL(...).origin`, and a
 * junk value there fails far away from its cause.
 */
function isLoopbackHost(hostname: string): boolean {
  const host = hostname.replace(/^\[|\]$/g, "").toLowerCase();
  if (host === "localhost" || host.endsWith(".localhost")) return true;
  if (host === "::1") return true;
  return /^127\.\d{1,3}\.\d{1,3}\.\d{1,3}$/.test(host);
}

function normalizeEndpoint(raw: string | null | undefined): string | null {
  if (typeof raw !== "string") return null;
  const trimmed = raw.trim();
  if (!trimmed) return null;
  // Printable ASCII only. Whitespace and separators would add sources to the CSP
  // connect-src the backend builds from this same value; control characters and
  // non-ASCII ones cannot reach it at all, since Starlette encodes header values
  // as latin-1, so a Unicode host there 500s every response. new URL() would
  // punycode such a host, and accepting it here alone would point this frontend
  // somewhere neither the backend nor the desktop CSP builder allows. The
  // punycode (xn--) form is printable ASCII and is accepted by all three.
  if (/[^\u0020-\u007e]/.test(trimmed) || /[\s;,'"\\]/.test(trimmed)) {
    return null;
  }
  const withScheme = trimmed.includes("://") ? trimmed : `https://${trimmed}`;
  let parsed: URL;
  try {
    parsed = new URL(withScheme);
  } catch {
    return null;
  }
  if (parsed.protocol !== "https:" && parsed.protocol !== "http:") return null;
  if (!parsed.hostname) return null;
  // Hub calls carry the user's token (`credentials: { accessToken }`), so a
  // plain-HTTP mirror off-box would put a bearer token on the wire in cleartext.
  // Loopback is the local-proxy case and never leaves the machine.
  if (parsed.protocol === "http:" && !isLoopbackHost(parsed.hostname)) return null;
  // "*" would reach the backend's CSP connect-src as "https://*", a source that
  // allows every https origin, and is never a host to send a request to.
  if (parsed.hostname.includes("*")) return null;
  // "host:" parses with no port and is not a usable origin, but the check has to
  // look at the authority alone: a path may legally end in a colon, and the
  // backend sanitizer tests parts.netloc, so a whole-string test here would
  // reject a mirror the backend accepts and split the two apart again.
  const authority = withScheme.slice(withScheme.indexOf("://") + 3).split(/[/?#]/)[0];
  if (authority.startsWith(":") || authority.endsWith(":")) return null;
  // Credentials, query and fragment are meaningless on a base URL and would be
  // carried into every request built from it.
  if (parsed.username || parsed.password || parsed.search || parsed.hash) {
    return null;
  }
  const path = parsed.pathname.replace(/\/+$/, "");
  return `${parsed.origin}${path}`;
}

/**
 * Apply the endpoints reported by `/api/health`. A blank, absent or malformed
 * value leaves the current value alone: older backends report neither field,
 * and resetting a configured mirror to the default would silently send a
 * mirror-only deployment back to huggingface.co.
 */
export function setHfEndpoints(
  endpoint?: string | null,
  datasetsServer?: string | null,
): void {
  const nextEndpoint = normalizeEndpoint(endpoint);
  if (nextEndpoint) _endpoint = nextEndpoint;
  const nextDatasetsServer = normalizeEndpoint(datasetsServer);
  if (nextDatasetsServer) _datasetsServer = nextDatasetsServer;
}

/** Reset both endpoints to their defaults. Exported for tests. */
export function resetHfEndpoints(): void {
  _endpoint = DEFAULT_HF_ENDPOINT;
  _datasetsServer = DEFAULT_DATASETS_SERVER;
}

/** Return the HuggingFace endpoint configured via the backend `HF_ENDPOINT` env var. */
export function getHfEndpoint(): string {
  return _endpoint;
}

/** Return the HuggingFace datasets-server base URL (from `HF_DATASETS_SERVER`). */
export function getHfDatasetsServerBase(): string {
  return _datasetsServer;
}
