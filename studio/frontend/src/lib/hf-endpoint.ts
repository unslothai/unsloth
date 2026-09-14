// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Centralised HuggingFace endpoint for the frontend. `config/env.ts` pushes the
 * values in when `/api/health` answers, so the backend's live env wins on every
 * page load. This module imports NOTHING on purpose: `network.ts` imports it and
 * the unit tests import that under bare node, which cannot evaluate the
 * `import.meta.env` and Zustand store `config/env.ts` would bring.
 */

export const DEFAULT_HF_ENDPOINT = "https://huggingface.co";
export const DEFAULT_DATASETS_SERVER = "https://datasets-server.huggingface.co";

let _endpoint = DEFAULT_HF_ENDPOINT;
let _datasetsServer = DEFAULT_DATASETS_SERVER;

/** Mirror the backend's `hf_endpoint_url()`: add a missing scheme, drop trailing
 * slashes, reject anything that is not a plain http(s) origin. */
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
  // Printable ASCII only: whitespace and separators would add sources to the CSP
  // built from this same value, and a Unicode host 500s every response there
  // (latin-1 headers). All three sides take the punycode form.
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
  // Hub calls carry the user's token, so http off-box puts it on the wire.
  if (parsed.protocol === "http:" && !isLoopbackHost(parsed.hostname)) return null;
  if (parsed.hostname.includes("*")) return null;
  // Authority, not the whole URL: a path may legally end in a colon.
  const authority = withScheme.slice(withScheme.indexOf("://") + 3).split(/[/?#]/)[0];
  if (authority.startsWith(":") || authority.endsWith(":")) return null;
  if (parsed.username || parsed.password || parsed.search || parsed.hash) {
    return null;
  }
  const path = parsed.pathname.replace(/\/+$/, "");
  return `${parsed.origin}${path}`;
}

/**
 * Apply the endpoints reported by `/api/health`. A blank, absent or malformed
 * value leaves the current one alone: older backends report neither field, and
 * resetting would strand a mirror-only deployment on huggingface.co.
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

export function resetHfEndpoints(): void {
  _endpoint = DEFAULT_HF_ENDPOINT;
  _datasetsServer = DEFAULT_DATASETS_SERVER;
}

export function getHfEndpoint(): string {
  return _endpoint;
}

export function getHfDatasetsServerBase(): string {
  return _datasetsServer;
}
