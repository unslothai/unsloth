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

/**
 * Accept a value from `/api/health` only if it parses as an http(s) URL.
 *
 * Deliberately NOT a second copy of the backend's policy. `utils/hf_endpoint.py`
 * already sanitises, folds and canonicalises this value, and it is the only
 * producer; re-deciding the rules here in another language is what made the two
 * disagree about a path ending in a colon, an IDN host and an uncompressed IPv6
 * literal. This is a parse check, so a junk value cannot crash a caller building
 * `new URL(getHfEndpoint())`.
 */
function usableEndpoint(raw: string | null | undefined): string | null {
  if (typeof raw !== "string") return null;
  const trimmed = raw.trim();
  if (!trimmed) return null;
  try {
    const { protocol } = new URL(trimmed);
    if (protocol !== "https:" && protocol !== "http:") return null;
  } catch {
    return null;
  }
  return trimmed.replace(/\/+$/, "");
}

/**
 * Apply the endpoints reported by `/api/health`. A blank, absent or unparseable
 * value leaves the current one alone: older backends report neither field, and
 * resetting would strand a mirror-only deployment on huggingface.co.
 */
export function setHfEndpoints(
  endpoint?: string | null,
  datasetsServer?: string | null,
): void {
  const nextEndpoint = usableEndpoint(endpoint);
  if (nextEndpoint) _endpoint = nextEndpoint;
  const nextDatasetsServer = usableEndpoint(datasetsServer);
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
