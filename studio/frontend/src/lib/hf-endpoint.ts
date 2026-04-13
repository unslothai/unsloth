// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Centralised HuggingFace endpoint for the frontend.
 *
 * The endpoint is sourced from the backend ``HF_ENDPOINT`` environment variable,
 * fetched at app startup via ``/api/health``.  The values are cached into
 * module-level variables by :func:`hydrateHfEndpoints` (called once during
 * app bootstrap) so that all subsequent reads are a cheap property access
 * without touching the Zustand store on every call.
 */

import { usePlatformStore } from "@/config/env";

const DEFAULT_HF_ENDPOINT = "https://huggingface.co";
const DEFAULT_DATASETS_SERVER = "https://datasets-server.huggingface.co";

// Cached after the first successful /api/health fetch.
let _cachedEndpoint = DEFAULT_HF_ENDPOINT;
let _cachedDatasetsServer = DEFAULT_DATASETS_SERVER;

/**
 * Cache the HF endpoint values from the Zustand store.
 *
 * Called once by the platform bootstrap after ``/api/health`` returns.
 * Subsequent calls to :func:`getHfEndpoint` and :func:`getHfDatasetsServerBase`
 * read the cached values directly.
 */
export function hydrateHfEndpoints(): void {
  const s = usePlatformStore.getState();
  _cachedEndpoint = s.hfEndpoint || DEFAULT_HF_ENDPOINT;
  _cachedDatasetsServer = s.hfDatasetsServer || DEFAULT_DATASETS_SERVER;
}

/** Return the HuggingFace endpoint configured via the backend ``HF_ENDPOINT`` env var. */
export function getHfEndpoint(): string {
  return _cachedEndpoint;
}

/**
 * Return the HuggingFace datasets-server base URL.
 *
 * Uses the dedicated ``hf_datasets_server`` value from the backend (sourced
 * from ``HF_DATASETS_SERVER`` env var).  This keeps datasets-server independent
 * from the Hub mirror — a mirror may only proxy Hub endpoints, not the
 * datasets-server API.
 */
export function getHfDatasetsServerBase(): string {
  return _cachedDatasetsServer;
}
