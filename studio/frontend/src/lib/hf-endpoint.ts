// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Centralised HuggingFace endpoint for the frontend.
 *
 * The endpoint is sourced from the backend ``HF_ENDPOINT`` environment variable,
 * fetched at app startup via ``/api/health`` and stored in the platform Zustand
 * store.  All frontend code that needs the HF hub URL should call
 * :func:`getHfEndpoint`.
 *
 * The datasets-server URL is resolved similarly via ``HF_DATASETS_SERVER``
 * (with automatic fallback to the same host as the mirror, or the official
 * ``datasets-server.huggingface.co``).
 */

import { usePlatformStore } from "@/config/env";

const DEFAULT_HF_ENDPOINT = "https://huggingface.co";
const DEFAULT_DATASETS_SERVER = "https://datasets-server.huggingface.co";

/** Return the HuggingFace endpoint configured via the backend ``HF_ENDPOINT`` env var. */
export function getHfEndpoint(): string {
  return usePlatformStore.getState().hfEndpoint || DEFAULT_HF_ENDPOINT;
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
  return usePlatformStore.getState().hfDatasetsServer || DEFAULT_DATASETS_SERVER;
}
