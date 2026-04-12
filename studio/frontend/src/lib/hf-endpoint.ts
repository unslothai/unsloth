// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Centralised HuggingFace endpoint for the frontend.
 *
 * The endpoint is sourced from the backend ``HF_ENDPOINT`` environment variable,
 * fetched at app startup via ``/api/health`` and stored in the platform Zustand
 * store.  All frontend code that needs the HF hub URL should call
 * :func:`getHfEndpoint`.
 */

import { usePlatformStore } from "@/config/env";

const DEFAULT_HF_ENDPOINT = "https://huggingface.co";

/** Return the HuggingFace endpoint configured via the backend ``HF_ENDPOINT`` env var. */
export function getHfEndpoint(): string {
  return usePlatformStore.getState().hfEndpoint || DEFAULT_HF_ENDPOINT;
}

/**
 * Return the HuggingFace datasets-server base URL.
 * Falls back to the official endpoint if no mirror is configured.
 */
export function getHfDatasetsServerBase(): string {
  const mirror = getHfEndpoint();
  if (mirror !== DEFAULT_HF_ENDPOINT) {
    // Common mirrors serve datasets-server at the same host
    return mirror;
  }
  return "https://datasets-server.huggingface.co";
}
