// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Centralised HuggingFace endpoint for the frontend.
 *
 * Both getters read live from the Zustand platform store, which is populated
 * from `/api/health` at app startup. Reading on every call keeps the values
 * consistent with the store (including late updates) and avoids a separate
 * cache-staleness failure mode. `getState()` is a cheap property access.
 *
 * The defaults here mirror the backend defaults and are used before
 * `/api/health` returns (or if the fetch fails). Hub mirror and datasets-server
 * URLs are tracked independently — a mirrored `HF_ENDPOINT` does not imply a
 * mirrored datasets-server.
 */

import { usePlatformStore } from "@/config/env";

const DEFAULT_HF_ENDPOINT = "https://huggingface.co";
const DEFAULT_DATASETS_SERVER = "https://datasets-server.huggingface.co";

/** Return the HuggingFace endpoint configured via the backend `HF_ENDPOINT` env var. */
export function getHfEndpoint(): string {
  return usePlatformStore.getState().hfEndpoint || DEFAULT_HF_ENDPOINT;
}

/** Return the HuggingFace datasets-server base URL (from `HF_DATASETS_SERVER`). */
export function getHfDatasetsServerBase(): string {
  return usePlatformStore.getState().hfDatasetsServer || DEFAULT_DATASETS_SERVER;
}
