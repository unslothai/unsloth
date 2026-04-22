// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Centralised HuggingFace endpoint for the frontend.
 *
 * The endpoint values are cached in module-level variables so that every
 * `getHfEndpoint()` / `getHfDatasetsServerBase()` call is a single property
 * read with no store lookup. A Zustand `subscribe` keeps the cache in sync
 * automatically — whenever `/api/health` populates the platform store, the
 * subscription fires and the cached values update, so there is no explicit
 * hydrate step and no pre-hydration staleness window.
 *
 * Storage semantics: everything here is in-memory. `usePlatformStore` is a
 * plain (non-`persist`) Zustand store and the module variables live in
 * module scope, so on every page load the frontend re-fetches `/api/health`
 * and the backend's live `HF_ENDPOINT` env var wins.
 */

import { usePlatformStore } from "@/config/env";

const DEFAULT_HF_ENDPOINT = "https://huggingface.co";
const DEFAULT_DATASETS_SERVER = "https://datasets-server.huggingface.co";

// Seeded from the store's current state (defaults until /api/health resolves).
let _endpoint = usePlatformStore.getState().hfEndpoint || DEFAULT_HF_ENDPOINT;
let _datasetsServer =
  usePlatformStore.getState().hfDatasetsServer || DEFAULT_DATASETS_SERVER;

usePlatformStore.subscribe((state) => {
  _endpoint = state.hfEndpoint || DEFAULT_HF_ENDPOINT;
  _datasetsServer = state.hfDatasetsServer || DEFAULT_DATASETS_SERVER;
});

/** Return the HuggingFace endpoint configured via the backend `HF_ENDPOINT` env var. */
export function getHfEndpoint(): string {
  return _endpoint;
}

/** Return the HuggingFace datasets-server base URL (from `HF_DATASETS_SERVER`). */
export function getHfDatasetsServerBase(): string {
  return _datasetsServer;
}
