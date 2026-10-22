// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { getCatalog } from "../api";
import type { NodeCatalog } from "../types";
import { useCatalogStore } from "./catalog-store";

let loading: Promise<NodeCatalog> | null = null;

/** The backend's catalog, once per session. */
export function ensureCatalog(): Promise<NodeCatalog> {
  const existing = useCatalogStore.getState().catalog;
  if (existing) return Promise.resolve(existing);
  loading ??= getCatalog()
    .then((catalog) => {
      useCatalogStore.getState().setCatalog(catalog);
      return catalog;
    })
    .catch((error: unknown) => {
      loading = null;
      throw error;
    });
  return loading;
}
