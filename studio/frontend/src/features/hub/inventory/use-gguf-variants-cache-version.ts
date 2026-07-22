// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useSyncExternalStore } from "react";
import {
  getGgufVariantsCacheVersion,
  subscribeGgufVariantsCache,
} from "./gguf-variants-cache-events";

export function useGgufVariantsCacheVersion(
  repoId?: string | null,
): string {
  return useSyncExternalStore(
    subscribeGgufVariantsCache,
    () => getGgufVariantsCacheVersion(repoId),
    () => getGgufVariantsCacheVersion(repoId),
  );
}
