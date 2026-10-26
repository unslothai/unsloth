// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// `loadedContextLength`, not the requested n_ctx: llama-server may serve less (memory fit or slots).
export function ragScopeContextLength(input: {
  isExternalRequest: boolean;
  loadedContextLength?: number | null;
  maxSeqLength?: number | null;
}): number | undefined {
  if (input.isExternalRequest) {
    return undefined;
  }
  return input.loadedContextLength ?? input.maxSeqLength ?? undefined;
}
