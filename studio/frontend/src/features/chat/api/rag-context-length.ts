// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// `loadedContextLength` and not `loadedCustomContextLength`: the latter is the n_ctx
// the load was invoked with, which llama-server reduces for a memory fit or a
// --parallel slot split. `_reconcile_effective_ctx_with_server` republishes the window
// it actually serves as this one, so budgeting a document against the pin can overflow
// a load requested at 8K that serves 4K.
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
