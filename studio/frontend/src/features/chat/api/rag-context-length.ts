// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export function ragScopeContextLength(input: {
  isExternalRequest: boolean;
  loadedCustomContextLength?: number | null;
  loadedContextLength?: number | null;
  maxSeqLength?: number | null;
}): number | undefined {
  if (input.isExternalRequest) {
    return undefined;
  }
  return (
    input.loadedCustomContextLength ??
    input.loadedContextLength ??
    input.maxSeqLength ??
    undefined
  );
}
