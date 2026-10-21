// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { effectiveMinPMode } from "./min-p-policy";
import type { MinPMode } from "../types/runtime";

const rejection =
  "The min_p and logit_bias sampling parameters are not yet supported with speculative decoding";

export function shouldOfferMinPRecovery(
  message: string,
  providerType: string | undefined,
  params: { minP: number; minPMode?: MinPMode },
): boolean {
  return (
    providerType === "vllm" &&
    message.includes(rejection) &&
    !(effectiveMinPMode(params) === "custom" && params.minP === 0)
  );
}

const pendingGuards = new Set<() => void>();

/** A newer send supersedes an old error action, even if its settings are identical. */
export function invalidateMinPRecoveries(): void {
  for (const invalidate of pendingGuards) invalidate();
}

export function createMinPRecoveryGuard(
  readContext: () => readonly unknown[],
  subscribe: (listener: () => void) => () => void,
) {
  const original = readContext();
  let valid = true;
  const invalidate = () => {
    valid = false;
  };
  const check = () => {
    const current = readContext();
    if (
      current.length !== original.length ||
      current.some((value, index) => !Object.is(value, original[index]))
    )
      invalidate();
    return valid;
  };
  pendingGuards.add(invalidate);
  const unsubscribe = subscribe(check);
  let disposed = false;
  return {
    isValid: check,
    dispose() {
      if (disposed) return;
      disposed = true;
      invalidate();
      pendingGuards.delete(invalidate);
      unsubscribe();
    },
  };
}
