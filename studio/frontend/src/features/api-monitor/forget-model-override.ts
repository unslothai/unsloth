// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { splitModelOverrideKey } from "@/features/model-picker/model-config/model-identity";

export type ForgetModelOverrideDeps = {
  removeRemote: (modelId: string, ggufVariant: string | null) => Promise<void>;
  /** False when a record was left behind, as deletePerModelConfig reports it. */
  removeLocal: (modelId: string, ggufVariant: string | null) => boolean;
  reload: () => Promise<void>;
  onError: (message: string) => void;
};

export const FORGET_MODEL_OVERRIDE_FAILED = "Failed to forget these settings";
export const FORGET_MODEL_OVERRIDE_LOCAL_FAILED =
  "Forgot these settings for the API, but this browser kept its own copy";

export async function forgetModelOverride(
  overrideKey: string,
  deps: ForgetModelOverrideDeps,
): Promise<void> {
  const [modelId, ggufVariant] = splitModelOverrideKey(overrideKey);
  try {
    await deps.removeRemote(modelId, ggufVariant);
  } catch (err: unknown) {
    deps.onError(
      err instanceof Error ? err.message : FORGET_MODEL_OVERRIDE_FAILED,
    );
    return;
  }
  // Reported rather than swallowed: the picker would still apply that copy, and the
  // model's next save mirrors it back to the server the row was just removed from.
  // The list is refetched either way, because the server entry is gone.
  if (!deps.removeLocal(modelId, ggufVariant)) {
    deps.onError(FORGET_MODEL_OVERRIDE_LOCAL_FAILED);
  }
  await deps.reload();
}
