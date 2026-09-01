// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { splitModelOverrideKey } from "@/features/model-picker/model-config/model-identity";

export type ForgetModelOverrideDeps = {
  removeRemote: (modelId: string, ggufVariant: string | null) => Promise<void>;
  removeLocal: (modelId: string, ggufVariant: string | null) => void;
  reload: () => Promise<void>;
  onError: (message: string) => void;
};

export const FORGET_MODEL_OVERRIDE_FAILED = "Failed to forget these settings";

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
  deps.removeLocal(modelId, ggufVariant);
  await deps.reload();
}
