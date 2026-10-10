// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { foldOverrideKey } from "@/features/model-picker/api/model-overrides";
import { splitModelOverrideKey } from "@/features/model-picker/model-config/model-identity";

export type ForgetModelOverrideDeps = {
  listedKeys: readonly string[];
  removeRemote: (
    modelId: string,
    ggufVariant: string | null,
  ) => Promise<{
    overrides: Readonly<Record<string, unknown>>;
    removedKeys: readonly string[];
  }>;
  removeLocal: (overrideKeys: readonly string[]) => boolean;
  reload: () => Promise<void>;
  onError: (message: string) => void;
};

export const FORGET_MODEL_OVERRIDE_FAILED = "Failed to forget these settings";
export const FORGET_MODEL_OVERRIDE_LOCAL_FAILED =
  "Forgot these settings for the API, but this browser kept its own copy";

function keysNoLongerHeld(
  listedKeys: readonly string[],
  overrides: Readonly<Record<string, unknown>>,
): string[] {
  const held = new Set(Object.keys(overrides).map(foldOverrideKey));
  return listedKeys.filter((key) => !held.has(foldOverrideKey(key)));
}

export async function forgetModelOverride(
  overrideKey: string,
  deps: ForgetModelOverrideDeps,
): Promise<void> {
  const [modelId, ggufVariant] = splitModelOverrideKey(overrideKey);
  let result: Awaited<ReturnType<ForgetModelOverrideDeps["removeRemote"]>>;
  try {
    result = await deps.removeRemote(modelId, ggufVariant);
  } catch (err: unknown) {
    deps.onError(
      err instanceof Error ? err.message : FORGET_MODEL_OVERRIDE_FAILED,
    );
    return;
  }
  // Older backends lack removed_keys; a leftover copy is reported since the next save mirrors it.
  const cleared =
    result.removedKeys.length > 0
      ? result.removedKeys
      : keysNoLongerHeld(deps.listedKeys, result.overrides);
  if (!deps.removeLocal([...new Set([overrideKey, ...cleared])])) {
    deps.onError(FORGET_MODEL_OVERRIDE_LOCAL_FAILED);
  }
  await deps.reload();
}
