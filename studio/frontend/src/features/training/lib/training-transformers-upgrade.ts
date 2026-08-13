// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  type TransformersUpgradeCheck,
  checkTransformersUpgrade,
  confirmTransformersUpgradeIfNeeded,
  useTransformersUpgradeDialogStore,
} from "@/features/transformers-upgrade";

/** What the caller must do next once the upgrade gate has run. */
export interface TrainingTransformersUpgradeOutcome {
  /** False when the install was declined or none exists: do not start the run. */
  proceed: boolean;
  /** Why the start was abandoned; set only when proceed is false. */
  error: string | null;
  /** The run will load 16-bit, not bnb 4-bit, because it routes to the latest sidecar. */
  forces16Bit: boolean;
}

/** Names the worker's own failure ("... is not supported yet in transformers==x.y.z")
 *  rather than a generic start error, and says what to do about it. */
export function getTrainingTransformersUpgradeRequiredMessage(
  modelName: string,
): string {
  return `${modelName} is not supported yet by the installed transformers, and the newer release it needs was not installed. Start the run again to install it.`;
}

/** Gate a training start on the transformers release the model needs.
 *
 * Chat pauses a load on this dialog from `/validate`; training never asked, so a model
 * whose architecture no installed transformers ships used to be accepted, spawned, and
 * killed minutes later at model load with an error the user could not act on. Same
 * dialog, same install, raised before the run starts.
 *
 * Additive: a backend that does not serve the check, or one that fails it, leaves the
 * start exactly as it was. */
export async function confirmTrainingTransformersUpgrade({
  modelName,
  hfToken,
}: {
  modelName: string;
  hfToken?: string | null;
}): Promise<TrainingTransformersUpgradeOutcome> {
  let check: TransformersUpgradeCheck;
  try {
    check = await checkTransformersUpgrade(modelName, hfToken);
  } catch {
    return { proceed: true, error: null, forces16Bit: false };
  }
  if (!check.upgrade) {
    return { proceed: true, error: null, forces16Bit: check.forces16Bit };
  }

  const upgraded = await confirmTransformersUpgradeIfNeeded({
    modelName,
    upgrade: check.upgrade,
    // No installable release: a model shipping its own modeling code can still go
    // through the trust_remote_code gate the caller runs next, exactly as chat does.
    trustRemoteCodeFallback: check.requiresTrustRemoteCode,
    // No forceCancelActive: training raises no "stop N chats" prompt of its own, so it
    // has no such answer to carry. A chat mid-generation makes the install refuse and
    // the dialog says so, rather than this tab killing someone else's stream unasked.
  });
  // Read before the resolve-time state is reused by any later consent.
  const installRan = useTransformersUpgradeDialogStore.getState().installRan;
  if (
    useTransformersUpgradeDialogStore.getState().consumeServerUnloadedChat()
  ) {
    // The install unloads the active chat model before the swap. Nothing on this tab
    // owns that selection, so resync it or chat keeps pointing at a model that is gone.
    void import("@/features/chat")
      .then((chat) => chat.resyncInferenceStatusAfterServerModelChange())
      .catch(() => undefined);
  }
  if (!upgraded) {
    return {
      proceed: false,
      error: getTrainingTransformersUpgradeRequiredMessage(modelName),
      forces16Bit: false,
    };
  }
  // Installed: the model now routes to the latest sidecar, which trains 16-bit. The
  // custom-code fallback resolves true WITHOUT installing and still loads 4-bit.
  return { proceed: true, error: null, forces16Bit: installRan };
}
