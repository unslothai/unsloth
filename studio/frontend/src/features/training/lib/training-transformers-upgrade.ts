// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  type ModelCachePin,
  type TransformersUpgradeCheck,
  checkTransformersUpgrade,
  confirmTransformersUpgradeIfNeeded,
  upgradeInstallVersion,
  useTransformersUpgradeDialogStore,
} from "@/features/transformers-upgrade";

export interface TrainingTransformersUpgradeOutcome {
  /** False when declined or none exists: do not start the run. */
  proceed: boolean;
  error: string | null;
  /** Routed to the latest sidecar, which loads 16-bit, not bnb 4-bit. */
  forces16Bit: boolean;
  /** Hand to the custom-code gate next, whose fallback reads a flag that is false for fresh runs. */
  requiresTrustRemoteCode: boolean;
}

export interface TrainingTransformersUpgradeNotice {
  installVersion: string | null;
  fourBitUnavailable: boolean;
  /** 4-bit survives only via the model's own code; Install activates the 16-bit sidecar. */
  installSwitchesTo16Bit: boolean;
}

/** With custom code and an installable release, the fallback keeps 4-bit while Install trains
 * 16-bit. `installable && !forces16Bit` is exactly that case. */
export function trainingTransformersUpgradeNotice(
  check: TransformersUpgradeCheck,
  loadsIn4Bit: boolean,
): TrainingTransformersUpgradeNotice {
  const installVersion = upgradeInstallVersion(check.upgrade);
  const installable = installVersion !== null;
  return {
    installVersion,
    fourBitUnavailable: check.forces16Bit && loadsIn4Bit,
    installSwitchesTo16Bit: installable && !check.forces16Bit && loadsIn4Bit,
  };
}

/** Names the worker's "not supported yet in transformers==x.y.z" failure and the fix. */
export function getTrainingTransformersUpgradeRequiredMessage(
  modelName: string,
): string {
  return `${modelName} is not supported yet by the installed transformers, and the newer release it needs was not installed. Start the run again to install it.`;
}

/** Dev-only: no PyPI release ships it, so there is no Install action to point at. */
export function getTrainingTransformersUpgradeUnavailableMessage(
  modelName: string,
): string {
  return `${modelName} is not supported yet by the installed transformers, and no released transformers version supports it either: the architecture is only on the transformers development branch, whose version could not be checked right now. Try again in a few minutes to install it, or pick a model the installed transformers supports.`;
}

/** The checkpoint needs a 4-bit load the latest sidecar permanently refuses. */
export function getTrainingResumeUpgradeWouldStrandMessage(
  modelName: string,
): string {
  return `${modelName} is not supported yet by the installed transformers, and installing the release it needs would permanently retire the 4-bit model load this checkpoint was attested with, so the resume would still fail. Start a new run on this model instead.`;
}

/** Raise chat's upgrade dialog before a run starts instead of failing at model load. Additive:
 * a backend without the check leaves the start unchanged. */
export async function confirmTrainingTransformersUpgrade({
  modelName,
  hfToken,
  modelCachePin,
  resumeRunId,
}: {
  modelName: string;
  hfToken?: string | null;
  /** Resolved like the custom-code gate: the repo's config.json may differ from the snapshot. */
  modelCachePin?: ModelCachePin;
  resumeRunId?: string | null;
}): Promise<TrainingTransformersUpgradeOutcome> {
  let check: TransformersUpgradeCheck;
  try {
    check = await checkTransformersUpgrade(modelName, hfToken, {
      ...modelCachePin,
      resumeRunId,
    });
  } catch {
    return {
      proceed: true,
      error: null,
      forces16Bit: false,
      requiresTrustRemoteCode: false,
    };
  }
  const requiresTrustRemoteCode = Boolean(check.requiresTrustRemoteCode);
  if (!check.upgrade) {
    return {
      proceed: true,
      error: null,
      forces16Bit: check.forces16Bit,
      requiresTrustRemoteCode,
    };
  }
  const installable = upgradeInstallVersion(check.upgrade) !== null;
  if (check.installBreaksExactResume) {
    // The latest sidecar is a persistent overlay that refuses this checkpoint's 4-bit load.
    if (check.requiresTrustRemoteCode) {
      // The custom-code gate loads this on the current transformers in 4-bit. Nothing to offer.
      return {
        proceed: true,
        error: null,
        forces16Bit: false,
        requiresTrustRemoteCode,
      };
    }
    if (installable) {
      // Installing activates the latest tier, after which effective_training_load_in_4bit raises
      // (provenance.py), so consent buys only an irreversible overlay.
      return {
        proceed: false,
        error: getTrainingResumeUpgradeWouldStrandMessage(modelName),
        forces16Bit: false,
        requiresTrustRemoteCode,
      };
    }
    // Nothing installable (main's version unknown), so nothing can strand anything, and
    // "start a new run instead" cannot work either. Fall through to the retry message.
  }

  const upgraded = await confirmTransformersUpgradeIfNeeded({
    modelName,
    upgrade: check.upgrade,
    // With no installable release, custom code can still go through the trust_remote_code gate.
    trustRemoteCodeFallback: requiresTrustRemoteCode,
    // No forceCancelActive: training has no "stop N chats" answer, so the install refuses instead.
  });
  const installRan = useTransformersUpgradeDialogStore.getState().installRan;
  if (
    useTransformersUpgradeDialogStore.getState().consumeServerUnloadedChat()
  ) {
    // The install unloads the active chat model, so resync chat.
    void import("@/features/chat")
      .then((chat) => chat.resyncInferenceStatusAfterServerModelChange())
      .catch(() => undefined);
  }
  if (!upgraded) {
    // "Start again to install it" only means something when there is something to install.
    return {
      proceed: false,
      error: installable
        ? getTrainingTransformersUpgradeRequiredMessage(modelName)
        : getTrainingTransformersUpgradeUnavailableMessage(modelName),
      forces16Bit: false,
      requiresTrustRemoteCode,
    };
  }
  // Installed: the latest sidecar trains 16-bit. The custom-code fallback still loads 4-bit.
  return {
    proceed: true,
    error: null,
    forces16Bit: installRan,
    requiresTrustRemoteCode,
  };
}
