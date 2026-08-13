// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  type ModelCachePin,
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

/** The dev-only case: the architecture is on the transformers development branch and no
 *  PyPI release ships it, so the dialog has no Install action at all. Telling the user to
 *  start again to install it would be an instruction the app can never carry out. */
export function getTrainingTransformersUpgradeUnavailableMessage(
  modelName: string,
): string {
  return `${modelName} is not supported yet by the installed transformers, and no released transformers version supports it either: the architecture is only on the transformers development branch, which Unsloth does not install. Wait for the next transformers release, or pick a model the installed transformers supports.`;
}

/** The resume that installing would strand: the checkpoint is attested against a 4-bit
 *  model load the latest sidecar permanently refuses, so the install cannot rescue it. */
export function getTrainingResumeUpgradeWouldStrandMessage(
  modelName: string,
): string {
  return `${modelName} is not supported yet by the installed transformers, and installing the release it needs would permanently retire the 4-bit model load this checkpoint was attested with, so the resume would still fail. Start a new run on this model instead.`;
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
  modelCachePin,
  resumeRunId,
}: {
  modelName: string;
  hfToken?: string | null;
  /** Which copy of the model the run will load, resolved exactly as the custom-code
   *  gate resolves it: a repo's current config.json describes a different architecture
   *  than the pinned snapshot on disk often enough to gate on the wrong one. */
  modelCachePin?: ModelCachePin;
  /** Set on the resume path, so the check can say whether installing would strand
   *  this checkpoint. */
  resumeRunId?: string | null;
}): Promise<TrainingTransformersUpgradeOutcome> {
  let check: TransformersUpgradeCheck;
  try {
    check = await checkTransformersUpgrade(modelName, hfToken, {
      ...modelCachePin,
      resumeRunId,
    });
  } catch {
    return { proceed: true, error: null, forces16Bit: false };
  }
  if (!check.upgrade) {
    return { proceed: true, error: null, forces16Bit: check.forces16Bit };
  }
  if (check.installBreaksExactResume) {
    // This resume is attested against a 4-bit model load the latest sidecar refuses,
    // and that sidecar is a persistent overlay: consenting here would strand the
    // checkpoint for good.
    if (check.requiresTrustRemoteCode) {
      // The model ships its own modeling code, so the custom-code gate that runs next
      // loads it on the CURRENT transformers, in the 4-bit mode the checkpoint needs.
      // Nothing to offer, so offer nothing.
      return { proceed: true, error: null, forces16Bit: false };
    }
    // No fallback either, and the install is not the way out it looks like: it activates
    // the latest tier, after which effective_training_load_in_4bit RAISES for exactly
    // this config (provenance.py gates on the same disjunction the backend answered
    // install_breaks_exact_resume with). The resume fails whether or not the user
    // consents, so the only thing consent buys is an irreversible overlay. Say so
    // instead of offering it.
    return {
      proceed: false,
      error: getTrainingResumeUpgradeWouldStrandMessage(modelName),
      forces16Bit: false,
    };
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
    // "Start again to install it" only means something when there is something to
    // install. A dev-only upgrade gives the dialog no Install action, so the same text
    // would send the user round a loop that can never end.
    const installable = Boolean(
      check.upgrade.supported_in_pypi && check.upgrade.pypi_version,
    );
    return {
      proceed: false,
      error: installable
        ? getTrainingTransformersUpgradeRequiredMessage(modelName)
        : getTrainingTransformersUpgradeUnavailableMessage(modelName),
      forces16Bit: false,
    };
  }
  // Installed: the model now routes to the latest sidecar, which trains 16-bit. The
  // custom-code fallback resolves true WITHOUT installing and still loads 4-bit.
  return { proceed: true, error: null, forces16Bit: installRan };
}
