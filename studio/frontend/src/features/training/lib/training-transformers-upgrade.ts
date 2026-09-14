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
  /** The model ships its own modeling code. Hand this to the custom-code gate that runs
   *  next: its fallback otherwise reads the stored config flag, false for a fresh run,
   *  and skips consent for a model that cannot load without it. Chat does the same. */
  requiresTrustRemoteCode: boolean;
}

/** What the Configure preview must disclose about transformers and precision. */
export interface TrainingTransformersUpgradeNotice {
  /** transformers release the run would offer to install, or null when none is needed. */
  installVersion: string | null;
  /** The picked method asks for 4-bit but the run would load 16-bit anyway. */
  fourBitUnavailable: boolean;
  /** 4-bit survives only if the user keeps the model's own code: taking Install instead
   *  activates the latest sidecar, which trains 16-bit. */
  installSwitchesTo16Bit: boolean;
}

/** Turn one upgrade check into the preview's disclosure.
 *
 * A model that both ships its own modeling code and is shipped by the offered release
 * gets both dialog actions, and they differ in precision: the custom-code fallback loads
 * 4-bit on the current transformers (so forces_16bit is false), while Install activates
 * the latest sidecar, which trains 16-bit. Disclose that choice rather than advertise
 * "QLoRA - 4-bit" for a run the user may well take at triple the VRAM.
 *
 * `installable && !forces16Bit` catches exactly that case: forces_16bit is
 * `latest_tier_active || install_only_upgrade`, and install_only_upgrade is itself
 * installable-and-no-custom-code. */
export function trainingTransformersUpgradeNotice(
  check: TransformersUpgradeCheck,
  loadsIn4Bit: boolean,
): TrainingTransformersUpgradeNotice {
  const installable = Boolean(
    check.upgrade?.supported_in_pypi && check.upgrade?.pypi_version,
  );
  return {
    installVersion: installable ? (check.upgrade?.pypi_version ?? null) : null,
    fourBitUnavailable: check.forces16Bit && loadsIn4Bit,
    installSwitchesTo16Bit: installable && !check.forces16Bit && loadsIn4Bit,
  };
}

/** Names the worker's own failure ("... is not supported yet in transformers==x.y.z")
 *  rather than a generic start error, and says what to do about it. */
export function getTrainingTransformersUpgradeRequiredMessage(
  modelName: string,
): string {
  return `${modelName} is not supported yet by the installed transformers, and the newer release it needs was not installed. Start the run again to install it.`;
}

/** Dev-only: the architecture is on the transformers development branch and no PyPI
 *  release ships it, so the dialog has no Install action. "Start again to install it"
 *  would be an instruction the app can never carry out. */
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
 * no installed transformers ships was accepted, spawned, and killed minutes later at
 * model load with an error the user could not act on. Same dialog, raised before the
 * run starts. Additive: a backend that does not serve the check, or fails it, leaves
 * the start as it was. */
export async function confirmTrainingTransformersUpgrade({
  modelName,
  hfToken,
  modelCachePin,
  resumeRunId,
}: {
  modelName: string;
  hfToken?: string | null;
  /** Which copy the run will load, resolved exactly as the custom-code gate resolves
   *  it: a repo's current config.json often names a different architecture than the
   *  pinned snapshot on disk. */
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
  // Only a released version is ever installed, so every branch that talks about
  // installing checks this first.
  const installable = Boolean(
    check.upgrade.supported_in_pypi && check.upgrade.pypi_version,
  );
  if (check.installBreaksExactResume) {
    // The checkpoint is attested against a 4-bit load the latest sidecar refuses, and
    // that sidecar is a persistent overlay: consenting would strand it for good.
    if (check.requiresTrustRemoteCode) {
      // The custom-code gate that runs next loads this on the CURRENT transformers, in
      // the 4-bit mode the checkpoint needs. Nothing to offer.
      return {
        proceed: true,
        error: null,
        forces16Bit: false,
        requiresTrustRemoteCode,
      };
    }
    if (installable) {
      // No fallback, and the install is not the way out it looks like: it activates the
      // latest tier, after which effective_training_load_in_4bit raises for this config
      // (provenance.py gates on the same disjunction). The resume fails either way, so
      // consent buys only an irreversible overlay. Say so instead of offering it.
      return {
        proceed: false,
        error: getTrainingResumeUpgradeWouldStrandMessage(modelName),
        forces16Bit: false,
        requiresTrustRemoteCode,
      };
    }
    // Dev-only: no release to install, so nothing can strand anything, and "start a new
    // run instead" cannot work either. Fall through to the dev-only path, which says the
    // true thing: wait for the next release.
  }

  const upgraded = await confirmTransformersUpgradeIfNeeded({
    modelName,
    upgrade: check.upgrade,
    // With no installable release, a model shipping its own code can still go through
    // the trust_remote_code gate the caller runs next, exactly as chat does.
    trustRemoteCodeFallback: requiresTrustRemoteCode,
    // No forceCancelActive: training raises no "stop N chats" prompt, so it has no such
    // answer to carry. A chat mid-generation makes the install refuse and the dialog
    // says so, rather than this tab killing someone else's stream unasked.
  });
  // Read before the resolve-time state is reused by any later consent.
  const installRan = useTransformersUpgradeDialogStore.getState().installRan;
  if (
    useTransformersUpgradeDialogStore.getState().consumeServerUnloadedChat()
  ) {
    // The install unloads the active chat model. Nothing on this tab owns that
    // selection, so resync or chat keeps pointing at a model that is gone.
    void import("@/features/chat")
      .then((chat) => chat.resyncInferenceStatusAfterServerModelChange())
      .catch(() => undefined);
  }
  if (!upgraded) {
    // "Start again to install it" only means something when there is something to
    // install; a dev-only upgrade would send the user round a loop that never ends.
    return {
      proceed: false,
      error: installable
        ? getTrainingTransformersUpgradeRequiredMessage(modelName)
        : getTrainingTransformersUpgradeUnavailableMessage(modelName),
      forces16Bit: false,
      requiresTrustRemoteCode,
    };
  }
  // Installed: the model now routes to the latest sidecar, which trains 16-bit. The
  // custom-code fallback resolves true without installing and still loads 4-bit.
  return {
    proceed: true,
    error: null,
    forces16Bit: installRan,
    requiresTrustRemoteCode,
  };
}
