// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type {
  TransformersUpgradeInfo,
  TransformersUpgradePhase,
} from "../types";

/** Which ways out the consent dialog offers, besides Cancel. */
export interface UpgradeDialogActions {
  /** A released version exists, so the install is a real action. */
  installable: boolean;
  /** Only transformers main ships the architecture; nothing to install. */
  devOnly: boolean;
  /** Offer "Continue with custom code": the model's own modeling code loads it on the
   *  transformers already installed, so the caller's trust_remote_code gate is a way out. */
  customCode: boolean;
}

/** Decide the dialog's actions from the check that raised it.
 *
 * The custom-code fallback used to appear only with nothing to install, or after an
 * install failed. It reads as a courtesy but is a correctness rule: the fallback loads
 * on the CURRENT transformers, the only path that still loads bnb 4-bit, since
 * installing activates the 16-bit sidecar. Hiding it behind Install leaves a QLoRA run
 * no way to start at its own precision. Whenever the fallback exists, it is offered. */
export function upgradeDialogActions({
  upgrade,
  phase,
  trustRemoteCodeFallback,
}: {
  upgrade: TransformersUpgradeInfo | null;
  phase: TransformersUpgradePhase;
  trustRemoteCodeFallback: boolean;
}): UpgradeDialogActions {
  const installable = Boolean(
    upgrade?.supported_in_pypi && upgrade?.pypi_version,
  );
  return {
    installable,
    devOnly: !installable && Boolean(upgrade?.supported_in_main),
    // Never mid-install: the install is running and this button would abandon it.
    customCode: trustRemoteCodeFallback && phase !== "installing",
  };
}
