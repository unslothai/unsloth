// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type {
  TransformersUpgradeInfo,
  TransformersUpgradePhase,
} from "../types";

export interface UpgradeDialogActions {
  installable: boolean;
  devOnly: boolean;
  customCode: boolean;
}

/** Always offer the custom-code fallback when it exists: it is the only path that keeps bnb 4-bit,
 * since installing activates the 16-bit sidecar. */
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
    // Never mid-install: this button would abandon it.
    customCode: trustRemoteCodeFallback && phase !== "installing",
  };
}
