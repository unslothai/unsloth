// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type {
  TransformersUpgradeInfo,
  TransformersUpgradePhase,
} from "../types";

export interface UpgradeDialogActions {
  /** A released version, or transformers main, can be installed. */
  installable: boolean;
  /** The install is transformers main: no release ships the architecture yet. */
  fromMain: boolean;
  /** Only transformers main ships the architecture and its version is unknown. */
  devOnly: boolean;
  customCode: boolean;
}

/** The version an install sends: the PyPI release, else transformers main. */
export function upgradeInstallVersion(
  upgrade: TransformersUpgradeInfo | null,
): string | null {
  if (upgrade?.supported_in_pypi && upgrade?.pypi_version) {
    return upgrade.pypi_version;
  }
  if (upgrade?.supported_in_main && upgrade?.main_version) {
    return upgrade.main_version;
  }
  return null;
}

/**
 * Derives the dialog actions from the check that raised it; the custom-code fallback is always
 * offered when it exists, as the only path loading bnb 4-bit (installing activates the 16-bit
 * sidecar).
 */
export function upgradeDialogActions({
  upgrade,
  phase,
  trustRemoteCodeFallback,
}: {
  upgrade: TransformersUpgradeInfo | null;
  phase: TransformersUpgradePhase;
  trustRemoteCodeFallback: boolean;
}): UpgradeDialogActions {
  const fromPypi = Boolean(upgrade?.supported_in_pypi && upgrade?.pypi_version);
  const fromMain =
    !fromPypi && Boolean(upgrade?.supported_in_main && upgrade?.main_version);
  const installable = fromPypi || fromMain;
  return {
    installable,
    fromMain,
    devOnly: !installable && Boolean(upgrade?.supported_in_main),
    // Never mid-install: this button would abandon it.
    customCode: trustRemoteCodeFallback && phase !== "installing",
  };
}
