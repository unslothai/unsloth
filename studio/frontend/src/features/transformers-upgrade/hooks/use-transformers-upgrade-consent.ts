// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useTransformersUpgradeDialogStore } from "../stores/transformers-upgrade-dialog-store";
import type { TransformersUpgradeInfo } from "../types";

interface ConfirmArgs {
  modelName: string;
  /** validate's transformers_upgrade payload; pass null/undefined when
   *  requires_transformers_upgrade is false to proceed without a dialog. */
  upgrade: TransformersUpgradeInfo | null | undefined;
  /** validate's requires_trust_remote_code: with no installable release the
   *  dialog then offers continuing into the custom-code consent flow (the
   *  caller's security gate) instead of hard-aborting the load. */
  trustRemoteCodeFallback?: boolean;
}

/** Gate a load whose architecture needs a newer transformers: pause on the consent
 *  dialog and run the consented sidecar install. Resolves true when the load can
 *  continue (no upgrade needed, the install succeeded, or the user chose the
 *  custom-code fallback); false when the user cancels or the architecture is not
 *  installable yet (dev-only on main) with no custom code to fall back to. */
export async function confirmTransformersUpgradeIfNeeded({
  modelName,
  upgrade,
  trustRemoteCodeFallback,
}: ConfirmArgs): Promise<boolean> {
  if (!upgrade) return true;
  return useTransformersUpgradeDialogStore
    .getState()
    .requestConsent(modelName, upgrade, {
      trustRemoteCodeFallback: Boolean(trustRemoteCodeFallback),
    });
}
