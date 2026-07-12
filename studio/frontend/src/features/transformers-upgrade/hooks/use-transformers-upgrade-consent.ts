// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useTransformersUpgradeDialogStore } from "../stores/transformers-upgrade-dialog-store";
import type { TransformersUpgradeInfo } from "../types";

interface ConfirmArgs {
  modelName: string;
  /** validate's transformers_upgrade payload; pass null/undefined when
   *  requires_transformers_upgrade is false to proceed without a dialog. */
  upgrade: TransformersUpgradeInfo | null | undefined;
}

/** Gate a load whose architecture needs a newer transformers: pause on the consent
 *  dialog and run the consented sidecar install. Resolves true when the load can
 *  continue (no upgrade needed, or the install succeeded); false when the user
 *  cancels or the architecture is not installable yet (dev-only on main). */
export async function confirmTransformersUpgradeIfNeeded({
  modelName,
  upgrade,
}: ConfirmArgs): Promise<boolean> {
  if (!upgrade) return true;
  return useTransformersUpgradeDialogStore
    .getState()
    .requestConsent(modelName, upgrade);
}
