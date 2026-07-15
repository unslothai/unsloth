// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useTransformersUpgradeDialogStore } from "../stores/transformers-upgrade-dialog-store";
import type { TransformersUpgradeInfo } from "../types";

interface ConfirmArgs {
  modelName: string;
  /** validate's transformers_upgrade payload; null/undefined skips the dialog. */
  upgrade: TransformersUpgradeInfo | null | undefined;
  /** When no release is installable, offer continuing into the caller's custom-code gate. */
  trustRemoteCodeFallback?: boolean;
}

/** Pause a load needing a newer transformers on the consent dialog and run the install.
 *  Resolves true when the load can continue; false on cancel or not-installable with no fallback. */
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
