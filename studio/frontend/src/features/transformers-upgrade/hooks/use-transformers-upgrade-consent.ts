// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useTransformersUpgradeDialogStore } from "../stores/transformers-upgrade-dialog-store";
import type { TransformersUpgradeInfo } from "../types";

interface ConfirmArgs {
  modelName: string;
  /** null/undefined skips the dialog. */
  upgrade: TransformersUpgradeInfo | null | undefined;
  trustRemoteCodeFallback?: boolean;
  /** Carry the user's "stop N chats" answer, or the install 409s on those chats. */
  forceCancelActive?: boolean;
}

/** False on cancel, or when nothing is installable and there is no fallback. */
export async function confirmTransformersUpgradeIfNeeded({
  modelName,
  upgrade,
  trustRemoteCodeFallback,
  forceCancelActive,
}: ConfirmArgs): Promise<boolean> {
  if (!upgrade) return true;
  return useTransformersUpgradeDialogStore
    .getState()
    .requestConsent(modelName, upgrade, {
      trustRemoteCodeFallback: Boolean(trustRemoteCodeFallback),
      forceCancelActive: Boolean(forceCancelActive),
    });
}
