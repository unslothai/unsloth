// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { DiffusionStatus } from "../api";

export const HELD_BY_OTHER_ACCOUNT_NOTICE =
  "Another account has an image model loaded. Select and load a model to generate with your own.";

/**
 * The status as this account sees it. Another account's resident answers `{loaded: true, yours: false}`
 * with nothing else (#16178): generating or ejecting against it fails, so the page reads it as idle.
 */
export function ownDiffusionStatus(status: DiffusionStatus): DiffusionStatus {
  return status.yours === false ? { ...status, loaded: false } : status;
}

export function heldByOtherAccount(status: DiffusionStatus | null): boolean {
  return status?.yours === false;
}
