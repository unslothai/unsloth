// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { DownloadStartOutcome } from "../download-manager/transport-conflict";

export interface UpdateStartFeedback {
  title: string;
  description: string;
}

export function busyUpdateStartFeedback(
  outcome: DownloadStartOutcome,
): UpdateStartFeedback | null {
  return outcome === "busy"
    ? {
        title: "A download for this model is already in progress",
        description: "Try updating again once it finishes.",
      }
    : null;
}
