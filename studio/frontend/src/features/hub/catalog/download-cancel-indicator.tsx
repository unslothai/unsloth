// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Cancel01Icon, PauseIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";

/** HTTP keeps the partial (pause); Xet starts over (cancel). */
export type DownloadStopMode = "cancel" | "pause";

/** Always visible since it is the only way to stop; fixed slot so the percentage never shifts. */
export function DownloadStopIndicator({ mode }: { mode: DownloadStopMode }) {
  return (
    <span className="hub-cta-indicator">
      <HugeiconsIcon
        icon={mode === "pause" ? PauseIcon : Cancel01Icon}
        strokeWidth={1.75}
      />
    </span>
  );
}
