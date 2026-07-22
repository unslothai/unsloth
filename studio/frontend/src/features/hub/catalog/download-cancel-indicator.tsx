// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Spinner } from "@/components/ui/spinner";
import { Cancel01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";

/**
 * Inspector action-button affordance during a download: spinner that cross-fades
 * to a cancel glyph on `.hub-action-btn` hover, in the same 16x16 slot so the
 * percentage label never shifts. The swap is pure CSS
 * (`.hub-action-btn:hover .hub-cta-indicator-*`); the component only carries
 * the marker classes.
 */
export function DownloadCancelIndicator() {
  return (
    <span className="hub-cta-indicator">
      <Spinner className="hub-cta-indicator-spinner" />
      <HugeiconsIcon
        icon={Cancel01Icon}
        strokeWidth={1.75}
        className="hub-cta-indicator-cancel"
      />
    </span>
  );
}
