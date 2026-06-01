// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Spinner } from "@/components/ui/spinner";
import { Cancel01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";

/**
 * In-progress affordance for the inspector action button while a download
 * is running. By default the spinner indicates work; on `.hub-action-btn`
 * hover it cross-fades to the cancel glyph so the click target reads as
 * "stop this download". Both glyphs occupy the same 16×16 slot, so the
 * percentage label beside them never shifts.
 *
 * The hover swap is driven entirely by CSS state selectors in
 * `.hub-action-btn:hover .hub-cta-indicator-*` — no JS hover wiring and no
 * Tailwind group/* variants. The component only carries the marker classes.
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
