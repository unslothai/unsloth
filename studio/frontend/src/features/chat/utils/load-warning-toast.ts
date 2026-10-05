// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { toast } from "@/lib/toast";

/** One id, so the second call REPLACES rather than stacks. */
export const LOAD_WARNING_TOAST_ID = "model-load-warning";

export function showLoadWarning(warning: string | null | undefined): void {
  if (!warning) {
    toast.dismiss(LOAD_WARNING_TOAST_ID);
    return;
  }
  toast.warning("Model loaded with a warning", {
    id: LOAD_WARNING_TOAST_ID,
    description: warning,
    duration: 12000,
    closeButton: true,
  });
}
