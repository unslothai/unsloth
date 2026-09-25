// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { toast } from "sonner";

export function showForkCreatedToast(containerSnapshotWarning: string | null): void {
  if (containerSnapshotWarning) {
    toast.info("Fork created", { description: containerSnapshotWarning });
  } else {
    toast.success("Fork created");
  }
}
