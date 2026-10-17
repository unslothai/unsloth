// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { isAccountOwner } from "@/features/auth/account-session";
import { useSettingsDialogStore } from "../stores/settings-dialog-store";

/** "Open settings" on a load refused by Model memory. Owner only: Resources is an owner tab. */
export function modelMemorySettingsAction():
  | { label: string; onClick: () => void }
  | undefined {
  if (!isAccountOwner()) return undefined;
  return {
    label: "Open settings",
    onClick: () =>
      useSettingsDialogStore
        .getState()
        .openDialog("resources", { scrollTarget: "resources-model-memory" }),
  };
}
