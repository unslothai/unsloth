// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { ChevronRightStandardIcon } from "@/lib/chevron-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useSettingsDialogStore } from "../stores/settings-dialog-store";
import { SettingsSection } from "./settings-section";

// The switches live on Remote & LAN, beside the exposure they depend on. English like that section.
export function KeylessApiAccessLink() {
  return (
    <SettingsSection
      title="Keyless API access"
      description="Use Unsloth from other apps without creating an API key. Turn it on in Remote & LAN."
      action={
        <button
          type="button"
          onClick={() =>
            useSettingsDialogStore
              .getState()
              .openDialog("remote-lan", { scrollTarget: "remote-lan-keyless" })
          }
          className="inline-flex items-center gap-1 rounded-full px-1.5 py-0.5 text-ui-11 font-medium text-muted-foreground transition-colors hover:text-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
        >
          Open in Remote &amp; LAN
          <HugeiconsIcon icon={ChevronRightStandardIcon} className="size-3" />
        </button>
      }
    >
      {null}
    </SettingsSection>
  );
}
