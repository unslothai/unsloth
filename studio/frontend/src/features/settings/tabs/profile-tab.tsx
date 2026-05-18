// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { ProfilePersonalizationPanel } from "@/features/profile";

export function ProfileTab() {
  return (
    <div className="flex flex-col gap-6">
      <header className="flex flex-col gap-1">
        <h1 className="text-lg font-semibold font-heading">Profile</h1>
        <p className="text-xs text-muted-foreground">
          Update how your profile appears in Studio.
        </p>
      </header>

      <ProfilePersonalizationPanel />
    </div>
  );
}
