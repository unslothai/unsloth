// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Suspense, lazy } from "react";
import { StatsSkeleton } from "./stats-skeleton";

// Lazy chunk: Settings is in the main bundle, and this is only needed on the Profile tab.
const ProfileStatsContent = lazy(() =>
  import("./profile-stats-content").then((module) => ({
    default: module.ProfileStatsContent,
  })),
);

export function ProfileStatsPanel() {
  return (
    <Suspense fallback={<StatsSkeleton />}>
      <ProfileStatsContent />
    </Suspense>
  );
}
