// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Suspense, lazy } from "react";
import { StatsSkeleton } from "./stats-skeleton";

// The stats content pulls in recharts for the rhythm charts. Settings live in
// the main bundle, so importing it eagerly would move ~300 KB of charting off
// its own lazy chunk and onto every cold app load. Split it here instead: the
// chunk is fetched only when someone actually opens Settings -> Profile.
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
