


import { Suspense, lazy } from "react";
import { StatsSkeleton } from "./stats-skeleton";

// Settings live in the main bundle, so keep the profile aggregation UI in its
// own chunk. It is fetched only when someone opens Settings -> Profile.
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
