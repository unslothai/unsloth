


import { Skeleton } from "@/components/ui/skeleton";

/**
 * Placeholder for the stats panel.
 *
 * Its own module so the lazy wrapper can render it as a Suspense fallback
 * without pulling the chart-bearing content chunk into the main bundle.
 */
export function StatsSkeleton() {
  return (
    <div className="flex flex-col gap-4">
      <Skeleton className="h-20 w-full rounded-2xl" />
      <Skeleton className="h-24 w-full rounded-2xl" />
      <Skeleton className="h-56 w-full rounded-2xl" />
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <Skeleton className="h-72 w-full rounded-2xl" />
        <Skeleton className="h-72 w-full rounded-2xl" />
      </div>
    </div>
  );
}
