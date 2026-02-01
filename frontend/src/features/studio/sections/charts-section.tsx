import { useWizardStore } from "@/stores/training";
import { type ReactElement, Suspense, lazy } from "react";

const ChartsContent = lazy(() =>
  import("./charts-content").then((module) => ({
    default: module.ChartsContent,
  })),
);
const SKELETON_KEYS = [
  "chart-skeleton-1",
  "chart-skeleton-2",
  "chart-skeleton-3",
  "chart-skeleton-4",
];

export function ChartsSection(): ReactElement | null {
  const metrics = useWizardStore((s) => s.trainingMetrics);

  if (!metrics) {
    return null;
  }

  return (
    <Suspense
      fallback={
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {SKELETON_KEYS.map((key) => (
            <div
              key={key}
              className="h-[280px] rounded-xl border bg-muted/30 animate-pulse"
            />
          ))}
        </div>
      }
    >
      <ChartsContent metrics={metrics} />
    </Suspense>
  );
}
