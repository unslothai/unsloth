// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useNavigate } from "@tanstack/react-router";
import { useTrainingRuntimeStore } from "@/features/training";
import { useTrainingHistorySidebarItems } from "@/features/training/hooks/use-training-history-sidebar";
import { HistoryCardGrid } from "./history-card-grid";

/**
 * Recent training runs, surfaced on the Data Recipes and Export pages so a user
 * can jump back into a past run without first returning to Train. Selecting a
 * run stores its id and navigates to the Studio page, which auto-opens its
 * History tab (studio-page reacts to selectedHistoryRunId). Renders nothing once
 * we know there are no runs, so pages without any training history stay clean.
 */
export function RecentTrainingsSection() {
  const navigate = useNavigate();
  const setSelectedHistoryRunId = useTrainingRuntimeStore(
    (s) => s.setSelectedHistoryRunId,
  );
  const { items, loaded } = useTrainingHistorySidebarItems(true);

  if (loaded && items.length === 0) return null;

  return (
    <section className="mt-10">
      <h2 className="mb-4 text-[18px] font-semibold tracking-[-0.02em] text-foreground">
        Recent trainings
      </h2>
      <HistoryCardGrid
        onSelectRun={(runId) => {
          setSelectedHistoryRunId(runId);
          navigate({ to: "/studio" });
        }}
        onResumeStarted={() => navigate({ to: "/studio" })}
      />
    </section>
  );
}
