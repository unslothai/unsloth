// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  downloadManager,
  useDownloadManagerStore,
} from "./download-manager-controller";
import { presentedProgress } from "./download-presentation";
import { DownloadProgressBar } from "./download-progress-bar";
import { assetLabel } from "./required-assets";
import type { Plan } from "./use-hub-download-queue";

/** "Model file · 1 of 2" while a step runs, "Queued" before. */
export function hubPlanStepLabel(plan: Plan, running: boolean): string {
  const entry = plan.remaining[0];
  if (!entry || !running) return "Queued";
  const what =
    entry.checkpoint !== false ? "Model file" : assetLabel(entry, "Required assets");
  const steps = plan.entries.length;
  return steps > 1
    ? `${what} · ${steps - plan.remaining.length + 1} of ${steps}`
    : what;
}

// The bar is the running step's, with the same numbers as its row in the downloads panel: a
// plan-wide total beside that row read as the two disagreeing (#14061).
export function HubPlanProgress({ plan }: { plan: Plan }) {
  const entry = plan.remaining[0];
  const job = useDownloadManagerStore((s) =>
    Object.values(s.jobs).find(
      (j) =>
        j.repoId === entry?.repoId &&
        j.variant === "@hub-assets" &&
        (j.state === "running" || j.state === "cancelling") &&
        entry.files.length === j.scopedFiles?.length &&
        entry.files.every((f) => j.scopedFiles?.includes(f)),
    ),
  );
  return (
    <div className="px-3">
      <div className="mb-2 flex justify-between text-ui-11 text-muted-foreground">
        <span>{hubPlanStepLabel(plan, Boolean(job))}</span>
        {job && (
          <button
            type="button"
            disabled={job.state === "cancelling"}
            onClick={() => void downloadManager.cancel(job.key)}
          >
            Cancel download
          </button>
        )}
      </div>
      <DownloadProgressBar
        progress={
          job
            ? presentedProgress(job)
            : { expectedBytes: Math.max(0, entry?.bytes ?? 0), downloadedBytes: 0, fraction: 0 }
        }
        bytesPerSec={job?.bytesPerSec ?? 0}
        cancelling={job?.state === "cancelling"}
        etaSeconds={job?.etaSeconds}
        activity={job?.activity}
      />
    </div>
  );
}
