// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  downloadManager,
  useDownloadManagerStore,
} from "./download-manager-controller";
import { DownloadProgressBar } from "./download-progress-bar";
import type { Plan } from "./use-hub-download-queue";
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
  const total = plan.entries.reduce((n, e) => n + Math.max(0, e.bytes), 0);
  const remaining = plan.remaining.reduce(
    (n, e) => n + Math.max(0, e.bytes),
    0,
  );
  const downloaded =
    total - remaining + Math.min(entry?.bytes ?? 0, job?.downloadedBytes ?? 0);
  return (
    <div className="px-3">
      <div className="mb-2 flex justify-between text-ui-11 text-muted-foreground">
        <span>
          {job
            ? entry.checkpoint !== false
              ? "Downloading model file"
              : "Downloading required files"
            : "Queued"}
        </span>
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
        progress={{
          expectedBytes: total,
          downloadedBytes: downloaded,
          fraction: total > 0 ? downloaded / total : 0,
        }}
        bytesPerSec={job?.bytesPerSec ?? 0}
        cancelling={job?.state === "cancelling"}
      />
    </div>
  );
}
