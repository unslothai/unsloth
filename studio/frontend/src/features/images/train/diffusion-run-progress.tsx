// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect, useState } from "react";
import { Progress } from "@/components/ui/progress";
import { Spinner } from "@/components/ui/spinner";
import type { DiffusionTrainingStatus } from "../api";
import type { diffusionTrainingProgress } from "./diffusion-train-progress";

export function DiffusionRunProgress({ progress, status, active }: {
  progress: ReturnType<typeof diffusionTrainingProgress>;
  status: DiffusionTrainingStatus | null;
  active: boolean;
}) {
  const waiting = progress.preparing && progress.percent === null;
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!waiting || !active) return;
    setNow(Date.now());
    const timer = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(timer);
  }, [waiting, active, status?.job_id]);
  const elapsed = typeof status?.started_at === "number"
    ? Math.max(0, Math.floor(now / 1000 - status.started_at)) : null;
  const count = waiting && elapsed !== null
    ? `Elapsed ${Math.floor(elapsed / 60)}m ${elapsed % 60}s`
    : progress.count;
  return (
    <section className="flex flex-col gap-3" aria-label="Training progress">
      <div className="flex items-center justify-between gap-3">
        <h2 className="flex items-center gap-2 text-sm font-semibold" aria-live="polite">
          {waiting && <Spinner className="size-3.5 text-muted-foreground" label="Waiting for measured progress" />}
          {progress.title}
        </h2>
        <span className="text-xs tabular-nums text-muted-foreground">{count}</span>
      </div>
      <Progress className="h-2" value={progress.percent ?? undefined} aria-label={progress.title} />
      {progress.detail && <p role="status" className="text-ui-11 text-muted-foreground">{progress.detail}</p>}
    </section>
  );
}
