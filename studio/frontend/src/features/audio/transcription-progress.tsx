// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import type { TranscriptProgress } from "./transcript-stream";
import "./transcription-progress.css";

export function TranscriptionProgress({
  startedAt,
  finishedAt,
  stopping,
  progress,
  onCancel,
}: {
  startedAt: number;
  finishedAt: number | null;
  stopping: boolean;
  progress: TranscriptProgress | null;
  onCancel: () => void;
}) {
  const [now, setNow] = useState(Date.now);
  useEffect(() => {
    if (finishedAt !== null) return;
    const timer = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, [finishedAt]);
  const seconds = Math.max(
    0,
    Math.floor(((finishedAt ?? now) - startedAt) / 1000),
  );
  const elapsed = `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, "0")}`;
  const running = finishedAt === null;
  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center gap-3 text-ui-13 text-muted-foreground">
        {running && (
          <span
            className="transcription-wave flex h-5 items-center gap-[calc(3px*var(--ui-space-scale,1))]"
            aria-hidden="true"
            data-stopping={stopping}
          >
            {[10, 16, 20, 14, 8].map((height, index) => (
              <span
                key={index}
                className="w-[3px] rounded-full bg-current"
                style={{ height, animationDelay: `${index * -230}ms` }}
              />
            ))}
          </span>
        )}
        <span role="status">
          {running ? (stopping ? "Stopping…" : "Transcribing…") : "Elapsed"}
        </span>
        <span className="tabular-nums" aria-label={`Elapsed time ${elapsed}`}>
          {elapsed}
        </span>
        {running && (
          <Button
            variant="ghost"
            size="sm"
            className="ml-auto"
            onClick={onCancel}
            disabled={stopping}
          >
            Cancel
          </Button>
        )}
      </div>
      {running && progress?.text && (
        <p className="line-clamp-2 text-ui-13 leading-relaxed text-muted-foreground">
          {progress.text.split(/\s+/).slice(-35).join(" ")}
        </p>
      )}
    </div>
  );
}
