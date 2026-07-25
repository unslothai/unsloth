// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Progress } from "@/components/ui/progress";
import type { CheckpointUploadProgress } from "@/features/training";
import { useT, type TranslationKey } from "@/i18n";
import { cn } from "@/lib/utils";
import {
  AlertCircle,
  CheckCircle2,
  ExternalLink,
  MinusCircle,
  UploadCloud,
  X,
} from "lucide-react";
import { useState, type ReactElement } from "react";

function formatBytes(value: number): string {
  if (value < 1024) return `${value} B`;
  if (value < 1024 ** 2) return `${(value / 1024).toFixed(1)} KiB`;
  if (value < 1024 ** 3) return `${(value / 1024 ** 2).toFixed(1)} MiB`;
  return `${(value / 1024 ** 3).toFixed(1)} GiB`;
}

const stateLabel = {
  idle: "studio.checkpointUpload.title",
  preparing: "studio.checkpointUpload.preparing",
  uploading: "studio.checkpointUpload.uploading",
  completed: "studio.checkpointUpload.completed",
  skipped: "studio.checkpointUpload.skipped",
  error: "studio.checkpointUpload.transferError",
} satisfies Record<CheckpointUploadProgress["state"], TranslationKey>;

function transferText(upload: CheckpointUploadProgress): string | null {
  if (upload.uploaded_bytes != null) {
    return upload.total_bytes != null
      ? `${formatBytes(upload.uploaded_bytes)} / ${formatBytes(upload.total_bytes)}`
      : formatBytes(upload.uploaded_bytes);
  }
  if (upload.uploaded_files != null) {
    return upload.total_files != null
      ? `${upload.uploaded_files} / ${upload.total_files} files`
      : `${upload.uploaded_files} files`;
  }
  return null;
}

export function CheckpointUploadCard({
  upload,
}: {
  upload: CheckpointUploadProgress;
}): ReactElement | null {
  const t = useT();
  const [dismissedCheckpoint, setDismissedCheckpoint] = useState<string | null>(null);
  // Checkpoint names are stable across lifecycle updates. Dismiss only this
  // checkpoint so the next checkpoint-N upload automatically becomes visible.
  const dismissalKey = upload.checkpoint ?? `${upload.state}:${upload.repository_id ?? ""}`;
  const percent =
    typeof upload.percentage === "number"
      ? Math.min(100, Math.max(0, upload.percentage))
      : null;
  const active = upload.state === "preparing" || upload.state === "uploading";
  const repositoryUrl = upload.repository_url?.startsWith("https://huggingface.co/")
    ? upload.repository_url.replace(/\/+$/, "")
    : null;
  const checkpointUrl =
    repositoryUrl && upload.checkpoint
      ? `${repositoryUrl}/tree/main/${encodeURIComponent(upload.checkpoint)}`
      : null;
  const transferred = transferText(upload);
  const StateIcon =
    upload.state === "completed"
      ? CheckCircle2
      : upload.state === "error"
        ? AlertCircle
        : upload.state === "skipped"
          ? MinusCircle
          : UploadCloud;

  if (dismissedCheckpoint === dismissalKey) {
    return null;
  }

  return (
    <aside
      aria-label={t("studio.checkpointUpload.title")}
      className={cn(
        "relative rounded-xl border bg-card px-3 py-2.5 shadow-xs",
        checkpointUrl && upload.state === "completed" && "transition-colors hover:bg-muted/40",
        upload.state === "error" && "border-destructive/30 bg-destructive/[0.03]",
      )}
    >
      {checkpointUrl && upload.state === "completed" ? (
        <a
          href={checkpointUrl}
          target="_blank"
          rel="noopener noreferrer"
          aria-label={`${t("studio.checkpointUpload.checkpoint")}: ${upload.checkpoint}`}
          className="absolute inset-0 rounded-xl focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
        />
      ) : null}
      <div className="pointer-events-none relative flex min-w-0 items-center gap-2.5">
        <div
          className={cn(
            "grid size-7 shrink-0 place-items-center rounded-lg bg-muted text-muted-foreground",
            active && "bg-control-accent/10 text-control-accent",
            upload.state === "completed" && "bg-emerald-500/10 text-emerald-600",
            upload.state === "error" && "bg-destructive/10 text-destructive",
          )}
        >
          <StateIcon className={cn("size-3.5", active && "animate-pulse")} />
        </div>

        <div className="min-w-0 flex-1">
          <div className="flex min-w-0 items-center gap-2 text-xs">
            <span className="shrink-0 font-medium">{t(stateLabel[upload.state])}</span>
            {upload.checkpoint ? (
              <span className="truncate text-muted-foreground">{upload.checkpoint}</span>
            ) : null}
            {percent !== null ? (
              <span className="ml-auto shrink-0 tabular-nums text-muted-foreground">
                {percent.toFixed(0)}%
              </span>
            ) : null}
          </div>

          {active ? (
            percent !== null ? (
              <Progress value={percent} className="mt-1.5 h-1" />
            ) : (
              <div className="mt-1.5 h-1 overflow-hidden rounded-full bg-foreground/[0.06]">
                <div className="h-full w-1/3 animate-pulse rounded-full bg-control-accent" />
              </div>
            )
          ) : null}

          <div className="mt-1 flex min-w-0 items-center gap-2 text-[11px] text-muted-foreground">
            {upload.message ? <span className="truncate">{upload.message}</span> : null}
            {upload.repository_id ? (
              <span
                className={cn(
                  "truncate",
                  upload.message && "before:mr-2 before:content-['·']",
                )}
              >
                {upload.repository_id}
              </span>
            ) : null}
            {transferred ? <span className="ml-auto shrink-0 tabular-nums">{transferred}</span> : null}
          </div>
        </div>

        {checkpointUrl && upload.state === "completed" ? (
          <span className="grid size-7 shrink-0 place-items-center text-muted-foreground">
            <ExternalLink className="size-3.5" />
          </span>
        ) : null}
        <button
          type="button"
          disabled={active}
          onClick={() => setDismissedCheckpoint(dismissalKey)}
          aria-label={t("common.close")}
          title={t("common.close")}
          className="pointer-events-auto grid size-7 shrink-0 place-items-center rounded-md text-muted-foreground transition-colors hover:bg-muted hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring disabled:cursor-not-allowed disabled:opacity-35 disabled:hover:bg-transparent disabled:hover:text-muted-foreground"
        >
          <X className="size-3.5" />
        </button>
      </div>

      {upload.state === "error" ? (
        <p role="alert" className="mt-2 border-t border-destructive/15 pt-2 text-xs text-destructive">
          {upload.error || t("studio.checkpointUpload.transferError")}
        </p>
      ) : null}
    </aside>
  );
}
