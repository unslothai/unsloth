// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { SectionCard } from "@/components/section-card";
import { Progress } from "@/components/ui/progress";
import { useT } from "@/i18n";
import type { CheckpointUploadProgress } from "@/features/training";
import type { ReactElement } from "react";

function formatBytes(value: number): string {
  if (value < 1024) return `${value} B`;
  if (value < 1024 ** 2) return `${(value / 1024).toFixed(1)} KiB`;
  if (value < 1024 ** 3) return `${(value / 1024 ** 2).toFixed(1)} MiB`;
  return `${(value / 1024 ** 3).toFixed(1)} GiB`;
}

export function CheckpointUploadCard({ upload }: { upload: CheckpointUploadProgress }): ReactElement {
  const t = useT();
  const percent = typeof upload.percentage === "number"
    ? Math.min(100, Math.max(0, upload.percentage)) : null;
  const repositoryUrl = upload.repository_url?.startsWith("https://huggingface.co/")
    ? upload.repository_url : null;
  const active = upload.state === "preparing" || upload.state === "uploading";
  const title = upload.state === "completed" ? t("studio.checkpointUpload.completed")
    : upload.state === "skipped" ? t("studio.checkpointUpload.skipped")
    : upload.state === "preparing" ? t("studio.checkpointUpload.preparing")
    : upload.state === "uploading" ? t("studio.checkpointUpload.uploading")
    : t("studio.checkpointUpload.title");

  return <SectionCard icon={<span aria-hidden>↑</span>} title={title} description={upload.message || title}>
    <div className="space-y-3 text-sm">
      {upload.checkpoint ? <div><span className="text-muted-foreground">{t("studio.checkpointUpload.checkpoint")}: </span>{upload.checkpoint}</div> : null}
      {upload.repository_id ? <div><span className="text-muted-foreground">{t("studio.checkpointUpload.destination")}: </span>{upload.repository_id}</div> : null}
      {active ? <Progress value={percent ?? undefined} className="h-2" /> : null}
      {percent !== null ? <div>{percent.toFixed(0)}%</div> : active ? <div>{t("studio.checkpointUpload.unknownTotal")}</div> : null}
      {upload.uploaded_bytes != null ? <div>{formatBytes(upload.uploaded_bytes)}{upload.total_bytes != null ? ` / ${formatBytes(upload.total_bytes)}` : ""}</div>
        : upload.uploaded_files != null ? <div>{upload.uploaded_files}{upload.total_files != null ? ` / ${upload.total_files} files` : " files"}</div> : null}
      {upload.message ? <p className="text-muted-foreground">{upload.message}</p> : null}
      {upload.state === "error" ? <div role="alert" className="rounded-md bg-destructive/10 p-3 text-destructive">{upload.error || t("studio.checkpointUpload.transferError")}</div> : null}
      {repositoryUrl && upload.state === "completed" ? <a href={repositoryUrl} target="_blank" rel="noopener noreferrer" className="text-primary underline">{repositoryUrl}</a> : null}
    </div>
  </SectionCard>;
}
