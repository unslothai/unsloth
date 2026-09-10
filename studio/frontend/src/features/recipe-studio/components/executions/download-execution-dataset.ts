// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { downloadFile, downloadUrlStreaming } from "@/lib/native-files";
import { downloadRecipeJobDataset } from "../../api";
import type { RecipeExecutionRecord } from "../../execution-types";

function sanitizeFilenameStem(value: string): string {
  const cleaned = value
    .trim()
    .replace(/[^\w.-]+/g, "-")
    .replace(/^-+|-+$/g, "");
  return cleaned || "recipe-dataset";
}

function buildDownloadFilename(execution: RecipeExecutionRecord): string {
  const runName = execution.run_name?.trim();
  if (runName) {
    return sanitizeFilenameStem(runName);
  }
  return sanitizeFilenameStem(execution.id);
}

function triggerClientJsonlDownload(
  rows: Record<string, unknown>[],
  filenameStem: string,
): Promise<void> {
  const body =
    rows.map((row) => JSON.stringify(row)).join("\n") + (rows.length > 0 ? "\n" : "");
  return downloadFile(body, `${filenameStem}.jsonl`, "application/x-ndjson");
}

export async function downloadExecutionDataset(
  execution: RecipeExecutionRecord,
): Promise<void> {
  const filenameStem = buildDownloadFilename(execution);

  if (execution.kind === "full" && execution.jobId) {
    const { url, filename } = await downloadRecipeJobDataset(execution.jobId, {
      artifactPath: execution.artifact_path,
      filename: filenameStem,
    });
    await downloadUrlStreaming(url, filename);
    return;
  }

  if (execution.dataset.length === 0) {
    throw new Error("This run does not have a dataset to download yet.");
  }

  await triggerClientJsonlDownload(execution.dataset, filenameStem);
}
