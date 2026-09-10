// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { isTauri } from "@/lib/api-base";
import {
  downloadFile,
  downloadUrlStreaming,
  isDownloadCancelled,
} from "@/lib/native-files";
import { downloadRecipeJobDataset } from "../../api";
import type { RecipeExecutionRecord } from "../../execution-types";

/** Whether the bytes are known to have landed: a browser anchor click resolves before the
 * request is even sent, while the native downloader streams and rejects a non-2xx. */
export type DownloadOutcome = "saved" | "started";

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

/** Whether the rows held on the client are the whole dataset rather than one page of it. */
function hasCompleteLocalDataset(execution: RecipeExecutionRecord): boolean {
  const total = execution.datasetTotal;
  return typeof total !== "number" || execution.dataset.length >= total;
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
): Promise<DownloadOutcome> {
  const filenameStem = buildDownloadFilename(execution);

  // The backend export pages the whole dataset; the rows here are one page of it.
  if (execution.jobId) {
    try {
      // A real authenticated request, so an unexportable run fails before anything is claimed.
      const { url, filename } = await downloadRecipeJobDataset(execution.jobId, {
        artifactPath: execution.artifact_path,
        filename: filenameStem,
      });
      await downloadUrlStreaming(url, filename);
      return isTauri ? "saved" : "started";
    } catch (error) {
      if (isDownloadCancelled(error)) {
        throw error;
      }
      // Only a stale preview lands here, and only worth writing whole. An artifact-backed run
      // must not: its images live beside the parquet, and a bare JSONL loses them.
      if (execution.artifact_path || !hasCompleteLocalDataset(execution)) {
        throw error;
      }
    }
  }

  if (execution.dataset.length === 0) {
    throw new Error("This run does not have a dataset to download yet.");
  }
  if (!hasCompleteLocalDataset(execution)) {
    throw new Error(
      "Only part of this dataset is loaded. Reopen the run and try again.",
    );
  }

  await triggerClientJsonlDownload(execution.dataset, filenameStem);
  return "saved";
}
