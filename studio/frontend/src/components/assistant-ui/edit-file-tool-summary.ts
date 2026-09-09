// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { toolArgText } from "./tool-arg-text.ts";

export interface EditFileSummary {
  path: string;
  editCount: number;
  mode: "create" | "replace";
  replaceAllCount: number;
}

function objectRecord(value: unknown): Record<string, unknown> | null {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}

function replaceAllIsTrue(value: unknown): boolean {
  if (value === true) {
    return true;
  }
  if (typeof value === "string") {
    return ["true", "1", "yes"].includes(value.trim().toLowerCase());
  }
  return typeof value === "number" && Number.isInteger(value) && value !== 0;
}

/** A bounded summary of the model-provided edit request for the approval card. */
export function summarizeEditFileArgs(args: unknown): EditFileSummary {
  const record = objectRecord(args) ?? {};
  const path = toolArgText(record.path).trim();
  const edits = Array.isArray(record.edits) ? record.edits : [];
  const firstEdit = edits.length === 1 ? objectRecord(edits[0]) : null;
  const mode =
    firstEdit !== null && firstEdit.old_string === "" ? "create" : "replace";
  const replaceAllCount = edits.reduce((count, edit) => {
    const entry = objectRecord(edit);
    return count + (replaceAllIsTrue(entry?.replace_all) ? 1 : 0);
  }, 0);

  return {
    path,
    editCount: edits.length,
    mode,
    replaceAllCount,
  };
}

export function editFileChangeLabel(summary: EditFileSummary): string {
  if (summary.mode === "create") {
    return "Create file";
  }
  if (summary.editCount === 0) {
    return "Replacement request";
  }
  return `${summary.editCount} replacement${summary.editCount === 1 ? "" : "s"}`;
}

export function editFileResultIsError(resultText: string): boolean {
  return resultText.trimStart().startsWith("Error:");
}
