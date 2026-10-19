// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { DatasetFormatError } from "../api/datasets-api";

const JSON_FILE = /\.jsonl?$/i;

export function missingDecisionColumns(columns: readonly string[]): string[] {
  const present = new Set(columns);
  const missing = ["state", "questions"].filter(
    (column) => !present.has(column),
  );
  if (!(present.has("gold") || present.has("answers"))) {
    missing.push("gold");
  }
  return missing;
}

// The decision trainer parses uploaded JSON itself, so a failed format check must not block it.
export async function checkDecisionDatasetColumns(
  check: () => Promise<{ columns: readonly string[] } | null>,
  uploadedFile: string | null,
): Promise<string[] | null> {
  let result: { columns: readonly string[] } | null;
  try {
    result = await check();
  } catch (error) {
    if (
      error instanceof DatasetFormatError &&
      uploadedFile !== null &&
      JSON_FILE.test(uploadedFile)
    ) {
      return [];
    }
    throw error;
  }
  return result ? missingDecisionColumns(result.columns) : null;
}
