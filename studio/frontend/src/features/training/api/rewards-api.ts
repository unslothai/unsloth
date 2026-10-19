// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import {
  checkRewardImportNames,
  type RewardImportFile,
} from "../lib/reward-import";

export interface RewardRecord {
  name: string;
  kind: "rule" | "python";
  description: string;
  source: "user" | "bundled";
  valid: boolean;
  shadowed: boolean;
  error: string | null;
  rule: Record<string, unknown> | null;
}

export interface RewardExplanation {
  reason:
    | "matched"
    | "not_matched"
    | "missing_response"
    | "missing_reference"
    | "invalid_response_number"
    | "invalid_reference_number"
    | "within_tolerance"
    | "outside_tolerance"
    | "invalid_json"
    | "wrong_json_type"
    | "missing_json_keys"
    | "within_limit"
    | "over_limit"
    | "regex_timeout";
  response?: string | null;
  reference?: string;
  normalized_response?: string;
  normalized_reference?: string;
  normalizers?: ("strip" | "lower" | "remove_commas" | "collapse_spaces")[];
  comparison_column?: string;
  reference_fallback?: boolean;
  pattern?: string;
  mode?: "fullmatch" | "search";
  json_type?: "object" | "array" | "string" | "number";
  required_keys?: string[];
  missing_keys?: string[];
  characters?: number;
  limit?: number;
  error?: string;
  error_kind?: "relative" | "absolute";
  tolerance?: number;
}

export interface RewardPreviewResponse {
  scores: {
    name: string;
    score: number;
    weighted: number;
    explanation?: RewardExplanation;
  }[];
  total: number;
}

export type RewardPreviewItem =
  | { name: string; markdown?: never; weight: number }
  | { markdown: string; name?: never; weight: number };

async function readJson<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = `${res.status}`;
    try {
      const body = await res.json();
      if (typeof body?.detail === "string") {
        detail = body.detail;
      }
    } catch {
      // Retain the HTTP status when an error response has no JSON detail.
    }
    throw new Error(detail);
  }
  return res.json() as Promise<T>;
}

export async function listRewards(): Promise<RewardRecord[]> {
  return readJson(await authFetch("/api/rewards"));
}

export async function importReward(
  markdown: string,
  overwrite = false,
): Promise<RewardRecord> {
  return readJson(
    await authFetch("/api/rewards", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ markdown, overwrite }),
    }),
  );
}

async function atPath<T>(path: string, run: () => Promise<T>): Promise<T> {
  try {
    return await run();
  } catch (error) {
    throw new Error(
      `${path}: ${error instanceof Error ? error.message : String(error)}`,
    );
  }
}

/** Validates every file before writing; on failure returns what was saved and what is left. */
export async function importRewardFiles(
  files: RewardImportFile[],
  overwrite = false,
) {
  const imported: RewardRecord[] = [];
  try {
    const names: string[] = [];
    for (const { path, markdown } of files) {
      const preview = await atPath(path, () =>
        previewRewards([{ markdown, weight: 1 }], "", null),
      );
      names.push(preview.scores[0].name);
    }
    checkRewardImportNames(
      names,
      (await listRewards()).map((r) => r.name),
      overwrite,
    );
    for (const { path, markdown } of files)
      imported.push(
        await atPath(path, () => importReward(markdown, overwrite)),
      );
    return { imported, remaining: [], error: null };
  } catch (error) {
    return { imported, remaining: files.slice(imported.length), error };
  }
}

export async function exportReward(name: string): Promise<string> {
  const body = await readJson<{ markdown: string }>(
    await authFetch(`/api/rewards/${encodeURIComponent(name)}/export`),
  );
  return body.markdown;
}

export async function deleteReward(name: string): Promise<void> {
  const res = await authFetch(`/api/rewards/${encodeURIComponent(name)}`, {
    method: "DELETE",
  });
  if (!res.ok && res.status !== 204) {
    await readJson(res);
  }
}

export async function previewRewards(
  rewards: RewardPreviewItem[],
  completion: string,
  reference: string | null,
  signal?: AbortSignal,
  explain = false,
): Promise<RewardPreviewResponse> {
  return readJson(
    await authFetch("/api/rewards/preview", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        rewards,
        completion,
        reference,
        ...(explain && { explain: true }),
      }),
      signal,
    }),
  );
}
