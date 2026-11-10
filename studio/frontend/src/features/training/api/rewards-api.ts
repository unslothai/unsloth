// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";

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

/** The values the backend evaluator used for one score (score_rule in rewards.py). */
export interface RewardExplanation {
  reason: string;
  response?: string | null;
  reference?: string;
  comparison_column?: string;
  reference_fallback?: boolean;
  normalizers?: string[];
  normalized_response?: string;
  normalized_reference?: string;
  pattern?: string;
  mode?: "fullmatch" | "search";
  json_type?: string;
  required_keys?: string[];
  missing_keys?: string[];
  characters?: number;
  limit?: number;
  error?: string;
  error_kind?: "relative" | "absolute";
  tolerance?: number;
}

export interface RewardPreviewScore {
  name: string;
  score: number;
  weighted: number;
  explanation?: RewardExplanation;
}

export interface RewardPreviewResponse {
  scores: RewardPreviewScore[];
  total: number;
}

/** A library reward by name, or an unsaved draft as REWARD.md text. */
export type RewardPreviewItem =
  | { name: string; weight: number }
  | { markdown: string; weight: number };

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

/** Imports every REWARD.md in the ZIP or none; a name clash fails unless overwrite is set. */
export async function importRewardZip(
  file: Blob,
  overwrite = false,
): Promise<RewardRecord[]> {
  return readJson(
    await authFetch(`/api/rewards/import-zip?overwrite=${overwrite}`, {
      method: "POST",
      headers: { "Content-Type": "application/zip" },
      body: file,
    }),
  );
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
  explain = false,
): Promise<RewardPreviewResponse> {
  return readJson(
    await authFetch("/api/rewards/preview", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ rewards, completion, reference, explain }),
    }),
  );
}
