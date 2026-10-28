// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import type {
  GrpoPythonRewardIsolation,
  GrpoRewardSelection,
} from "@/types/training";

export interface RewardRecord {
  name: string;
  kind: "rule" | "python";
  description: string;
  source: "user" | "bundled";
  valid: boolean;
  shadowed: boolean;
  error: string | null;
  rule: Record<string, unknown> | null;
  entry?: string | null;
  code?: string | null;
}

export interface RewardPreviewResponse {
  scores: {
    name: string;
    score: number | null;
    weighted: number | null;
    /** Why this reward did not score; the others still did. */
    error?: string | null;
  }[];
  total: number;
  isolation: { backend: string; os_isolation: boolean } | null;
}

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
  rewards: GrpoRewardSelection[],
  completion: string,
  reference: string | null,
  row: Record<string, unknown> | null = null,
  prompt: string | Record<string, unknown>[] | null = null,
  systemPrompt: string | null = null,
  pythonRewardIsolation: GrpoPythonRewardIsolation = "required",
): Promise<RewardPreviewResponse> {
  return readJson(
    await authFetch("/api/rewards/preview", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        rewards,
        completion,
        reference,
        row,
        prompt,
        system_prompt: systemPrompt,
        python_reward_isolation: pythonRewardIsolation,
      }),
    }),
  );
}

/** What Start would be refused with for Python rewards on this host, or null. */
export async function pythonRewardIsolationRefusal(
  mode: GrpoPythonRewardIsolation,
): Promise<string | null> {
  const body = await readJson<{ refusal: string | null }>(
    await authFetch(`/api/rewards/python-isolation?mode=${mode}`),
  );
  return body.refusal;
}
