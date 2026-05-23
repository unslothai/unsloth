// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth/api";

export interface ScoreNode {
  score: number;
  n_leaves: number;
  note?: string;
  matched_option?: number;
  children?: Record<string, ScoreNode> | ScoreNode[];
}

export interface ScoreResult {
  score: number;
  breakdown: ScoreNode | null;
}

export interface ScoreRequest {
  ground_truth: unknown;
  prediction: unknown;
  schema?: unknown;
  default_comparator?: string;
  return_key_scores?: boolean;
}

export async function scoreDocument(payload: ScoreRequest): Promise<ScoreResult> {
  const res = await authFetch("/api/scoring/score", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ return_key_scores: true, ...payload }),
  });
  if (!res.ok) {
    let detail = `Scoring failed (${res.status})`;
    try {
      const data = await res.json();
      if (data?.detail) {
        detail =
          typeof data.detail === "string"
            ? data.detail
            : JSON.stringify(data.detail);
      }
    } catch {
      // non-JSON error body — keep the status-based message
    }
    throw new Error(detail);
  }
  return (await res.json()) as ScoreResult;
}
