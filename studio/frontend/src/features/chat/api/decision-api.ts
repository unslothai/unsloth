// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { formatFastApiDetail } from "@/lib/format-fastapi-error";
// eslint-disable-next-line no-restricted-imports -- Leaf module: types and request builders, no React.
import type {
  DecisionResponse,
  WireQuestion,
} from "@/features/settings/lib/decision-request";

export type LoadedDecisionBody = {
  state: string;
  questions: Record<string, WireQuestion>;
  /** Base64 data URLs, for a vision decision model. */
  images?: string[];
};

/** llama-server's /v1/systemone body, plus the fields the Studio proxy adds. */
export type LoadedDecisionResponse = DecisionResponse & {
  latency_ms?: number;
  model?: string | null;
};

export class LoadedDecisionError extends Error {
  readonly status: number | null;

  constructor(message: string, status: number | null) {
    super(message);
    this.name = "LoadedDecisionError";
    this.status = status;
  }
}

/**
 * Answers a decision request with the loaded decision-only GGUF.
 *
 * llama-server serves these models on /v1/systemone only, so the Chat Decision
 * console posts here instead of /v1/chat/completions.
 */
export async function runLoadedDecision(
  body: LoadedDecisionBody,
  signal?: AbortSignal,
): Promise<LoadedDecisionResponse> {
  const res = await authFetch("/api/inference/decision", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  if (res.ok) {
    return (await res.json()) as LoadedDecisionResponse;
  }
  const data = (await res.json().catch(() => null)) as {
    detail?: unknown;
  } | null;
  const detail = data?.detail;
  const fields =
    detail && typeof detail === "object"
      ? (detail as Record<string, unknown>)
      : {};
  const message =
    (typeof fields.message === "string" ? fields.message : null) ??
    formatFastApiDetail(detail) ??
    res.statusText;
  throw new LoadedDecisionError(message, res.status);
}
