// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { readFastApiError } from "@/lib/format-fastapi-error";
import {
  normalizeParallelCapabilities,
  normalizeParallelPlan,
} from "../lib/parallel-plan";
import type { ParallelCapabilities, ParallelPlan } from "../types/parallel";

export const PARALLEL_CAPABILITIES_PATH = "/api/train/parallel/capabilities";
export const PARALLEL_PLAN_PATH = "/api/train/parallel/plan";

export type ParallelResult<T> =
  | { kind: "ok"; value: T }
  /** A backend without the multi-GPU routes: the UI stays as before and sends nothing new. */
  | { kind: "unsupported" }
  | { kind: "error"; message: string };

async function readResult<T>(
  response: Response,
  normalize: (raw: unknown) => T | null,
): Promise<ParallelResult<T>> {
  if (response.status === 404 || response.status === 405) {
    return { kind: "unsupported" };
  }
  if (!response.ok) {
    return { kind: "error", message: await readFastApiError(response) };
  }
  const value = normalize(await response.json());
  return value === null
    ? { kind: "error", message: `Unreadable response (${response.status})` }
    : { kind: "ok", value };
}

export async function fetchParallelCapabilities(
  signal?: AbortSignal,
): Promise<ParallelResult<ParallelCapabilities>> {
  const response = await authFetch(PARALLEL_CAPABILITIES_PATH, { signal });
  return readResult(response, normalizeParallelCapabilities);
}

export async function fetchParallelPlan(
  request: Record<string, unknown>,
  signal?: AbortSignal,
): Promise<ParallelResult<ParallelPlan>> {
  const response = await authFetch(PARALLEL_PLAN_PATH, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
    signal,
  });
  return readResult(response, normalizeParallelPlan);
}
