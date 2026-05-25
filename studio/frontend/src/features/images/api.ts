// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Thin client for the diffusion image-generation routes exposed by
// studio/backend/routes/inference.py (images/load, images/generate,
// images/status, images/unload). Mirrors the shape returned by
// DiffusionBackend.status() and DiffusionGenerateResponse so the
// page can render results without re-deriving fields client-side.

import { authFetch } from "@/features/auth";
import { readFastApiError } from "@/lib/format-fastapi-error";

export interface DiffusionFamily {
  name: string;
  pipeline_class: string;
  base_repo: string;
}

export interface DiffusionStatus {
  is_loaded: boolean;
  is_loading: boolean;
  repo_id: string | null;
  family: string | null;
  pipeline_class: string | null;
  base_repo: string | null;
  gguf_filename: string | null;
  device: string | null;
  dtype: string | null;
  loaded_at: number | null;
  last_error: string | null;
  supported_families: DiffusionFamily[];
}

export interface DiffusionLoadRequest {
  repo_id: string;
  gguf_filename?: string;
  base_repo?: string;
  family?: string;
  hf_token?: string;
  enable_model_cpu_offload?: boolean;
}

export interface DiffusionGenerateRequest {
  prompt: string;
  negative_prompt?: string;
  num_inference_steps?: number;
  guidance_scale?: number;
  width?: number;
  height?: number;
  // bigint when the seed exceeds Number.MAX_SAFE_INTEGER, otherwise
  // number. The wire format is always a JSON integer; see
  // ``stringifyWithBigInt`` below.
  seed?: number | bigint;
}

export interface DiffusionGenerateResponse {
  image_b64: string;
  image_mime: string;
  width: number;
  height: number;
  num_inference_steps: number;
  guidance_scale: number;
  seed: number | null;
  duration_ms: number;
  model: string | null;
  family: string | null;
}

async function parseJson<T>(res: Response): Promise<T> {
  if (!res.ok) throw new Error(await readFastApiError(res));
  return (await res.json()) as T;
}

export async function fetchDiffusionStatus(): Promise<DiffusionStatus> {
  return parseJson<DiffusionStatus>(
    await authFetch("/api/inference/images/status"),
  );
}

export async function loadDiffusionModel(
  payload: DiffusionLoadRequest,
): Promise<DiffusionStatus> {
  return parseJson<DiffusionStatus>(
    await authFetch("/api/inference/images/load", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),
  );
}

export async function unloadDiffusionModel(): Promise<{ is_loaded: boolean }> {
  return parseJson<{ is_loaded: boolean }>(
    await authFetch("/api/inference/images/unload", { method: "POST" }),
  );
}

/** JSON.stringify cannot serialise BigInt directly. We only ever
 * have BigInts in the seed field, which is an integer; emit the
 * literal digits so the server receives a JSON integer rather than
 * a string. Pydantic v2 accepts arbitrarily large ints. */
function stringifyWithBigInt(value: unknown): string {
  return JSON.stringify(value, (_, v) =>
    typeof v === "bigint" ? `__bigint__:${v.toString()}` : v,
  ).replace(/"__bigint__:(-?\d+)"/g, "$1");
}

export async function generateDiffusionImage(
  payload: DiffusionGenerateRequest,
): Promise<DiffusionGenerateResponse> {
  return parseJson<DiffusionGenerateResponse>(
    await authFetch("/api/inference/images/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: stringifyWithBigInt(payload),
    }),
  );
}
