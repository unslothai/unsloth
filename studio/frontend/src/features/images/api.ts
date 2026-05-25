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
  /**
   * Numeric seed. Safe ONLY for values <= Number.MAX_SAFE_INTEGER.
   * For larger seeds, prefer ``seed_str`` (full-precision decimal).
   */
  seed: number | null;
  /** Decimal string with full uint64 precision. Use this for display
   *  and reproduction when the user pastes the seed back in. */
  seed_str: string | null;
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

/** JSON.stringify cannot serialise BigInt directly. Pull the seed
 * BigInt out, stringify the rest of the payload normally, then
 * splice the seed's decimal digits back into the JSON literal at the
 * exact ``"seed":<int>`` slot.
 *
 * Avoids the previous regex-over-JSON approach, which could be
 * tripped by a user-supplied prompt that exactly matched the
 * sentinel string. With this approach the only thing we touch is
 * the literal ``"seed":<number>`` substring we wrote ourselves.
 */
function stringifyWithBigInt(value: DiffusionGenerateRequest): string {
  const { seed, ...rest } = value;
  if (typeof seed !== "bigint") {
    return JSON.stringify(value);
  }
  // Serialise the rest without seed, then inject the seed at the end
  // of the object literal as a JSON integer. Strip the trailing "}"
  // and re-append once the field is added.
  const base = JSON.stringify(rest);
  const inner = base.length === 2 /* '{}' */ ? "" : base.slice(1, -1) + ",";
  return `{${inner}"seed":${seed.toString()}}`;
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
