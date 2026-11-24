// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Load-time LoRA helpers for the Video page. A video adapter is merged into the denoiser at load, so the
 * picker holds a pending selection and the resident build reports what it baked (`baked_lora_specs`). */

import type { LoraSpecInput } from "@/features/images/api";

import type { VideoStatus } from "./api";

/** The video load accepts at most this many adapters (VideoLoadRequest.loras max_length). */
export const MAX_VIDEO_LORAS = 8;

/** The specs a load bakes: trimmed ids, weight != 0, first of a repeated id; undefined = a plain load. */
export function videoLoraSpecs(
  selection: readonly LoraSpecInput[] | null | undefined,
): LoraSpecInput[] | undefined {
  const out: LoraSpecInput[] = [];
  const seen = new Set<string>();
  for (const entry of selection ?? []) {
    const id = typeof entry?.id === "string" ? entry.id.trim() : "";
    const weight = Number(entry?.weight);
    if (!id || !Number.isFinite(weight) || weight === 0 || seen.has(id)) continue;
    seen.add(id);
    out.push({ id, weight });
  }
  return out.length > 0 ? out.slice(0, MAX_VIDEO_LORAS) : undefined;
}

/** What the resident build baked, as the load requested it; [] when nothing is loaded or nothing was baked. */
export function bakedVideoLoraSpecs(status: VideoStatus | null | undefined): LoraSpecInput[] {
  if (!status?.loaded) return [];
  return videoLoraSpecs(status.baked_lora_specs ?? []) ?? [];
}

function specKey(specs: readonly LoraSpecInput[] | undefined): string {
  return (specs ?? [])
    .map((s) => `${s.id}\u0000${s.weight}`)
    .sort()
    .join("\u0001");
}

/** True when the picker differs from the resident build; order-free, as the merge sums deltas. */
export function videoLoraReloadNeeded(
  selection: readonly LoraSpecInput[],
  status: VideoStatus | null | undefined,
): boolean {
  if (!status?.loaded) return false;
  return specKey(videoLoraSpecs(selection)) !== specKey(bakedVideoLoraSpecs(status));
}

/** Only a full diffusers pipeline load has a dense denoiser to bake into; GGUF and single-file picks and
 * two-expert families refuse (`lora_refusal`). */
export function videoLoraCapable(status: VideoStatus | null | undefined): boolean {
  return Boolean(status?.loaded && status.model_kind === "pipeline" && !status.lora_refusal);
}

/** The catalog id of a trained adapter: its file stem, as the LoRA scan keys a local file. */
export function trainedLoraId(catalogPath: string): string {
  const base = catalogPath.replace(/\\/g, "/").split("/").pop() ?? "";
  return base.replace(/\.(safetensors|gguf)$/i, "");
}
