// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// torchao INT8 / FP8 / NVFP4 / MXFP8 loads (and MiniMax-H3-Image) take LoRAs only at load: unbaked ones need a reload.

/** Backend: diffusion_lora.LORA_RELOAD_REFUSAL, sent in X-Unsloth-Refusal on /images/generate. */
export const LORA_RELOAD_REFUSAL = "lora-reload";
export const LORA_RELOAD_TITLE = "This LoRA selection needs a reload";
export const LORA_RELOAD_ACTION = "Reload with LoRAs";

// Backend: diffusion._apply_loras, the schemes that route to _adjust_baked_loras.
const BAKING_SCHEMES = new Set(["int8", "fp8", "nvfp4", "mxfp8"]);
const H3_IMAGE_FAMILY = "minimax-h3-image";

export class LoraReloadRefusalError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "LoraReloadRefusalError";
  }
}

export function isLoraReloadRefusal(
  status: number,
  refusalHeader: string | null | undefined,
): boolean {
  return status === 400 && (refusalHeader ?? "").trim().toLowerCase() === LORA_RELOAD_REFUSAL;
}

export interface LoraBakeStatus {
  loaded?: boolean;
  family?: string | null;
  transformer_quant?: string | null;
  supports_lora?: boolean;
  baked_lora_specs?: Array<{ id: string; weight: number }> | null;
}

export interface LoraPick {
  id: string;
  weight: number;
}

function bakeScheme(status: LoraBakeStatus): string | null {
  const quant = (status.transformer_quant ?? "").trim().toLowerCase();
  if (BAKING_SCHEMES.has(quant)) return quant.toUpperCase();
  return null;
}

export function bakesLorasAtLoad(status: LoraBakeStatus | null | undefined): boolean {
  if (!status?.loaded) return false;
  return bakeScheme(status) !== null || (status.family ?? "").trim().toLowerCase() === H3_IMAGE_FAMILY;
}

function activePicks(loras: readonly LoraPick[]): LoraPick[] {
  return loras
    .map((l) => ({ id: l.id.trim(), weight: l.weight }))
    .filter((l) => l.id && l.weight !== 0);
}

/** Why Generate must reload first, or null for no adapters or a subset of the baked set (backend re-scales). */
export function loraReloadReason(
  status: LoraBakeStatus | null | undefined,
  loras: readonly LoraPick[],
): string | null {
  if (!status?.loaded || status.supports_lora === false || !bakesLorasAtLoad(status)) return null;
  // No baked set: let the backend decide. Reading it as "none baked" makes every click reload, never generate.
  if (status.baked_lora_specs == null) return null;
  const picks = activePicks(loras);
  const baked = status.baked_lora_specs.filter((s) => s && s.id && s.weight !== 0);
  const h3 = (status.family ?? "").trim().toLowerCase() === H3_IMAGE_FAMILY;
  if (h3) {
    // Merged into the weights: the selection AND weights are fixed for the load.
    const key = (l: LoraPick) => `${l.id}\u0000${l.weight}`;
    const want = picks.map(key).sort();
    const have = baked.map(key).sort();
    if (want.length === have.length && want.every((k, i) => k === have[i])) return null;
    return (
      "MiniMax-H3-Image merges LoRAs into its weights when it loads, so Generate reloads the model with this " +
      "LoRA selection first."
    );
  }
  if (picks.length === 0) return null;
  const remaining = baked.map((s) => s.id);
  for (const pick of picks) {
    const at = remaining.indexOf(pick.id);
    if (at < 0) {
      const scheme = bakeScheme(status) ?? "quantized";
      return (
        `This ${scheme} build takes LoRAs only when it loads, so Generate reloads the model with them first. ` +
        "That can take a few minutes, longer if the full-precision weights have to download."
      );
    }
    remaining.splice(at, 1);
  }
  return null;
}
