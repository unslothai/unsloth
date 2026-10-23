// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

const MLX_DRAFTER_LABELS: Record<string, string> = {
  mtp: "MTP",
  dflash: "DFlash",
  dspark: "DSpark",
  eagle3: "EAGLE-3",
};

/** Copy for an MLX load's spec_fallback_reason, or null for a code MLX does not report. Several
 *  backend paths share each code, so the copy states only what every one of them implies, and the
 *  outcome follows what runs: the reason can name a drafter passed over while another attached. */
export function mlxSpecFallbackMessage(
  reason: string,
  drafterKind: string | null,
): string | null {
  const outcome =
    drafterKind == null
      ? "so it is running without speculative decoding"
      : drafterKind === "ngram"
        ? "so it is drafting with n-gram copies only"
        : `so it is drafting with ${MLX_DRAFTER_LABELS[drafterKind] ?? drafterKind} instead`;
  switch (reason) {
    case "drafter_not_found":
      return `A drafter this load asked for was not found locally, or is not a drafter, ${outcome}. Download it, or check the drafter named in this model's settings, then reload.`;
    case "drafter_incompatible":
      return `A drafter was found but could not be used with this model or mode, ${outcome}. Try a drafter made for this model and mode, then reload.`;
    case "drafter_no_memory":
      return `The drafter could not be confirmed to fit in memory beside this model, ${outcome}. When memory is the limit, a shorter context length or a smaller drafter can help.`;
    case "runtime_error":
      return `Speculative decoding could not start for this load, ${outcome}.`;
    case "kv_quant":
      return "Speculative decoding cannot run with KV cache quantization, which this load asked for. Setting KV Cache Dtype to Auto in this model's settings removes that obstacle.";
    case "auto_context_cost":
      return `Auto left out a drafter that does not fit beside this model at its full context length, ${outcome}. To try it anyway, choose its mode in this model's settings, with a shorter or automatic context length.`;
    default:
      return null;
  }
}
