// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Mirrors SOURCE_QUANTS in backend core/export/q4nx.py; any other quant is dequantized and rounded again.
const DIRECT_QUANTS = ["q4_0", "q4_1", "q4_k_m"];

// GGUF general.architecture values the pinned ROCm/FLM_Q4NX_Converter maps (q4nx_converter_pins.json),
// with the caveat when only some sizes of that architecture convert.
const Q4NX_ARCHITECTURES: Record<string, string | null> = {
  llama: null,
  qwen2: null,
  qwen2vl: null,
  qwen3: null,
  qwen3vl: null,
  qwen35: "Only Qwen3.5 0.8B, 2B, 4B, 9B and Omni convert.",
  qwen35moe: "Only Qwen3.5 35B-A3B converts.",
  gemma3: null,
  gemma4: null,
  lfm2: null,
  phi3: "Only Phi-4 mini converts.",
  "gpt-oss": null,
  nanbeige: null,
};

export interface Q4nxVariant {
  filename: string;
  quant: string;
}

export function packsDirectly(quant: string): boolean {
  return DIRECT_QUANTS.includes(quant.trim().toLowerCase());
}

/** Directly packed quants first (backend order), then the rest; the first one is the default pick. */
export function sortQ4nxVariants<T extends Q4nxVariant>(variants: T[]): T[] {
  const rank = (v: T) => {
    const i = DIRECT_QUANTS.indexOf(v.quant.trim().toLowerCase());
    return i === -1 ? DIRECT_QUANTS.length : i;
  };
  return variants
    .map((variant, index) => ({ variant, index }))
    .sort((a, b) => rank(a.variant) - rank(b.variant) || a.index - b.index)
    .map(({ variant }) => variant);
}

/** The card's base_model, else the repo id without its -GGUF suffix. */
export function q4nxBaseModel(
  repoId: string | null | undefined,
  cardBaseModel: string | null | undefined,
): string {
  const card = cardBaseModel?.trim();
  if (card) return card;
  return (repoId ?? "").trim().replace(/-GGUF$/i, "");
}

export type Q4nxArchFit =
  | { status: "supported"; note: string | null }
  | { status: "unsupported" | "unknown"; note: string };

export function q4nxArchFit(
  architecture: string | null | undefined,
): Q4nxArchFit {
  const arch = architecture?.trim().toLowerCase();
  if (!arch) {
    return {
      status: "unknown",
      note: "The architecture is read from the file during conversion, so this model may not be supported.",
    };
  }
  if (!(arch in Q4NX_ARCHITECTURES)) {
    return {
      status: "unsupported",
      note: `The Q4NX converter does not support the ${arch} architecture.`,
    };
  }
  return { status: "supported", note: Q4NX_ARCHITECTURES[arch] };
}

/** A local row's file: the row itself when it is a .gguf, else the variant inside its folder. */
export function localGgufPath(
  rowPath: string,
  variant: { filename: string } | null,
): string | null {
  if (/\.gguf$/i.test(rowPath.trim())) return rowPath.trim();
  if (!variant) return null;
  const sep = rowPath.includes("\\") && !rowPath.includes("/") ? "\\" : "/";
  return `${rowPath.replace(/[\\/]+$/, "")}${sep}${variant.filename.replace(/^[\\/]+/, "")}`;
}
