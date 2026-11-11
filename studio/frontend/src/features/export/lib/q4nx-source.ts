// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Mirrors SOURCE_QUANTS in backend core/export/q4nx.py; any other quant is dequantized and rounded again.
const DIRECT_QUANTS = ["q4_0", "q4_1", "q4_k_m"];

// GGUF general.architecture values the pinned ROCm/FLM_Q4NX_Converter accepts (q4nx_converter_pins.json),
// with the caveat when it only takes some models of that architecture. Some (qwen35, qwen2vl) it
// matches by general.basename prefix rather than by architecture.
const Q4NX_ARCHITECTURES: Record<string, string | null> = {
  llama: null,
  qwen2: null,
  qwen2vl: "Only Qwen2.5-VL converts.",
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

// modelInfo fields for a typed repo. "gguf" is outside the SDK's expandable-key type but is forwarded to expand= as is.
export const Q4NX_INFO_FIELDS = ["tags", "gguf"] as unknown as "tags"[];

export interface Q4nxVariant {
  filename: string;
  quant: string;
}

/** The quant's last path segment: a listed HF cache folder keys its variants by path (`snapshots/<sha>/m-Q4_1`). */
export function q4nxQuantLabel(quant: string): string {
  return quant.trim().split(/[\\/]/).pop() ?? "";
}

// Matches the bare token or a stem ending in it ("Qwen3-0.6B-Q4_1"), not "IQ4_0" or "UD-Q4_K_XL".
function directRank(quant: string): number {
  const label = q4nxQuantLabel(quant).toLowerCase();
  const i = DIRECT_QUANTS.findIndex(
    (q) => label === q || ["-", "_", "."].some((sep) => label.endsWith(sep + q)),
  );
  return i === -1 ? DIRECT_QUANTS.length : i;
}

export function packsDirectly(quant: string): boolean {
  return directRank(quant) < DIRECT_QUANTS.length;
}

/** Directly packed quants first (backend order), then the rest; the first one is the default pick. */
export function sortQ4nxVariants<T extends Q4nxVariant>(variants: T[]): T[] {
  const rank = (v: T) => directRank(v.quant);
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

// The converter finds no qwen2vl architecture name, so a Qwen2-VL basename falls to the dense qwen2 prefix.
const QWEN2_VL_NAME = /(^|[\s/\\._-])qwen2[-_]?vl/i;

/** `name` is the repo id or file path, used to refuse models the converter would misroute, even when the architecture is unknown. */
export function q4nxArchFit(
  architecture: string | null | undefined,
  name?: string | null,
): Q4nxArchFit {
  const arch = architecture?.trim().toLowerCase();
  if ((!arch || arch === "qwen2vl") && QWEN2_VL_NAME.test(name ?? "")) {
    return {
      status: "unsupported",
      note: "The Q4NX converter does not support Qwen2-VL. Only Qwen2.5-VL converts.",
    };
  }
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

/** Which listing the file picker shows. Only the active source counts: a picked local folder survives a switch to the Hub. */
export function q4nxVariantLookup(
  source: "hub" | "local",
  repoId: string,
  localFolder: string | null,
): { key: string | null; localDirPath: string | null } {
  if (source === "hub") {
    return { key: repoId ? `hub::${repoId}` : null, localDirPath: null };
  }
  return {
    key: localFolder ? `local::${localFolder}` : null,
    localDirPath: localFolder,
  };
}

/** Whether the field still shows the picked model; typing a new query leaves it picked but uncommitted. */
export function q4nxPickCommitted(selected: string, inputText: string): boolean {
  return !!selected && inputText.trim() === selected;
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
