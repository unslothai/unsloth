// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { type NpuCompat, type NpuTag, npuFit } from "../../npu/compat.ts";

// Mirrors SOURCE_QUANTS in backend core/export/q4nx.py; any other quant is dequantized and rounded again.
const DIRECT_QUANTS = ["q4_0", "q4_1", "q4_k_m"];
// Only the q4k converter unpacks Q4_K, for q4k_architectures; q4nx rounds Q4_K_M again.
const Q4K_ARCHITECTURES = ["qwen35", "qwen35moe"];
// Q4_0 and Q4_1 blocks are packed as is, so only the layout the converter config reads works: Q4_0
// here, Q4_1 on every other q4nx architecture, Q4_K on the q4k ones. The backend refuses the rest.
const Q4_0_ARCHITECTURES = ["qwen2", "lfm2"];

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

/** "by-architecture": Q4_K_M of a file whose architecture is not known yet. "mismatch": blocks the model reads in another layout. */
export type Q4nxPacking = "direct" | "by-architecture" | "requantized" | "mismatch";

// Only names whose layout differs from the Q4_1 default; a Qwen2.5-VL is qwen2vl, which reads Q4_1.
const NAME_ARCHITECTURES: [RegExp, string][] = [
  [/qwen3\.[56](?![0-9])/i, "qwen35"],
  [/qwen2\.5(?![-_.]?vl)(?![0-9])/i, "qwen2"],
  [/(^|[^a-z0-9])lfm2/i, "lfm2"],
];

// The last path segment: a local GGUF's id is its path, and a parent folder names nothing about it.
function leafName(name?: string | null): string {
  return (name ?? "").trim().split(/[\\/]/).pop() ?? "";
}

/** The architecture, else the one a model's name gives: a local GGUF is not read before converting. */
export function q4nxArchitecture(architecture?: string | null, name?: string | null): string | null {
  const arch = architecture?.trim().toLowerCase();
  if (arch) return arch;
  return NAME_ARCHITECTURES.find(([re]) => re.test(leafName(name)))?.[1] ?? null;
}

/** The quant that packs directly for an architecture, or null when it is not known. */
export function q4nxFitQuant(architecture?: string | null, name?: string | null): string | null {
  const arch = q4nxArchitecture(architecture, name);
  if (!arch) return null;
  // The q4k converter resolves a Qwen3.5 named Omni to qwen3.5_omni.json, which reads Q4_1.
  if (arch === "qwen35" && /omni/i.test(leafName(name))) return "q4_1";
  if (Q4K_ARCHITECTURES.includes(arch)) return "q4_k_m";
  return Q4_0_ARCHITECTURES.includes(arch) ? "q4_0" : "q4_1";
}

export function q4nxPacking(
  quant: string,
  architecture?: string | null,
  name?: string | null,
): Q4nxPacking {
  // Both converters pack Q8_0 as is, and no model reads 8-bit in its 4-bit slots.
  if (q4nxQuantLabel(quant).toLowerCase().endsWith("q8_0")) return "mismatch";
  const rank = directRank(quant);
  if (rank === DIRECT_QUANTS.length) return "requantized";
  const fit = q4nxFitQuant(architecture, name);
  if (DIRECT_QUANTS[rank] !== "q4_k_m") {
    return !fit || fit === DIRECT_QUANTS[rank] ? "direct" : "mismatch";
  }
  if (!fit) return "by-architecture";
  if (fit === "q4_k_m") return "direct";
  // The q4k converter packs Q4_K as is too, so only the q4nx one re-quantizes it.
  return Q4K_ARCHITECTURES.includes(q4nxArchitecture(architecture, name) ?? "")
    ? "mismatch"
    : "requantized";
}

export function q4nxIsFirstChoice(packing: Q4nxPacking): boolean {
  return packing === "direct" || packing === "by-architecture";
}

export function packsDirectly(quant: string, architecture?: string | null): boolean {
  return q4nxPacking(quant, architecture) === "direct";
}

const PACKING_ORDER: Q4nxPacking[] = ["direct", "by-architecture", "requantized", "mismatch"];

// A split GGUF is listed as one variant named by its first shard; the converter reads one file.
const SPLIT_SHARD = /-\d{3,}-of-\d{3,}\.gguf$/i;

export function q4nxIsSplitFile(path: string): boolean {
  return SPLIT_SHARD.test(path.trim());
}

export function q4nxHasSplitFiles(variants: readonly { filename: string }[]): boolean {
  return variants.some((v) => q4nxIsSplitFile(v.filename));
}

/** Drops empty leftover quant folders and split GGUFs. The listing collapses a quant shipped both
 *  whole and split to its shards, so a split row becomes its whole file when `repoFiles` has it. */
export function q4nxConvertibleFiles<T extends { filename: string; cleanable?: boolean }>(
  variants: T[],
  repoFiles?: ReadonlySet<string>,
): T[] {
  return variants.flatMap((v) => {
    if (v.cleanable) return [];
    if (!SPLIT_SHARD.test(v.filename)) return [v];
    const whole = v.filename.replace(SPLIT_SHARD, ".gguf");
    return repoFiles?.has(whole) ? [{ ...v, filename: whole }] : [];
  });
}

export function sortQ4nxVariants<T extends Q4nxVariant>(
  variants: T[],
  architecture?: string | null,
  name?: string | null,
): T[] {
  const rank = (v: T) =>
    PACKING_ORDER.indexOf(q4nxPacking(v.quant, architecture, name)) * 10 + directRank(v.quant);
  return variants
    .map((variant, index) => ({ variant, index }))
    .sort((a, b) => rank(a.variant) - rank(b.variant) || a.index - b.index)
    .map(({ variant }) => variant);
}

export function q4nxBaseModel(
  repoId: string | null | undefined,
  cardBaseModel: string | null | undefined,
): string {
  const card = cardBaseModel?.trim();
  if (card) return card;
  return (repoId ?? "").trim().replace(/-GGUF$/i, "");
}

// "Qwen3.5-9B", "Qwen3-0.6B-GGUF"; in "35B-A3B" the first, total count wins.
const NAME_PARAMS_RE = /(?:^|[^A-Za-z0-9.])(\d+(?:\.\d+)?)B(?![A-Za-z0-9])/i;

export function q4nxParamsFromName(name: string): number | null {
  const billions = Number(NAME_PARAMS_RE.exec(name)?.[1]);
  return billions > 0 ? billions * 1e9 : null;
}

export type Q4nxArchFit =
  | { status: "supported"; note: string | null; tag: NpuTag | null }
  | { status: "unsupported"; note: string; tag: NpuTag }
  | { status: "unknown"; note: string; tag: null };

// The converter finds no qwen2vl architecture name, so a Qwen2-VL basename falls to the dense qwen2 prefix.
const QWEN2_VL_NAME = /(^|[\s/\\._-])qwen2[-_]?vl/i;

/** `name` is the repo id or file path, used to refuse models the converter would misroute, even
 *  when the architecture is unknown. `compat` is GET /api/npu/compat. */
export function q4nxArchFit(
  architecture: string | null | undefined,
  name?: string | null,
  compat?: NpuCompat | null,
): Q4nxArchFit {
  const arch = architecture?.trim().toLowerCase();
  if ((!arch || arch === "qwen2vl") && QWEN2_VL_NAME.test(name ?? "")) {
    return {
      status: "unsupported",
      note: "The Q4NX converter does not support Qwen2-VL. Only Qwen2.5-VL converts.",
      tag: "unsupported",
    };
  }
  if (!arch) {
    return {
      status: "unknown",
      note: "The architecture is read from the file during conversion, so this model may not be supported.",
      tag: null,
    };
  }
  const fit = npuFit(compat, arch, name);
  if (fit.tag === null) {
    return {
      status: "unknown",
      note: "Which models the NPU runs is read once the NPU is enabled; the size is checked during conversion.",
      tag: null,
    };
  }
  if (fit.tag === "convertRuns") {
    return { status: "supported", note: fit.note, tag: fit.tag };
  }
  return { status: "unsupported", note: fit.note ?? "", tag: fit.tag };
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

export interface Q4nxExpansion {
  expandAll: boolean;
  /** Models toggled against the default: closed under expandAll, else the one that is open. */
  toggled: readonly string[];
}

export function q4nxIsExpanded(state: Q4nxExpansion, id: string): boolean {
  return state.toggled.includes(id) !== state.expandAll;
}

export function q4nxToggleExpanded(state: Q4nxExpansion, id: string): Q4nxExpansion {
  const open = q4nxIsExpanded(state, id);
  if (!state.expandAll) return { ...state, toggled: open ? [] : [id] };
  return {
    ...state,
    toggled: open ? [...state.toggled, id] : state.toggled.filter((t) => t !== id),
  };
}

// Context buffers and the runtime on top of the weights; the backend's NPU memory fit (_NPU_LOAD_MARGIN_GB) uses the same margin.
const NPU_LOAD_MARGIN_GB = 1;
// About Q4_K_M's bits per weight, for a repo whose files are not listed yet.
const Q4_BYTES_PER_PARAM = 0.6;
const GIB = 1024 ** 3;

export interface Q4nxWeights {
  bytes: number;
  /** Not the size of a file that packs directly. */
  estimated: boolean;
}

type SizedVariant = Q4nxVariant & { size_bytes?: number | null };

export function q4nxWeights(
  variant: SizedVariant | null,
  siblings: readonly SizedVariant[] = [],
  totalParams?: number | null,
  architecture?: string | null,
): Q4nxWeights | null {
  if (variant?.size_bytes && packsDirectly(variant.quant, architecture)) {
    return { bytes: variant.size_bytes, estimated: false };
  }
  const direct = sortQ4nxVariants(
    siblings.filter((v) => v.size_bytes && packsDirectly(v.quant, architecture)),
    architecture,
  )[0];
  if (direct?.size_bytes) return { bytes: direct.size_bytes, estimated: true };
  if (totalParams && totalParams > 0) {
    return { bytes: totalParams * Q4_BYTES_PER_PARAM, estimated: true };
  }
  return null;
}

export interface Q4nxMemory {
  weightsGb: number;
  neededGb: number;
  /** null when the window is unknown. */
  fits: boolean | null;
  /** Share of the window the weights and the load margin take, each 0 to 100. */
  weightsPct: number;
  marginPct: number;
  label: string;
}

export function q4nxNpuMemory(
  weights: Q4nxWeights | null,
  windowGb: number | null | undefined,
): Q4nxMemory | null {
  if (!weights) return null;
  const weightsGb = weights.bytes / GIB;
  const neededGb = weightsGb + NPU_LOAD_MARGIN_GB;
  const need = `${weights.estimated ? "~" : ""}${neededGb.toFixed(1)}`;
  if (!windowGb || windowGb <= 0) {
    return { weightsGb, neededGb, fits: null, weightsPct: 0, marginPct: 0, label: `${need} GB NPU` };
  }
  const weightsPct = Math.min((weightsGb / windowGb) * 100, 100);
  return {
    weightsGb,
    neededGb,
    // Compared at one decimal, like the backend and the label.
    fits: Number(neededGb.toFixed(1)) <= windowGb,
    weightsPct,
    marginPct: Math.min((NPU_LOAD_MARGIN_GB / windowGb) * 100, 100 - weightsPct),
    label: `${need} of ${windowGb.toFixed(1)} GB NPU`,
  };
}

/** The warning for a source larger than the window, or null. It only warns: the window is an estimate. */
export function q4nxMemoryWarning(
  memory: Q4nxMemory | null,
  windowGb: number | null | undefined,
): string | null {
  if (memory?.fits !== false || windowGb == null) return null;
  return `Needs about ${memory.neededGb.toFixed(1)} GB of NPU memory; this PC can give the NPU about ${windowGb.toFixed(1)} GB.`;
}

export function q4nxQuantName(quant: string): string {
  const label = q4nxQuantLabel(quant);
  const match = /(?:^|[-_.])((?:UD-)?I?Q\d\w*|BF16|F16|F32)$/i.exec(label);
  return match ? match[1].toUpperCase() : label;
}

export const Q4NX_GUIDE_STEPS = [
  "Pick a GGUF model from Hugging Face or from this device",
  "Choose its GGUF file: Q4_K_M always works, Q4_0 or Q4_1 only where it matches the model",
  "Check the original model and save directory in the convert settings",
  "Click Convert to write the Q4NX files",
] as const;

export const Q4NX_PACKING_TEXT: Record<Q4nxPacking, string> = {
  direct: "packs directly",
  "by-architecture": "packs directly for Qwen3.5 only",
  requantized: "re-quantized to 4-bit",
  mismatch: "not readable by this model",
};

export interface Q4nxSummaryInput {
  model: string | null;
  file: string | null;
  quant: string | null;
  packing: Q4nxPacking | null;
  baseModel: string;
  saveDirectory: string;
  size: string | null;
  memory: Pick<Q4nxMemory, "fits" | "label"> | null;
  archFit: Q4nxArchFit | null;
}

export function q4nxSummaryRows(
  input: Q4nxSummaryInput,
): { label: string; value: string; mono?: boolean }[] {
  const { archFit, memory, quant } = input;
  const dash = "\u2014";
  return [
    { label: "Model", value: input.model || dash },
    { label: "GGUF file", value: input.file || dash },
    {
      label: "Quant",
      value:
        quant && input.packing
          ? `${q4nxQuantName(quant)}, ${Q4NX_PACKING_TEXT[input.packing]}`
          : dash,
    },
    { label: "Original model", value: input.baseModel.trim() || dash },
    { label: "Save directory", value: input.saveDirectory.trim() || dash },
    { label: "Est. Q4NX size", value: input.size ?? dash },
    {
      label: "NPU memory",
      value: memory
        ? memory.fits === false
          ? `May not fit: ${memory.label}`
          : memory.label
        : dash,
    },
    {
      label: "Convertible",
      value: !archFit
        ? dash
        : archFit.status === "unsupported" || input.packing === "mismatch"
          ? "No"
          : archFit.status === "unknown"
            ? "Checked during conversion"
            : (archFit.note ?? "Yes"),
    },
  ];
}

export function localGgufPath(
  rowPath: string,
  variant: { filename: string } | null,
): string | null {
  if (/\.gguf$/i.test(rowPath.trim())) return rowPath.trim();
  if (!variant) return null;
  const sep = rowPath.includes("\\") && !rowPath.includes("/") ? "\\" : "/";
  return `${rowPath.replace(/[\\/]+$/, "")}${sep}${variant.filename.replace(/^[\\/]+/, "")}`;
}
