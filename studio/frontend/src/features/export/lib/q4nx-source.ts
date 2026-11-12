// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Mirrors SOURCE_QUANTS in backend core/export/q4nx.py; any other quant is dequantized and rounded again.
const DIRECT_QUANTS = ["q4_0", "q4_1", "q4_k_m"];
// Only the q4k converter unpacks Q4_K, and only these architectures use it
// (q4k_architectures in q4nx_converter_pins.json); the q4nx converter rounds Q4_K_M again.
const Q4K_ARCHITECTURES = ["qwen35", "qwen35moe"];

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

// Parameter counts, in billions, behind the size caveats above. Qwen3.5 Omni is matched by name.
const Q4NX_SIZES: Record<string, readonly number[]> = {
  qwen35: [0.8, 2, 4, 9],
  qwen35moe: [35],
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

/** "by-architecture": Q4_K_M of a file whose architecture is not known yet. */
export type Q4nxPacking = "direct" | "by-architecture" | "requantized";

export function q4nxPacking(
  quant: string,
  architecture?: string | null,
): Q4nxPacking {
  const rank = directRank(quant);
  if (rank === DIRECT_QUANTS.length) return "requantized";
  if (DIRECT_QUANTS[rank] !== "q4_k_m") return "direct";
  const arch = architecture?.trim().toLowerCase();
  if (!arch) return "by-architecture";
  return Q4K_ARCHITECTURES.includes(arch) ? "direct" : "requantized";
}

export function packsDirectly(quant: string, architecture?: string | null): boolean {
  return q4nxPacking(quant, architecture) === "direct";
}

const PACKING_ORDER: Q4nxPacking[] = ["direct", "by-architecture", "requantized"];

/** A 4xx from the convert endpoint: the same request gets the same answer. 5xx and network drops do not count. */
export function q4nxIsRefusal(error: unknown): boolean {
  const status = (error as { status?: unknown } | null)?.status;
  return typeof status === "number" && status >= 400 && status < 500;
}

// A split GGUF is listed as one variant named by its first shard; the converter reads one file.
const SPLIT_SHARD = /-\d{3,}-of-\d{3,}\.gguf$/i;

/** Drops what cannot be converted: empty leftover quant folders (no file) and split GGUFs. */
export function q4nxConvertibleFiles<T extends { filename: string; cleanable?: boolean }>(
  variants: T[],
): T[] {
  return variants.filter((v) => !v.cleanable && !SPLIT_SHARD.test(v.filename));
}

/** Directly packed quants first (backend order), then the rest; the first one is the default pick. */
export function sortQ4nxVariants<T extends Q4nxVariant>(
  variants: T[],
  architecture?: string | null,
): T[] {
  const rank = (v: T) =>
    PACKING_ORDER.indexOf(q4nxPacking(v.quant, architecture)) * 10 + directRank(v.quant);
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

// "Qwen3.5-9B", "Qwen3-0.6B-GGUF"; in "35B-A3B" the first, total count wins.
const NAME_PARAMS_RE = /(?:^|[^A-Za-z0-9.])(\d+(?:\.\d+)?)B(?![A-Za-z0-9])/i;

/** Parameter count read off a model name, for a row with no count from the Hub. */
export function q4nxParamsFromName(name: string): number | null {
  const billions = Number(NAME_PARAMS_RE.exec(name)?.[1]);
  return billions > 0 ? billions * 1e9 : null;
}

export type Q4nxArchFit =
  | { status: "supported"; note: string | null }
  // `sized`: the architecture converts, this model's size does not.
  | { status: "unsupported"; note: string; sized?: true }
  | { status: "unknown"; note: string };

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
  const caveat = Q4NX_ARCHITECTURES[arch];
  const sizes = Q4NX_SIZES[arch];
  if (!sizes) return { status: "supported", note: caveat };
  // A size named in the repo or file settles the caveat up front, not after the download.
  const billions = (q4nxParamsFromName(name ?? "") ?? 0) / 1e9;
  if (/omni/i.test(name ?? "") || sizes.includes(billions)) {
    return { status: "supported", note: null };
  }
  if (billions) {
    return {
      status: "unsupported",
      note: `The Q4NX converter does not support this ${billions}B model. ${caveat}`,
      sized: true,
    };
  }
  return { status: "supported", note: caveat };
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

/** Which models show their quantizations. "Expand quantizations" opens every model; without it the list is by model, one open at a time. */
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

/** Size of the 4-bit weights a source converts to: its own when it packs directly, else a sibling that does, else from the parameter count. */
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

/** A source's NPU memory need against the window this PC can give the NPU (status.memory_window_gb). */
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

/** The quant alone ("Q4_K_M") out of a variant key or a file stem such as "Qwen3-0.6B-Q4_K_M". */
export function q4nxQuantName(quant: string): string {
  const label = q4nxQuantLabel(quant);
  const match = /(?:^|[-_.])((?:UD-)?I?Q\d\w*|BF16|F16|F32)$/i.exec(label);
  return match ? match[1].toUpperCase() : label;
}

export const Q4NX_GUIDE_STEPS = [
  "Pick a GGUF model from Hugging Face or from this device",
  "Choose its GGUF file: Q4_0 and Q4_1 pack directly, Q4_K_M for Qwen3.5",
  "Check the original model and save directory in the convert settings",
  "Click Convert to write the Q4NX files",
] as const;

export const Q4NX_PACKING_TEXT: Record<Q4nxPacking, string> = {
  direct: "packs directly",
  "by-architecture": "packs directly for Qwen3.5 only",
  requantized: "re-quantized to 4-bit",
};

export interface Q4nxSummaryInput {
  model: string | null;
  /** The picked GGUF file's name and quant key; null until one is picked. */
  file: string | null;
  quant: string | null;
  packing: Q4nxPacking | null;
  baseModel: string;
  saveDirectory: string;
  /** Formatted size of the converted weights. */
  size: string | null;
  memory: Pick<Q4nxMemory, "fits" | "label"> | null;
  archFit: Q4nxArchFit | null;
}

/** The export page's summary rows while the Q4NX card is in use. */
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
        : archFit.status === "unsupported"
          ? "No"
          : archFit.status === "unknown"
            ? "Checked during conversion"
            : (archFit.note ?? "Yes"),
    },
  ];
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
