// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Presets, prompts and the numbers behind the chart. Kept free of React and of `@/`
// imports so node --test can load it directly.

/** Load overrides one variant applies on top of the model's current load. */
export interface VariantLoad {
  speculative_type?: string | null;
  spec_draft_n_max?: number | null;
  cache_type_kv?: string | null;
  max_seq_length?: number;
  n_parallel?: number | null;
  spec_draft_cache_type?: string | null;
  gpu_memory_mode?: "auto" | "manual";
  gpu_layers?: number;
  n_cpu_moe?: number;
  // biome-ignore lint/style/useNamingConvention: API schema
  llama_extra_args?: string[] | null;
}

/** What the offload sweep scales its rows by: the model's block and MoE layer counts. */
export interface ModelShape {
  layers: number | null;
  moeLayers: number | null;
}

export interface Variant {
  label: string;
  load: VariantLoad;
}

export type SweepKind =
  | "spec"
  | "draft"
  | "ngram"
  | "dspark"
  | "kv"
  | "context"
  | "parallel"
  | "offload"
  | "tune";
/** The sweep picker's presets. Auto-tune is its own card, not a preset. */
export const SWEEP_KINDS: SweepKind[] = [
  "spec",
  "draft",
  "ngram",
  "dspark",
  "kv",
  "context",
  "parallel",
  "offload",
];

export const SWEEP_TITLE: Record<SweepKind, string> = {
  spec: "Speculative decoding",
  draft: "MTP draft depth",
  ngram: "Ngram tuning",
  dspark: "DSpark tuning",
  kv: "KV cache type",
  context: "Context length",
  parallel: "Parallel slots",
  offload: "RAM offload",
  tune: "Auto-tune",
};

export const SWEEP_BLURB: Record<SweepKind, string> = {
  spec: "Every mode Studio offers, from off to MTP + ngram. Modes this model can't run are skipped with the reason.",
  draft:
    "Speculation off against MTP at 1 to 8 draft tokens. Finds the depth where drafting stops paying.",
  ngram:
    "Ngram alone, MTP alone and both together, across the ngram key length and discard floor.",
  dspark:
    "Speculation off against the DSpark drafter at 1 to 8 draft tokens, plus its draft cache at q8_0 and q4_0. Needs a model that ships one.",
  kv: "f16, q8_0 and q4_0 KV cache. Smaller caches fit more context but can cost speed.",
  context:
    "The same prompt at growing context windows, with load time for each.",
  parallel:
    "1, 2 and 4 decode slots. More slots serve more users, each a little slower.",
  offload:
    "For MoE models bigger than VRAM: Studio's own fit, then expert layers moved to RAM from all of them down to none. Every row runs at 8K context. Dense models get GPU layer steps instead.",
  tune: "Off, MTP at 1 to 4 draft tokens, ngram, and MTP + ngram. Picks the fastest and hands it to chat.",
};

/** Fixed colour slot per speculative family: colour follows the mode, never its rank. */
export const FAMILIES = [
  "mtp",
  "mtp+ngram",
  "ngram",
  "auto",
  "dspark",
  "dflash",
  "other",
  "off",
] as const;
export type Family = (typeof FAMILIES)[number];

export const FAMILY_LABEL: Record<Family, string> = {
  mtp: "MTP",
  "mtp+ngram": "MTP + ngram",
  ngram: "Ngram",
  auto: "Auto",
  dspark: "DSpark",
  dflash: "DFlash",
  other: "Other",
  off: "Speculation off",
};

export function familyOf(load: VariantLoad): Family {
  if (load.speculative_type === undefined) return "other";
  const raw = String(load.speculative_type ?? "auto")
    .toLowerCase()
    .trim();
  if (raw === "off" || raw === "none" || raw === "disabled") return "off";
  if (raw === "mtp+ngram" || raw === "ngram+mtp") return "mtp+ngram";
  if (raw === "mtp" || raw === "draft-mtp") return "mtp";
  if (raw === "ngram" || raw === "ngram-mod" || raw === "ngram-simple")
    return "ngram";
  if (raw === "auto" || raw === "default" || raw === "") return "auto";
  if (raw === "dspark" || raw === "draft-dspark") return "dspark";
  if (raw === "dflash" || raw === "draft-dflash") return "dflash";
  return "other";
}

/** Studio's spellings folded to one, so a requested mode can be checked against the served one. */
export function canonicalSpec(raw: string | null | undefined): string {
  const s = String(raw ?? "")
    .toLowerCase()
    .trim();
  if (!s || s === "default") return "auto";
  if (s === "none" || s === "disable" || s === "disabled") return "off";
  if (s.includes(",")) {
    const mtp = s.includes("mtp");
    const ngram = s.includes("ngram");
    if (mtp && ngram) return "mtp+ngram";
    if (mtp) return "mtp";
    if (ngram) return "ngram";
  }
  if (s === "draft-mtp") return "mtp";
  if (s === "ngram-mod") return "ngram";
  if (s === "draft-dspark") return "dspark";
  if (s === "draft-dflash") return "dflash";
  return s;
}

// llama.cpp's ngram-mod drafter as two one-variable sweeps, so a winning rung names its knob:
// the floor sweep holds the key at stock 24, the key sweep holds the floor at 8.
export const NGRAM_N_MAX = 64;
export interface NgramStep {
  match: number;
  min: number;
}
export const NGRAM_STOCK: NgramStep = { match: 24, min: 48 };
const FLOOR_SWEEP: NgramStep[] = [2, 8, 16, 32, 48].map((min) => ({
  match: 24,
  min,
}));
const KEY_SWEEP: NgramStep[] = [4, 8, 12, 16, 24].map((match) => ({
  match,
  min: 8,
}));
export const NGRAM_STEPS: NgramStep[] = [...FLOOR_SWEEP, ...KEY_SWEEP].filter(
  (s, i, all) =>
    all.findIndex((o) => o.match === s.match && o.min === s.min) === i,
);

export function ngramArgs(step: NgramStep): string[] {
  return [
    "--spec-ngram-mod-n-match",
    String(step.match),
    "--spec-ngram-mod-n-min",
    String(step.min),
    "--spec-ngram-mod-n-max",
    String(NGRAM_N_MAX),
  ];
}

function ngramTag(s: NgramStep): string {
  const stock = s.match === NGRAM_STOCK.match && s.min === NGRAM_STOCK.min;
  return `match ${s.match}, min ${s.min}${stock ? " (stock)" : ""}`;
}

const OFF: Variant = {
  label: "Speculation off",
  load: { speculative_type: "off", spec_draft_n_max: null },
};

/** The presets behind the sweep picker. Labels double as row identities. */
export function sweepVariants(
  kind: SweepKind,
  maxContext?: number | null,
  shape?: ModelShape | null,
): Variant[] {
  switch (kind) {
    case "offload":
      return offloadVariants(shape);
    case "spec":
      return [
        OFF,
        {
          label: "Auto (Studio default)",
          load: { speculative_type: "auto", spec_draft_n_max: null },
        },
        {
          label: "Ngram",
          load: { speculative_type: "ngram", spec_draft_n_max: null },
        },
        ...[2, 3, 4].map((n) => ({
          label: `MTP · ${n} draft tokens`,
          load: { speculative_type: "mtp", spec_draft_n_max: n },
        })),
        ...[2, 3, 4].map((n) => ({
          label: `MTP + ngram · ${n} draft tokens`,
          load: { speculative_type: "mtp+ngram", spec_draft_n_max: n },
        })),
        {
          label: "DSpark · 3 draft tokens",
          load: { speculative_type: "dspark", spec_draft_n_max: 3 },
        },
        {
          label: "DFlash · 3 draft tokens",
          load: { speculative_type: "dflash", spec_draft_n_max: 3 },
        },
      ];
    case "draft":
      return [
        OFF,
        ...[1, 2, 3, 4, 6, 8].map((n) => ({
          label: `MTP · ${n} draft tokens`,
          load: { speculative_type: "mtp", spec_draft_n_max: n },
        })),
      ];
    case "ngram":
      return [
        OFF,
        ...[2, 3, 4].map((n) => ({
          label: `MTP · ${n} draft tokens`,
          load: { speculative_type: "mtp", spec_draft_n_max: n },
        })),
        ...NGRAM_STEPS.flatMap((s) => [
          {
            label: `Ngram · ${ngramTag(s)}`,
            load: {
              speculative_type: "ngram",
              spec_draft_n_max: null,
              llama_extra_args: ngramArgs(s),
            },
          },
          {
            label: `MTP + ngram · ${ngramTag(s)}`,
            load: {
              speculative_type: "mtp+ngram",
              spec_draft_n_max: 3,
              llama_extra_args: ngramArgs(s),
            },
          },
        ]),
      ];
    case "dspark":
      return [
        OFF,
        ...[1, 2, 3, 4, 6, 8].map((n) => ({
          label: `DSpark · ${n} draft token${n === 1 ? "" : "s"}`,
          load: { speculative_type: "dspark", spec_draft_n_max: n },
        })),
        ...["q8_0", "q4_0"].map((t) => ({
          label: `DSpark · 3 draft tokens · draft KV ${t}`,
          load: {
            speculative_type: "dspark",
            spec_draft_n_max: 3,
            spec_draft_cache_type: t,
          },
        })),
        {
          label: "MTP · 3 draft tokens",
          load: { speculative_type: "mtp", spec_draft_n_max: 3 },
        },
      ];
    case "kv":
      return ["f16", "q8_0", "q4_0"].map((t) => ({
        label: `KV cache ${t}`,
        load: { cache_type_kv: t },
      }));
    case "context":
      return contextSteps(maxContext).map((n) => ({
        label: `Context ${fmtTokens(n)}`,
        load: { max_seq_length: n },
      }));
    case "parallel":
      return [1, 2, 4].map((n) => ({
        label: `${n} parallel slot${n === 1 ? "" : "s"}`,
        load: { n_parallel: n },
      }));
    case "tune":
      return [
        OFF,
        ...[1, 2, 3, 4].map((n) => ({
          label: `MTP · ${n} draft token${n === 1 ? "" : "s"}`,
          load: { speculative_type: "mtp", spec_draft_n_max: n },
        })),
        {
          label: `Ngram · ${ngramTag(NGRAM_STOCK)}`,
          load: {
            speculative_type: "ngram",
            spec_draft_n_max: null,
            llama_extra_args: ngramArgs(NGRAM_STOCK),
          },
        },
        ...[2, 3].map((n) => ({
          label: `MTP + ngram · ${n} draft tokens`,
          load: {
            speculative_type: "mtp+ngram",
            spec_draft_n_max: n,
            llama_extra_args: ngramArgs(NGRAM_STOCK),
          },
        })),
      ];
  }
}

/** Context every offload row loads at: Studio's own offload cap, so the auto row compares. */
export const OFFLOAD_CONTEXT = 8192;

/**
 * Studio's fit, then manual placements from the least VRAM to the most, so the first out of
 * memory row lets the runner skip the rest. MoE models move expert layers (n_cpu_moe, which the
 * backend clamps and offsets past dense layers); dense ones step GPU layers.
 */
export function offloadVariants(shape?: ModelShape | null): Variant[] {
  const ctx = OFFLOAD_CONTEXT;
  const auto: Variant = {
    label: "Studio auto",
    load: { gpu_memory_mode: "auto", gpu_layers: -1, n_cpu_moe: 0, max_seq_length: ctx },
  };
  const manual = (gpu_layers: number, n_cpu_moe: number) => ({
    gpu_memory_mode: "manual" as const,
    gpu_layers,
    n_cpu_moe,
    max_seq_length: ctx,
  });
  const moe = shape?.moeLayers ?? null;
  const layers = shape?.layers ?? null;
  if (moe === 0 && layers) {
    const steps = [...new Set([0.25, 0.5, 0.75].map((f) => Math.max(1, Math.round(layers * f))))];
    return [
      auto,
      ...steps.map((k) => ({
        label: `GPU layers · ${k} of ${layers}`,
        load: manual(k, 0),
      })),
      { label: "All on GPU", load: manual(999, 0) },
    ];
  }
  // Unknown shape: the backend clamps, so fixed counts still read sensibly.
  const counts = moe
    ? [...new Set([0.75, 0.5, 0.25].map((f) => Math.round(moe * f)))].filter(
        (n) => n > 0 && n < moe,
      )
    : [32, 24, 16, 8];
  return [
    auto,
    {
      label: moe ? `Experts on CPU · all ${moe} layers` : "Experts on CPU · all",
      load: manual(999, moe ?? 999),
    },
    ...counts.map((n) => ({
      label: moe ? `Experts on CPU · ${n} of ${moe} layers` : `Experts on CPU · ${n} layers`,
      load: manual(999, n),
    })),
    { label: "All on GPU", load: manual(999, 0) },
  ];
}

/** How much VRAM a manual row asks for, higher is more: the runner's skip-after-OOM order. */
export function offloadDemand(load: VariantLoad): number | null {
  if (load.gpu_memory_mode !== "manual") return null;
  return (load.gpu_layers ?? 999) * 1000 - (load.n_cpu_moe ?? 0);
}

/** How much a setting asks of the machine: the tie-break when two rows run alike. */
function complexity(v: Variant | undefined): number {
  if (!v) return 99;
  const family = familyOf(v.load);
  const rank: Record<Family, number> = {
    off: 0,
    ngram: 1,
    mtp: 2,
    "mtp+ngram": 3,
    auto: 4,
    dspark: 5,
    dflash: 5,
    other: 6,
  };
  return (
    rank[family] * 10 +
    (v.load.spec_draft_n_max ?? 0) +
    (v.load.llama_extra_args?.length ? 1 : 0)
  );
}

export interface TuneVerdict {
  /** The row auto-tune hands to chat. */
  pick: AggRow;
  /** The raw fastest row, when the pick gave it up for a simpler setting inside the margin. */
  fastest: AggRow | null;
  off: AggRow | null;
  /** Percent over speculation off, or null without an off row. */
  gain: number | null;
}

/** Within 3% of the fastest, the simpler setting wins: the difference is noise, the extra work is not. */
export const TUNE_MARGIN = 0.03;

export function tuneVerdict(
  rows: AggRow[],
  variants: Variant[],
): TuneVerdict | null {
  if (rows.length === 0) return null;
  const fastest = rows.reduce((a, b) => (b.mean > a.mean ? b : a));
  const close = rows.filter((r) => r.mean >= fastest.mean * (1 - TUNE_MARGIN));
  const byLabel = new Map(variants.map((v) => [v.label, v]));
  const pick = close.reduce((a, b) =>
    complexity(byLabel.get(b.label)) < complexity(byLabel.get(a.label)) ? b : a,
  );
  const off =
    rows.find((r) => familyOf(byLabel.get(r.label)?.load ?? {}) === "off") ??
    null;
  return {
    pick,
    fastest: pick === fastest ? null : fastest,
    off,
    gain:
      off && off.mean > 0 ? ((pick.mean - off.mean) / off.mean) * 100 : null,
  };
}

/** Powers of two from 4K up to the model's window, capped at five rows. */
export function contextSteps(maxContext?: number | null): number[] {
  const top = maxContext && maxContext > 0 ? maxContext : 32768;
  const out: number[] = [];
  for (let n = 4096; n <= top; n *= 2) out.push(n);
  if (out.length === 0) out.push(top);
  else if (out[out.length - 1] !== top && top > out[out.length - 1])
    out.push(top);
  return out.slice(-5);
}

export function fmtTokens(n: number): string {
  if (n >= 1024 && n % 1024 === 0) return `${n / 1024}K`;
  return n.toLocaleString("en-US");
}

/** Which row is the natural baseline for a preset, when it has one. */
export function defaultBaseline(variants: Variant[]): string | null {
  const auto = variants.find((v) => v.load.gpu_memory_mode === "auto");
  if (auto) return auto.label;
  const off = variants.find((v) => familyOf(v.load) === "off");
  return off?.label ?? variants[0]?.label ?? null;
}

// --- load planning ------------------------------------------------------------

/** The slice of a load request the planner touches; the rest passes through untouched. */
export interface LoadPayload extends VariantLoad {
  model_path: string;
  force_reload?: boolean;
}

/** The slice of /api/inference/status a served-config check reads. */
export interface ServedStatus {
  speculative_type?: string | null;
  spec_fallback_reason?: string | null;
  cache_type_kv?: string | null;
  /** Slots the server actually serves after load_model's clamps. */
  parallel_slots?: number | null;
}

const NGRAM_FLAGS = [
  "--spec-ngram-mod-n-match",
  "--spec-ngram-mod-n-min",
  "--spec-ngram-mod-n-max",
];

// The backend's fallback codes, in words a user can act on.
const FALLBACK_REASON: Record<string, string> = {
  binary_no_mtp: "this llama.cpp build has no MTP support",
  binary_outdated: "the installed llama.cpp is too old for this mode",
  drafter_not_found: "this model ships no drafter for this mode",
  runtime_error: "llama-server refused the drafter at start-up",
  mtp_partial_offload: "MTP needs the whole model on the GPU",
  drafter_no_vram: "no VRAM left for the drafter",
  mla_mtp_disabled: "MTP is disabled for this attention type",
};

export function describeFallback(code: string): string {
  return FALLBACK_REASON[code] ?? code.replace(/_/g, " ");
}

/** The user's own extra args, minus any ngram tuning an earlier variant left behind. */
export function userExtraArgs(
  args: readonly string[] | null | undefined,
): string[] {
  const out: string[] = [];
  const list = args ?? [];
  for (let i = 0; i < list.length; i++) {
    if (NGRAM_FLAGS.includes(list[i])) {
      i++;
      continue;
    }
    out.push(list[i]);
  }
  return out;
}

export function variantLoad<T extends LoadPayload>(
  base: T,
  variant: Variant,
): T {
  const { llama_extra_args: extra, ...rest } = variant.load;
  return {
    ...base,
    ...rest,
    // Always explicit: omitted, Studio re-applies the model's stored args, which after
    // an ngram row would carry that row's tuning into the next one.
    llama_extra_args: [...(base.llama_extra_args ?? []), ...(extra ?? [])],
    // A fresh server per row: timings include the load, and no drafter cache carries over.
    force_reload: true,
  };
}

/** Why a load did not give this row what it asked for, or null when it did. */
export function servedMismatch(
  variant: Variant,
  st: ServedStatus,
): string | null {
  const want = variant.load.speculative_type;
  // Only rows that pick a mode can be let down by one; a KV or context row keeps whatever ran before.
  if (want !== undefined && want !== null && st.spec_fallback_reason)
    return describeFallback(st.spec_fallback_reason);
  if (want !== undefined && want !== null && canonicalSpec(want) !== "auto") {
    const got = canonicalSpec(st.speculative_type);
    if (got !== canonicalSpec(want))
      return `Studio served ${got} instead of ${canonicalSpec(want)}`;
  }
  if (
    variant.load.cache_type_kv &&
    st.cache_type_kv &&
    st.cache_type_kv !== variant.load.cache_type_kv
  ) {
    return `Studio served KV ${st.cache_type_kv} instead of ${variant.load.cache_type_kv}`;
  }
  // A build without --kv-unified clamps multi-slot loads back to one, so a 2- or 4-slot row
  // would measure the same one-slot server; skip it instead of charting it as its own config.
  const wantSlots = variant.load.n_parallel;
  if (
    wantSlots != null &&
    wantSlots > 1 &&
    typeof st.parallel_slots === "number" &&
    st.parallel_slots < wantSlots
  ) {
    return `Studio served ${st.parallel_slots} parallel slot${st.parallel_slots === 1 ? "" : "s"} instead of ${wantSlots}`;
  }
  return null;
}

// --- prompts -----------------------------------------------------------------

export interface PromptSet {
  id: string;
  name: string;
  hint: string;
  prompts: string[];
}

// Several prompts per set so measured runs can rotate: llama.cpp's ngram cache learns a
// repeated answer, and one prompt sent N times measures copying, not drafting.
export const PROMPT_SETS: PromptSet[] = [
  {
    id: "chat",
    name: "Everyday chat",
    hint: "Short questions, short answers",
    prompts: [
      "Explain how a refrigerator keeps food cold, in plain language, in about two paragraphs.",
      "What are three good habits for keeping a small vegetable garden healthy through a hot summer?",
      "I have eggs, spinach, rice and a lemon. Suggest a simple dinner and walk me through making it.",
      "Why does the sky look red at sunset but blue at noon? Keep it friendly and concrete.",
      "Give me a gentle plan for getting back into running after a year off, week by week for a month.",
    ],
  },
  {
    id: "code",
    name: "Code",
    hint: "Structured, repetitive output",
    prompts: [
      "Write a Python function that parses a CSV file of orders (id, customer, amount, date) and returns total spend per customer per month. Include type hints and a short docstring.",
      "Implement a small LRU cache class in TypeScript with get, set and a max size, plus three example usages.",
      "Write a bash script that finds every .log file older than 7 days under a directory, compresses it with gzip, and prints a summary of the space saved.",
      "Write a SQL schema for a library: books, authors (many-to-many), members and loans, then a query listing overdue loans with member names.",
      "Write a Rust function that reads a text file and returns the ten most common words with their counts, ignoring case and punctuation.",
    ],
  },
  {
    id: "writing",
    name: "Long-form writing",
    hint: "Open-ended prose",
    prompts: [
      "Write a short story of about 400 words about a lighthouse keeper who finds a message in a bottle addressed to them.",
      "Write a detailed product description for a waterproof hiking backpack, covering materials, capacity, comfort and who it is for.",
      "Write a blog post introduction and outline on why learning to cook at home saves money and time.",
      "Describe a busy morning market in a coastal town, using all five senses, in about 350 words.",
      "Write a thoughtful letter from a grandparent to a grandchild starting university, full of practical advice.",
    ],
  },
];

export const CUSTOM_PROMPT_ID = "custom";

export function promptsFor(setId: string, custom: string): string[] {
  if (setId === CUSTOM_PROMPT_ID) {
    const lines = custom
      .split(/\n\s*---+\s*\n/)
      .map((p) => p.trim())
      .filter(Boolean);
    return lines.length ? lines : [];
  }
  return (
    PROMPT_SETS.find((s) => s.id === setId)?.prompts ?? PROMPT_SETS[0].prompts
  );
}

// --- results -------------------------------------------------------------------

export interface RunResult {
  variant: string;
  /** 0-based, warm-ups included. */
  rep: number;
  warmup: boolean;
  promptIndex: number;
  /** Server decode rate (llama-server timings.predicted_per_second). */
  tps: number | null;
  promptTps: number | null;
  promptTokens: number | null;
  genTokens: number | null;
  ttftMs: number | null;
  wallMs: number;
  /** Client-side decode rate, the stand-in when the server sent no timings. */
  clientTps: number | null;
  draftN: number | null;
  draftAccepted: number | null;
  /** Set on the first run after a load. */
  loadMs: number | null;
  at: number;
}

export type VariantState =
  "queued" | "loading" | "running" | "done" | "skipped" | "error" | "cancelled";

export interface VariantOutcome {
  label: string;
  state: VariantState;
  /** Why a row was skipped or failed, in the user's words where we can. */
  reason?: string;
  /** What Studio actually served after the load. */
  served?: Record<string, unknown>;
}

export interface RunMeta {
  gpu?: string | null;
  vramGb?: number | null;
  backend?: string | null;
  /** The GPU runtime torch was built against, e.g. "ROCm 7.1" or "CUDA 12.8". */
  runtime?: string | null;
  llamaTag?: string | null;
  studioVersion?: string | null;
  /** platform.platform(): "Windows-11-10.0.26200", "Linux-6.8...", "macOS-15.3-arm64". */
  os?: string | null;
  cpuThreads?: number | null;
  ramGb?: number | null;
}

export interface BenchConfig {
  sweep: SweepKind;
  /** The model to benchmark when it is not the one chat has loaded (any kind of run). */
  tuneModel?: string | null;
  /** Its GGUF quant, from the model picker. */
  tuneVariant?: string | null;
  variants: Variant[];
  baseline: string | null;
  promptSet: string;
  customPrompt: string;
  rotatePrompts: boolean;
  maxTokens: number;
  warmup: number;
  repetitions: number;
  temperature: number;
  seed: number;
  restoreAfter: boolean;
}

/** One Run Settings value from chat, as the page and the export print it. */
export interface BaseSetting {
  /** The load field it feeds, so a sweep that varies it can say so. */
  field: string;
  label: string;
  value: string;
}

export interface BenchRun {
  id: string;
  createdAt: number;
  finishedAt: number | null;
  model: string;
  ggufVariant: string | null;
  kv: string | null;
  context: number | null;
  config: BenchConfig;
  meta: RunMeta;
  /** Chat's Run Settings when the sweep started: every row loads these plus its own overrides. */
  base?: BaseSetting[];
  outcomes: VariantOutcome[];
  results: RunResult[];
}

export interface AggRow {
  label: string;
  family: Family;
  mean: number;
  min: number;
  max: number;
  n: number;
  /** Percent over the baseline mean; null for the baseline itself or without one. */
  pct: number | null;
  isBaseline: boolean;
  ttftMs: number | null;
  loadMs: number | null;
  acceptRate: number | null;
  /** The first measured run, which the ngram cache has not seen yet. */
  first: number | null;
  clientOnly: boolean;
}

function mean(xs: number[]): number | null {
  return xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : null;
}

function finite(x: number | null): x is number {
  return x !== null && Number.isFinite(x);
}

/**
 * One row per variant with a measured (non-warm-up) run, fastest first and the baseline
 * last. Throughput is the server's own rate; the client rate stands in only when a run
 * came back without timings, and the row says so.
 */
export function aggregate(
  results: RunResult[],
  variants: Variant[],
  baseline: string | null,
): AggRow[] {
  const byLabel = new Map<string, RunResult[]>();
  for (const r of results) {
    if (r.warmup) continue;
    const list = byLabel.get(r.variant) ?? [];
    list.push(r);
    byLabel.set(r.variant, list);
  }
  const known = new Map(variants.map((v) => [v.label, v]));
  const rows: AggRow[] = [];
  for (const [label, list] of byLabel) {
    list.sort((a, b) => a.rep - b.rep);
    const server = list.map((r) => r.tps).filter(finite);
    // Fall back to the client rate per completion, not per row, so one run that came back
    // without server timings doesn't drop out of the mean, range and sample count.
    const rates = list.map((r) => r.tps ?? r.clientTps).filter(finite);
    if (!rates.length) continue;
    const drafts = list.reduce((a, r) => a + (r.draftN ?? 0), 0);
    const accepted = list.reduce((a, r) => a + (r.draftAccepted ?? 0), 0);
    const variant = known.get(label);
    rows.push({
      label,
      family: variant ? familyOf(variant.load) : "other",
      mean: mean(rates) ?? 0,
      min: Math.min(...rates),
      max: Math.max(...rates),
      n: rates.length,
      pct: null,
      isBaseline: baseline !== null && label === baseline,
      ttftMs: mean(list.map((r) => r.ttftMs).filter(finite)),
      loadMs: mean(
        results
          .filter((r) => r.variant === label)
          .map((r) => r.loadMs)
          .filter(finite),
      ),
      acceptRate: drafts > 0 ? accepted / drafts : null,
      first: rates[0] ?? null,
      clientOnly: server.length === 0,
    });
  }
  const base = rows.find((r) => r.isBaseline);
  if (base && base.mean > 0) {
    for (const r of rows)
      if (!r.isBaseline) r.pct = ((r.mean - base.mean) / base.mean) * 100;
  }
  rows.sort(
    (a, b) => Number(a.isBaseline) - Number(b.isBaseline) || b.mean - a.mean,
  );
  return rows;
}

/** The one sentence under the chart title. */
export function headline(rows: AggRow[]): string | undefined {
  const base = rows.find((r) => r.isBaseline);
  const others = rows.filter((r) => !r.isBaseline);
  if (others.length === 0) return undefined;
  const best = others[0];
  if (!base) return `${best.label} is fastest at ${fmtRate(best.mean)}.`;
  if (best.mean <= base.mean)
    return `Nothing beats ${inSentence(base.label)} on this model.`;
  const beat = others.filter((r) => r.mean > base.mean).length;
  const x = best.mean / base.mean;
  return `${best.label} runs ${x.toFixed(x >= 10 ? 0 : 1)}× ${inSentence(base.label)}. ${beat} of ${others.length} settings beat it.`;
}

export interface Highlights {
  best: AggRow | null;
  baseline: AggRow | null;
  speedup: number | null;
  bestAccept: AggRow | null;
}

export function highlights(rows: AggRow[]): Highlights {
  const baseline = rows.find((r) => r.isBaseline) ?? null;
  const others = rows.filter((r) => !r.isBaseline);
  // The fastest row overall, which can be the baseline (an offload sweep's Studio fit).
  const best =
    baseline && (!others[0] || baseline.mean >= others[0].mean)
      ? baseline
      : (others[0] ?? null);
  // The best challenger against the baseline, below 1 when nothing beat it.
  const speedup =
    others[0] && baseline && baseline.mean > 0
      ? others[0].mean / baseline.mean
      : null;
  const withAccept = others.filter((r) => r.acceptRate !== null);
  const bestAccept = withAccept.length
    ? withAccept.reduce((a, b) =>
        (b.acceptRate ?? 0) > (a.acceptRate ?? 0) ? b : a,
      )
    : null;
  return { best, baseline, speedup, bestAccept };
}

/**
 * Rows whose measured runs climb steadily, the ngram-cache signature: the steady state
 * is more than 1.5× the first measured run.
 */
export function rampingRows(rows: AggRow[]): AggRow[] {
  return rows.filter(
    (r) =>
      r.first !== null &&
      r.n >= 3 &&
      r.max > r.first * 1.5 &&
      r.family !== "off",
  );
}

export interface RunPoint {
  seq: number;
  warmup: boolean;
  value: number;
}
export interface RunSeries {
  label: string;
  family: Family;
  points: RunPoint[];
}

/** Every run of every variant in order, for the run-by-run chart. */
export function runSeries(
  results: RunResult[],
  variants: Variant[],
): RunSeries[] {
  const out: RunSeries[] = [];
  for (const v of variants) {
    const list = results
      .filter((r) => r.variant === v.label)
      .sort((a, b) => a.rep - b.rep);
    const points: RunPoint[] = [];
    for (const r of list) {
      const value = r.tps ?? r.clientTps;
      if (value === null || !Number.isFinite(value)) continue;
      points.push({ seq: r.rep + 1, warmup: r.warmup, value });
    }
    if (points.length)
      out.push({ label: v.label, family: familyOf(v.load), points });
  }
  return out;
}

export interface DepthPoint {
  n: number;
  mean: number;
  min: number;
  max: number;
  label: string;
}
export interface DepthSeries {
  family: Family;
  points: DepthPoint[];
}

/** Throughput against draft depth, one line per family. Rows with no depth are left out. */
export function depthSeries(
  rows: AggRow[],
  variants: Variant[],
): DepthSeries[] {
  const byFamily = new Map<Family, DepthPoint[]>();
  for (const r of rows) {
    const n = variants.find((v) => v.label === r.label)?.load.spec_draft_n_max;
    if (r.family === "off" || typeof n !== "number") continue;
    // Tuned ngram rows share a depth with the plain one; the depth line keeps plain rows only.
    if (
      variants.find((v) => v.label === r.label)?.load.llama_extra_args?.length
    )
      continue;
    const list = byFamily.get(r.family) ?? [];
    list.push({ n, mean: r.mean, min: r.min, max: r.max, label: r.label });
    byFamily.set(r.family, list);
  }
  const out: DepthSeries[] = [];
  for (const family of FAMILIES) {
    const points = byFamily.get(family);
    if (points && points.length > 1)
      out.push({ family, points: points.sort((a, b) => a.n - b.n) });
  }
  return out;
}

// --- formatting ----------------------------------------------------------------

export function fmtRate(x: number): string {
  return `${x >= 100 ? x.toFixed(0) : x.toFixed(1)} tok/s`;
}

export function fmtPct(p: number | null): string {
  if (p === null) return "";
  return `${p > 0 ? "+" : ""}${Math.round(p)}%`;
}

export function fmtMs(ms: number | null): string {
  if (ms === null) return "";
  if (ms < 1000) return `${Math.round(ms)} ms`;
  return `${(ms / 1000).toFixed(1)} s`;
}

export function modelShort(path: string | null | undefined): string {
  const s = String(path ?? "").trim();
  const tail = s.split(/[\\/]/).filter(Boolean).pop() ?? s;
  return tail.replace(/-GGUF$/i, "").replace(/\.gguf$/i, "");
}

/** The load fields a sweep changes from row to row. */
export function variedFields(variants: Variant[]): Set<string> {
  const pinned = pinnedFields(variants);
  const out = new Set<string>();
  for (const v of variants)
    for (const k of Object.keys(v.load)) if (!pinned.has(k)) out.add(k);
  // The drafting pair moves together: a mode row resets the depth too.
  if (out.has("speculative_type")) out.add("spec_draft_n_max");
  return out;
}

/** Fields every row sets to the same value (the offload sweep's context), by value. */
export function pinnedFields(variants: Variant[]): Map<string, string> {
  const out = new Map<string, string>();
  if (variants.length < 2) return out;
  const [first, ...rest] = variants;
  for (const [k, v] of Object.entries(first.load)) {
    const same = rest.every(
      (r) =>
        k in r.load &&
        JSON.stringify((r.load as Record<string, unknown>)[k]) === JSON.stringify(v),
    );
    if (same && v !== null && v !== undefined) out.set(k, String(v));
  }
  return out;
}

/** Chat's settings as one footer line; the ones this sweep varies read "varied". */
export function baseLine(
  base: BaseSetting[] | undefined,
  variants: Variant[],
): string {
  if (!base?.length) return "";
  const varied = variedFields(variants);
  const pinned = pinnedFields(variants);
  return base
    .map((b) => {
      if (varied.has(b.field)) return `${b.label} varied`;
      const p = pinned.get(b.field);
      return p === undefined ? `${b.label} ${b.value}` : `${b.label} ${p} (pinned)`;
    })
    .join(" · ");
}

const BACKEND_NAME: Record<string, string> = {
  cuda: "CUDA",
  rocm: "ROCm",
  vulkan: "Vulkan",
  metal: "Metal",
  cpu: "CPU",
  sycl: "SYCL",
};

/** The context rows ran at: a sweep's pinned value, "varied", or what chat had loaded. */
export function ctxNote(run: Pick<BenchRun, "config" | "context">): string {
  if (variedFields(run.config.variants).has("max_seq_length")) return "ctx varied";
  const pinned = pinnedFields(run.config.variants).get("max_seq_length");
  if (pinned) return `ctx ${fmtTokens(Number(pinned))}`;
  return run.context ? `ctx ${fmtTokens(run.context)}` : "";
}

/** Every queued row ran to an end (measured, skipped or failed); false when a Stop cut it short. */
export function ranToEnd(run: Pick<BenchRun, "outcomes">): boolean {
  return run.outcomes.every(
    (o) => o.state === "done" || o.state === "skipped" || o.state === "error",
  );
}

/** A row's name mid-sentence: "1.9× speculation off", but "1.2× Studio auto". */
export function inSentence(label: string): string {
  return label.startsWith("Speculation ") ? label.toLowerCase() : label;
}

/** What ran, where and on which build: the chart's footer and the export's provenance. */
export function footerLines(run: BenchRun): string[] {
  const c = run.config;
  const what = [
    `${modelShort(run.model)}${run.ggufVariant ? ` ${run.ggufVariant}` : ""}`,
    run.kv ? `KV ${run.kv}` : "",
    ctxNote(run),
    `${c.repetitions} measured run${c.repetitions === 1 ? "" : "s"} per setting${c.warmup ? ` (+${c.warmup} warm-up)` : ""}`,
    `${c.maxTokens} max tokens`,
    c.rotatePrompts ? "prompts rotated" : "one prompt repeated",
  ].filter(Boolean);
  const m = run.meta;
  const machine = [
    m.gpu ? `${m.gpu}${m.vramGb ? ` ${Math.round(m.vramGb)} GB` : ""}` : "",
    m.backend ? (BACKEND_NAME[m.backend.toLowerCase()] ?? m.backend) : "",
    m.runtime ?? "",
    osShort(m.os),
    m.cpuThreads ? `${m.cpuThreads} threads` : "",
    m.ramGb ? `${Math.round(m.ramGb)} GB RAM` : "",
  ].filter(Boolean);
  const build = [
    m.llamaTag ? `llama.cpp ${m.llamaTag}` : "",
    m.studioVersion ? `Unsloth Studio ${m.studioVersion}` : "",
    new Date(run.createdAt).toISOString().slice(0, 10),
  ].filter(Boolean);
  return [
    what.join(" · "),
    baseLine(run.base, run.config.variants),
    machine.join(" · "),
    build.join(" · "),
  ].filter(Boolean);
}

/** "Windows-11-10.0.26200" reads as "Windows 11 (10.0.26200)"; Linux keeps its kernel, macOS its version. */
export function osShort(os: string | null | undefined): string {
  if (!os) return "";
  const win = /^Windows-(\d+)-([\d.]+)/.exec(os);
  if (win) return `Windows ${win[1]} (${win[2]})`;
  const mac = /^macOS-([\d.]+)-(\w+)/.exec(os);
  if (mac) return `macOS ${mac[1]} ${mac[2]}`;
  const linux = /^Linux-([\d.]+[^-]*)/.exec(os);
  if (linux) return `Linux ${linux[1]}`;
  return os;
}

export function toMarkdown(run: BenchRun, rows: AggRow[]): string {
  const lines = [
    `### ${SWEEP_TITLE[run.config.sweep]} · ${modelShort(run.model)}`,
    "",
    "| Setting | Throughput | vs baseline | Runs | TTFT | Load | Draft accept |",
    "|---|---|---|---|---|---|---|",
  ];
  for (const r of rows) {
    const spread = r.n > 1 ? ` (${r.min.toFixed(1)}–${r.max.toFixed(1)})` : "";
    const label = r.clientOnly ? `${r.label} *(client-timed)*` : r.label;
    lines.push(
      `| ${label} | ${r.mean.toFixed(1)} tok/s${spread} | ${r.isBaseline ? "baseline" : fmtPct(r.pct)} | ${r.n} | ${fmtMs(r.ttftMs)} | ${fmtMs(r.loadMs)} | ${
        r.acceptRate === null ? "" : `${Math.round(r.acceptRate * 100)}%`
      } |`,
    );
  }
  const skipped = run.outcomes.filter(
    (o) => o.state === "skipped" || o.state === "error",
  );
  if (skipped.length) {
    lines.push(
      "",
      ...skipped.map(
        (o) => `- ${o.label}: ${o.state}${o.reason ? `, ${o.reason}` : ""}`,
      ),
    );
  }
  if (run.base?.length) {
    const varied = variedFields(run.config.variants);
    lines.push(
      "",
      "<details><summary>Settings from chat</summary>",
      "",
      "| Setting | Value |",
      "|---|---|",
      ...run.base.map(
        (b) =>
          `| ${b.label} | ${varied.has(b.field) ? "*varied by this sweep*" : b.value} |`,
      ),
      "",
      "</details>",
    );
  }
  lines.push("", ...footerLines(run).map((l) => `_${l}_`));
  return lines.join("\n");
}

const CSV_COLS: (keyof RunResult)[] = [
  "variant",
  "rep",
  "warmup",
  "promptIndex",
  "tps",
  "promptTps",
  "promptTokens",
  "genTokens",
  "ttftMs",
  "wallMs",
  "clientTps",
  "draftN",
  "draftAccepted",
  "loadMs",
  "at",
];

function csvCell(v: unknown): string {
  if (v === null || v === undefined) return "";
  const s = String(v);
  return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}

export function toCsv(run: BenchRun): string {
  return [
    CSV_COLS.join(","),
    ...run.results.map((r) => CSV_COLS.map((c) => csvCell(r[c])).join(",")),
  ].join("\n");
}
