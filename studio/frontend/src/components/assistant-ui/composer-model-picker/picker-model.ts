// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// No path aliases here, so node:test can import it directly.

export const FAVORITES_TAB = "favorites";
export const RECOMMENDED_TAB = "recommended";
export const DEVICE_TAB = "device";
export const CONNECTED_TAB = "connected";
/** One-click setups offered on an empty Connected tab, beside Add connection. */
export const DEFAULT_PROVIDERS = [
  { type: "openai", label: "Connect ChatGPT", connect: "openai_codex" },
  { type: "anthropic", label: "Connect Anthropic", connect: "anthropic" },
] as const;
export const OFF_LEVEL = "off";

export type RowKind =
  | "gguf"
  | "cached"
  | "local"
  | "localGguf"
  | "loaded"
  | "hub"
  | "npu"
  | "external";

export const DEVICE_SECTIONS = [
  "unsloth",
  "other",
  "npu",
  "fineTuned",
  "custom",
  "lmstudio",
  "modelsDir",
] as const;
export type DeviceSection = (typeof DEVICE_SECTIONS)[number];
export type ListSection = DeviceSection | "recommended";
export const SECTION_LABEL: Record<ListSection, string> = {
  recommended: "Recommended",
  unsloth: "Unsloth",
  other: "Other models",
  npu: "NPU",
  fineTuned: "Trained",
  custom: "Custom folders",
  lmstudio: "LM Studio",
  modelsDir: "Local models",
};

export function repoSection(id: string): DeviceSection {
  return id.trim().toLowerCase().startsWith("unsloth/") ? "unsloth" : "other";
}

/** Local rows carry their store as `baseModel` ("LM Studio", "Local models", ...). */
export function localSection(
  source: string | undefined,
  baseModel: string | undefined,
): DeviceSection {
  if (source === "training" || source === "exported") {
    return "fineTuned";
  }
  if (baseModel === "LM Studio") {
    return "lmstudio";
  }
  if (baseModel === "Local models") {
    return "modelsDir";
  }
  return "custom";
}

export function orderBySection<T extends { section?: ListSection }>(
  rows: readonly T[],
): T[] {
  const order: readonly ListSection[] = DEVICE_SECTIONS;
  const rank = (row: T) => {
    const at = row.section ? order.indexOf(row.section) : -1;
    return at >= 0 ? at : order.length;
  };
  return rows
    .map((row, i) => ({ row, i }))
    .sort((a, b) => rank(a.row) - rank(b.row) || a.i - b.i)
    .map(({ row }) => row);
}

/** Custom-folder rows count too: a search that finds only one must not also say "No models found". */
export function sectionsHaveRows(
  sections: readonly ListSection[],
  rows: readonly { section?: ListSection }[],
): boolean {
  return rows.some((row) => row.section && sections.includes(row.section));
}

export type FormatTone = "gguf" | "mlx" | "checkpoint" | "adapter";

export interface PickerRow {
  key: string;
  id: string;
  label: string;
  tab: string;
  kind: RowKind;
  format?: FormatTone;
  detail?: string;
  providerType?: string;
  providerName?: string;
  aliases?: readonly string[];
  loaded?: boolean;
  onDevice?: boolean;
  groups?: string[];
  section?: ListSection;
  sizeBytes?: number;
  /** In the listing's own units. */
  updatedAt?: number;
}

export type DeviceSort = "recent" | "size" | "name";
export const DEVICE_SORT_LABEL: Record<DeviceSort, string> = {
  recent: "Recent",
  size: "Size",
  name: "Name",
};
export const DEVICE_SORT_ORDER: Record<
  DeviceSort,
  { desc: string; asc: string }
> = {
  recent: { desc: "Newest first", asc: "Oldest first" },
  size: { desc: "Largest first", asc: "Smallest first" },
  name: { desc: "Z to A", asc: "A to Z" },
};
/** Each key's first direction: newest and largest first, names A to Z. */
export const DEVICE_SORT_DESC: Record<DeviceSort, boolean> = {
  recent: true,
  size: true,
  name: false,
};
export type FormatFilter = "all" | "gguf" | "mlx" | "safetensors" | "npu";
export const FORMAT_FILTER_LABEL: Record<FormatFilter, string> = {
  all: "All",
  gguf: "GGUF",
  mlx: "MLX",
  safetensors: "Safetensors",
  npu: "NPU",
};

export function matchesFormat(
  row: Pick<PickerRow, "format" | "kind">,
  filter: FormatFilter,
): boolean {
  switch (filter) {
    case "all":
      return true;
    case "npu":
      return row.kind === "npu";
    case "gguf":
      return row.format === "gguf";
    case "mlx":
      return row.format === "mlx";
    case "safetensors":
      return row.format === "checkpoint" || row.format === "adapter";
  }
}

export function sortDeviceRows<T extends PickerRow>(
  rows: readonly T[],
  key: DeviceSort,
  lastLoaded: (row: T) => number,
  desc: boolean = DEVICE_SORT_DESC[key],
): T[] {
  const byDate = (a: T, b: T) =>
    (b.updatedAt ?? -1) - (a.updatedAt ?? -1) || a.label.localeCompare(b.label);
  const sorted = [...rows].sort((a, b) => {
    if (key === "name") {
      return a.label.localeCompare(b.label);
    }
    if (key === "size") {
      return (
        (b.sizeBytes ?? -1) - (a.sizeBytes ?? -1) ||
        a.label.localeCompare(b.label)
      );
    }
    return lastLoaded(b) - lastLoaded(a) || byDate(a, b);
  });
  return desc === DEVICE_SORT_DESC[key] ? sorted : sorted.reverse();
}

/** A model id's first version run (`gpt-5.6-sol` -> [5, 6], `claude-opus-4-7` -> [4, 7]). */
export function modelVersion(id: string): number[] {
  const run = /\d+(?:[.-]\d+)*/.exec(id.replace(/^.*\//, ""));
  return run ? run[0].split(/[.-]/).map(Number) : [];
}

/** Each connection's favorites first, then newest version first; ties keep the provider's order. */
export function sortConnectedRows<T extends PickerRow>(
  rows: readonly T[],
  isFavorite: (row: T) => boolean = () => false,
): T[] {
  const at = new Map(rows.map((row, index) => [row, index]));
  const newer = (a: T, b: T) => {
    const va = modelVersion(a.label);
    const vb = modelVersion(b.label);
    for (let i = 0; i < Math.max(va.length, vb.length); i++) {
      const diff = (vb[i] ?? -1) - (va[i] ?? -1);
      if (diff !== 0) {
        return diff;
      }
    }
    return 0;
  };
  return [...rows].sort(
    (a, b) =>
      (a.providerName ?? "").localeCompare(b.providerName ?? "") ||
      a.tab.localeCompare(b.tab) ||
      Number(isFavorite(b)) - Number(isFavorite(a)) ||
      newer(a, b) ||
      (at.get(a) ?? 0) - (at.get(b) ?? 0),
  );
}

const norm = (value: string | null | undefined): string =>
  (value ?? "").trim().toLowerCase();

export function rowMatchesCheckpoint(
  row: Pick<PickerRow, "id" | "aliases">,
  checkpoint: string | null | undefined,
): boolean {
  const target = norm(checkpoint);
  if (!target) {
    return false;
  }
  if (norm(row.id) === target) {
    return true;
  }
  return (row.aliases ?? []).some((alias) => norm(alias) === target);
}

export function findChosenRow<T extends PickerRow>(
  rows: readonly T[],
  checkpoint: string | null | undefined,
): T | undefined {
  return (
    rows.find(
      (row) => row.kind !== "hub" && rowMatchesCheckpoint(row, checkpoint),
    ) ?? rows.find((row) => rowMatchesCheckpoint(row, checkpoint))
  );
}

export function filterRows<T extends PickerRow>(
  rows: readonly T[],
  tab: string,
  query: string,
  isFavorite: (row: T) => boolean,
  /** Rows a Hub search already matched (it expands maker aliases), kept without a text match. */
  matched?: (row: T) => boolean,
): T[] {
  const q = norm(query);
  const seen = new Set<string>();
  return rows.filter((row) => {
    if (tab === FAVORITES_TAB) {
      if (!isFavorite(row) || seen.has(norm(row.id))) {
        return false;
      }
      seen.add(norm(row.id));
    } else if (row.tab !== tab && !row.groups?.includes(tab)) {
      return false;
    }
    return (
      !q ||
      matched?.(row) === true ||
      norm(row.label).includes(q) ||
      norm(row.id).includes(q) ||
      norm(row.providerName).includes(q)
    );
  });
}

export function dedupeRows<T extends PickerRow>(rows: readonly T[]): T[] {
  const seen = new Set<string>();
  const out: T[] = [];
  for (const row of rows) {
    const key = norm(row.id);
    if (seen.has(key)) {
      continue;
    }
    seen.add(key);
    out.push(row);
  }
  return out;
}

export function pinCoversId(pinned: readonly string[], id: string): boolean {
  const base = norm(id);
  return pinned.some((key) => {
    const k = norm(key);
    return k === base || k.startsWith(`${base}::`);
  });
}

const CHAT_TASKS = new Set([
  "text-generation",
  "text2text-generation",
  "conversational",
  "image-text-to-text",
  "image-to-text",
  "visual-question-answering",
  "video-text-to-text",
  "any-to-any",
  "audio-text-to-text",
]);

export function isChatTask(task: string | null | undefined): boolean {
  const tag = norm(task);
  return tag === "" || CHAT_TASKS.has(tag);
}

export interface QuantChoice {
  quant: string;
  downloaded?: boolean;
  partial?: boolean;
}

export function downloadedQuants<T extends QuantChoice>(
  variants: readonly T[],
): T[] {
  return variants.filter((v) => v.downloaded === true && v.partial !== true);
}

export function chooseQuant<T extends QuantChoice>(
  candidates: readonly T[],
  prefer: readonly (string | null | undefined)[],
): T | undefined {
  for (const want of prefer) {
    const hit = want
      ? candidates.find((v) => norm(v.quant) === norm(want))
      : undefined;
    if (hit) {
      return hit;
    }
  }
  return candidates[0];
}

export interface SliderLevel {
  value: string;
  label: string;
}

export function effortLabel(level: string, modelId?: string | null): string {
  if (level === OFF_LEVEL) {
    return "Off";
  }
  if (level !== "xhigh") {
    return level.charAt(0).toUpperCase() + level.slice(1);
  }
  const id = norm(modelId);
  return id.startsWith("claude-opus-4-6") || id.startsWith("claude-sonnet-4-6")
    ? "Max"
    : "Extra High";
}

export function sliderLevels(
  levels: readonly string[],
  supportsOff: boolean,
  modelId?: string | null,
): SliderLevel[] {
  const out: SliderLevel[] = supportsOff
    ? [{ value: OFF_LEVEL, label: effortLabel(OFF_LEVEL) }]
    : [];
  for (const level of levels) {
    if (level === "none" && supportsOff) {
      continue;
    }
    out.push({ value: level, label: effortLabel(level, modelId) });
  }
  return out;
}

export function sliderValue(
  enabled: boolean,
  effort: string,
  supportsOff: boolean,
): string {
  if (supportsOff && (!enabled || effort === "none")) {
    return OFF_LEVEL;
  }
  return effort;
}

// Judged once per opening from the trigger, never the panel's height: a new chat opens down while
// 180px fit there; otherwise down only when above is short of 180px and below has more room.
const PICKER_MIN_ROOM = 180;
/** The panel's gap to the window edge, which the side rule's room already leaves. */
export const PICKER_EDGE = 8;
export function pickerSide(
  preferred: "top" | "bottom",
  trigger: { top: number; bottom: number } | null,
  viewportHeight: number,
): "top" | "bottom" {
  if (!trigger) {
    return preferred;
  }
  const above = trigger.top - PICKER_EDGE;
  const below = viewportHeight - trigger.bottom - PICKER_EDGE;
  if (preferred === "bottom") {
    return below >= PICKER_MIN_ROOM ? "bottom" : "top";
  }
  return above < PICKER_MIN_ROOM && below > above ? "bottom" : "top";
}

export const PICKER_GAP = 8;
/** Short of room below, a downward panel rides up over the composer, never past the window top. */
export function pickerDropOffset(
  triggerBottom: number,
  panelHeight: number,
  viewportHeight: number,
): number {
  const fits = viewportHeight - PICKER_EDGE - panelHeight - triggerBottom;
  return Math.max(PICKER_EDGE - triggerBottom, Math.min(PICKER_GAP, fits));
}

/** Slides a fixed-side panel back inside a narrow window, since Radix's own shift is off. */
export const PICKER_WIDTH = 420;
export function pickerAlignOffset(
  align: "start" | "end",
  trigger: { left: number; right: number } | null,
  panelWidth: number,
  viewportWidth: number,
): number {
  if (!trigger) {
    return 0;
  }
  // PopoverContent caps its width at the window less 32px.
  const width = Math.min(panelWidth, viewportWidth - 32);
  const left = align === "end" ? trigger.right - width : trigger.left;
  const clamped = Math.max(
    PICKER_EDGE,
    Math.min(left, viewportWidth - PICKER_EDGE - width),
  );
  return align === "end" ? left - clamped : clamped - left;
}

// llama.cpp's stories260K test fixtures, which Discover's all-uploader search surfaces.
const TEST_FIXTURE = /stories\d+k/i;
export function isTestFixtureRepo(id: string): boolean {
  return TEST_FIXTURE.test(id);
}

/** Discover falls back to the chat list only before a search has results; else rows the Hub matched
 * without a text match would show unrelated models. */
export function discoverUsesFallback(
  hubQuery: string,
  hubResultCount: number,
): boolean {
  return hubQuery === "" || hubResultCount === 0;
}

/** Reads `items` with at most `workers` in flight; stops taking new ones once `live()` is false. */
export async function readBounded<T, R>(
  items: T[],
  workers: number,
  read: (item: T) => Promise<R>,
  onResult: (result: R) => void,
  live: () => boolean,
): Promise<void> {
  const worker = async () => {
    for (
      let item = items.shift();
      item !== undefined && live();
      item = items.shift()
    ) {
      onResult(await read(item));
    }
  };
  await Promise.all(
    Array.from({ length: Math.min(workers, items.length) }, worker),
  );
}

/** The chosen model goes to the host's guarded unload (no id), as the header picker's Eject did. */
export function ejectTargetId(
  row: { id: string; loadedId?: string },
  chosenId: string | undefined,
): string | undefined {
  return row.id === chosenId ? undefined : (row.loadedId ?? row.id);
}
