// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// No path aliases here, so node:test can import it directly.

export const FAVORITES_TAB = "favorites";
export const RECOMMENDED_TAB = "recommended";
export const DEVICE_TAB = "device";
export const CONNECTED_TAB = "connected";
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

export type DeviceSort = "recent" | "size" | "name" | "downloaded";
export const DEVICE_SORT_LABEL: Record<DeviceSort, string> = {
  recent: "Recent",
  size: "Size",
  name: "Name",
  downloaded: "Downloaded",
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
): T[] {
  const byDate = (a: T, b: T) =>
    (b.updatedAt ?? -1) - (a.updatedAt ?? -1) || a.label.localeCompare(b.label);
  return [...rows].sort((a, b) => {
    if (key === "name") {
      return a.label.localeCompare(b.label);
    }
    if (key === "size") {
      return (
        (b.sizeBytes ?? -1) - (a.sizeBytes ?? -1) ||
        a.label.localeCompare(b.label)
      );
    }
    if (key === "recent") {
      return lastLoaded(b) - lastLoaded(a) || byDate(a, b);
    }
    return byDate(a, b);
  });
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
