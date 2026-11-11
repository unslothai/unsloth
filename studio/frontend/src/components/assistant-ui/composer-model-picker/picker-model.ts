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
  | "external";

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
