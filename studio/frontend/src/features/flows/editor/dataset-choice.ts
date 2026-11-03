// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** A Hugging Face Dataset node's params as the Train page's picker reads them; empty means unset. */
export function hfDatasetChoice(params: Record<string, unknown>): {
  repo: string | null;
  subset: string | null;
  split: string | null;
} {
  return {
    repo: String(params.repo ?? "").trim() || null,
    subset: String(params.subset ?? "") || null,
    // An unset split runs as "train", the param's default.
    split: String(params.split ?? "train") || null,
  };
}

/** The split the dropdown shows: a saved expression like `train[:10%]` or `train+test` shows its first split. */
export function shownSplit(split: string | null): string | null {
  return split?.match(/^[\w.-]+/)?.[0] ?? split;
}

/** What a split pick saves: the dropdown selecting the split it already shows keeps the saved expression. */
export function pickedSplit(saved: string | null, picked: string | null): string {
  return picked !== null && saved && picked === shownSplit(saved) ? saved : (picked ?? "");
}
