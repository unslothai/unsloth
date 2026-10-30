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
