// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { CachedInventoryRow, LocalInventoryRow } from "../types";

export interface ModelAnnotation {
  display_name: string;
  labels: string[];
  notes: string;
}

export type ModelAnnotations = Record<string, ModelAnnotation>;

/** Add view-only details by stable inventory id, preserving every action and capability field. */
export function annotateInventoryRows<
  T extends CachedInventoryRow | LocalInventoryRow,
>(rows: T[], annotations: ModelAnnotations): T[] {
  return rows.map((row) =>
    annotations[row.id] ? { ...row, annotation: annotations[row.id] } : row,
  );
}
