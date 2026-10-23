// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * One place that knows how to show structured results. Node cards get a one-line summary (no chart code
 * loaded); the inspector and run drawer get the full viewer, loaded on first use. Values of any other kind
 * return null here so callers keep their existing media / text / JSON rendering.
 */

import { type ReactElement, Suspense, lazy } from "react";
import {
  type StructuredKind,
  type StructuredValue,
  structuredKind,
  structuredSummary,
} from "./value-kinds";

const full = () => import("./value-viewers-full");

const VIEWERS: Record<
  StructuredKind,
  React.LazyExoticComponent<
    (props: { value: StructuredValue }) => ReactElement | null
  >
> = {
  table: lazy(() => full().then((m) => ({ default: m.TableViewer }))),
  chart: lazy(() => full().then((m) => ({ default: m.ChartViewer }))),
  heatmap: lazy(() => full().then((m) => ({ default: m.HeatmapViewer }))),
  tensor: lazy(() => full().then((m) => ({ default: m.TensorViewer }))),
};

export function isStructuredValue(value: unknown): value is StructuredValue {
  return structuredKind(value) !== null;
}

/** Full viewer for a structured value, or null when `value` is not one. */
export function StructuredOutput({
  value,
}: { value: unknown }): ReactElement | null {
  const kind = structuredKind(value);
  if (!kind) return null;
  const Viewer = VIEWERS[kind];
  return (
    <Suspense
      fallback={
        <p className="text-[11px] text-zinc-500">
          {structuredSummary(value as StructuredValue)}
        </p>
      }
    >
      <Viewer value={value as StructuredValue} />
    </Suspense>
  );
}

/** Compact badge for node cards. */
export function StructuredBadge({
  value,
}: { value: unknown }): ReactElement | null {
  if (!isStructuredValue(value)) return null;
  return (
    <span className="rounded bg-white/5 px-1.5 py-0.5 text-[11px] text-zinc-300">
      {structuredSummary(value)}
    </span>
  );
}
