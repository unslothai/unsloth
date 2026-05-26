// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { cn } from "@/lib/utils";
import type { ScoreNode } from "../api/eval-api";

export function scoreColor(score: number): string {
  if (score >= 0.999) return "text-emerald-500";
  if (score >= 0.5) return "text-amber-500";
  return "text-red-500";
}

/** Recursive per-field breakdown row for a json_document ScoreNode. */
export function BreakdownTree({
  label,
  node,
  depth,
}: {
  label: string;
  node: ScoreNode;
  depth: number;
}) {
  const children = node.children;
  const entries: [string, ScoreNode][] = Array.isArray(children)
    ? children.map((c, i) => [`[${i}]`, c])
    : children
      ? Object.entries(children)
      : [];

  return (
    <div className="text-sm">
      <div
        className="flex items-center justify-between gap-3 py-1"
        style={{ paddingLeft: `${depth * 16}px` }}
      >
        <span className="truncate font-medium text-muted-foreground">
          {label}
          {node.note ? (
            <span className="ml-2 text-xs text-red-500/80">({node.note})</span>
          ) : null}
        </span>
        <span className={cn("tabular-nums font-semibold", scoreColor(node.score))}>
          {node.score.toFixed(3)}
        </span>
      </div>
      {entries.map(([k, child]) => (
        <BreakdownTree key={k} label={k} node={child} depth={depth + 1} />
      ))}
    </div>
  );
}
