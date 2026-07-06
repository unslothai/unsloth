// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import { type ReactElement, useMemo } from "react";
import { useRecipeStudioStore } from "../stores/recipe-studio";
import type { EvaluationDocumentScoreConfig, NodeConfig } from "../types";
import { FieldLabel } from "./shared/field-label";

type EvaluationDocumentScoreDialogProps = {
  config: EvaluationDocumentScoreConfig;
  onUpdate: (patch: Partial<EvaluationDocumentScoreConfig>) => void;
};

function columnsFromConfig(config: NodeConfig): string[] {
  if (config.kind === "seed") {
    return (config.seed_columns ?? [])
      .map((column) => column.trim())
      .filter(Boolean);
  }
  if (
    config.kind === "llm" ||
    config.kind === "sampler" ||
    config.kind === "expression"
  ) {
    const name = config.name?.trim();
    return name ? [name] : [];
  }
  return [];
}

export function EvaluationDocumentScoreDialog({
  config,
  onUpdate,
}: EvaluationDocumentScoreDialogProps): ReactElement {
  const edges = useRecipeStudioStore((state) => state.edges);
  const configs = useRecipeStudioStore((state) => state.configs);

  // BFS over edges connected to this Score block. Walks both directions
  // because the visible handles and the hidden duplicate handles share the
  // same node side, so React Flow can store edges with either source/target
  // assignment depending on which handle the user grabbed during the drag.
  // Treating both endpoints as connected is forgiving — the worst case is
  // an extra unused column in the dropdown.
  const upstreamColumns = useMemo(() => {
    const visited = new Set<string>([config.id]);
    const queue: string[] = [config.id];
    const columns = new Set<string>();
    while (queue.length > 0) {
      const current = queue.shift();
      if (!current) {
        continue;
      }
      for (const edge of edges) {
        let neighbor: string | null = null;
        if (edge.source === current) {
          neighbor = edge.target;
        } else if (edge.target === current) {
          neighbor = edge.source;
        }
        if (!neighbor || visited.has(neighbor)) {
          continue;
        }
        visited.add(neighbor);
        const neighborConfig = configs[neighbor];
        if (!neighborConfig) {
          continue;
        }
        // Don't traverse outward from another evaluation block — they're
        // sinks, not producers, and a chain of evaluations would just leak
        // their score columns into siblings.
        if (neighborConfig.kind !== "evaluation") {
          queue.push(neighbor);
        }
        for (const column of columnsFromConfig(neighborConfig)) {
          columns.add(column);
        }
      }
    }
    return Array.from(columns).sort((a, b) => a.localeCompare(b));
  }, [edges, configs, config.id]);

  const predictionOptions = useMemo(() => {
    const set = new Set(upstreamColumns);
    if (config.prediction_column) {
      set.add(config.prediction_column);
    }
    return Array.from(set).sort((a, b) => a.localeCompare(b));
  }, [upstreamColumns, config.prediction_column]);

  const referenceOptions = useMemo(() => {
    const set = new Set(upstreamColumns);
    if (config.reference_column) {
      set.add(config.reference_column);
    }
    return Array.from(set).sort((a, b) => a.localeCompare(b));
  }, [upstreamColumns, config.reference_column]);

  const hasUpstream = upstreamColumns.length > 0;

  return (
    <div className="space-y-3">
      <div className="grid gap-1.5">
        <FieldLabel
          label="Name"
          htmlFor={`${config.id}-name`}
          hint="Step name shown in graph and payload."
        />
        <Input
          id={`${config.id}-name`}
          className="nodrag"
          value={config.name}
          onChange={(event) => onUpdate({ name: event.target.value })}
        />
      </div>
      {!hasUpstream && (
        <div className="corner-squircle rounded-xl border border-amber-400/60 bg-amber-50/60 px-3 py-2 text-xs text-amber-900 dark:bg-amber-950/40 dark:text-amber-200">
          No upstream blocks connected. Draw an edge from a Source data block
          and/or an AI step into this Document score block — its output columns
          will then appear in the dropdowns below.
        </div>
      )}
      <div className="grid grid-cols-2 gap-3">
        <div className="grid gap-1.5">
          <FieldLabel
            label="Prediction column"
            htmlFor={`${config.id}-pred`}
            hint="Column with the model's JSON output."
          />
          <Select
            value={config.prediction_column || undefined}
            onValueChange={(value) => onUpdate({ prediction_column: value })}
          >
            <SelectTrigger
              id={`${config.id}-pred`}
              className="nodrag"
              disabled={predictionOptions.length === 0}
            >
              <SelectValue
                placeholder={
                  predictionOptions.length === 0
                    ? "Connect an upstream block"
                    : "Select a column"
                }
              />
            </SelectTrigger>
            <SelectContent>
              {predictionOptions.map((column) => (
                <SelectItem key={column} value={column}>
                  {column}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
        <div className="grid gap-1.5">
          <FieldLabel
            label="Reference column"
            htmlFor={`${config.id}-ref`}
            hint="Column with the ground-truth JSON."
          />
          <Select
            value={config.reference_column || undefined}
            onValueChange={(value) => onUpdate({ reference_column: value })}
          >
            <SelectTrigger
              id={`${config.id}-ref`}
              className="nodrag"
              disabled={referenceOptions.length === 0}
            >
              <SelectValue
                placeholder={
                  referenceOptions.length === 0
                    ? "Connect an upstream block"
                    : "Select a column"
                }
              />
            </SelectTrigger>
            <SelectContent>
              {referenceOptions.map((column) => (
                <SelectItem key={column} value={column}>
                  {column}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      </div>
      <div className="grid gap-1.5">
        <FieldLabel
          label="Schema (optional, JSON)"
          htmlFor={`${config.id}-schema`}
          hint="Field→comparator map. Leave empty to apply the default comparator to every leaf."
        />
        <Textarea
          id={`${config.id}-schema`}
          className="corner-squircle nodrag min-h-[140px]"
          value={config.schema}
          onChange={(event) => onUpdate({ schema: event.target.value })}
        />
      </div>
      <div className="grid grid-cols-3 gap-3">
        <div className="grid gap-1.5">
          <FieldLabel
            label="Default comparator"
            htmlFor={`${config.id}-cmp`}
            hint="Fallback comparator when no schema is given."
          />
          <Select
            value={config.default_comparator || "string"}
            onValueChange={(value) => onUpdate({ default_comparator: value })}
          >
            <SelectTrigger id={`${config.id}-cmp`} className="nodrag">
              <SelectValue placeholder="Select a comparator" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="string">string</SelectItem>
              <SelectItem value="categorical">categorical</SelectItem>
              <SelectItem value="numeric">numeric</SelectItem>
              <SelectItem value="money">money</SelectItem>
              <SelectItem value="date">date</SelectItem>
            </SelectContent>
          </Select>
        </div>
        <div className="grid gap-1.5">
          <FieldLabel
            label="Score column"
            htmlFor={`${config.id}-score`}
            hint="Output column name for the per-row float score."
          />
          <Input
            id={`${config.id}-score`}
            className="nodrag"
            value={config.score_column}
            onChange={(event) => onUpdate({ score_column: event.target.value })}
          />
        </div>
        <div className="grid gap-1.5">
          <FieldLabel
            label="Breakdown column (optional)"
            htmlFor={`${config.id}-bd`}
            hint="Empty = no breakdown. If set, writes per-field score JSON to this column."
          />
          <Input
            id={`${config.id}-bd`}
            className="nodrag"
            value={config.breakdown_column}
            onChange={(event) =>
              onUpdate({ breakdown_column: event.target.value })
            }
          />
        </div>
      </div>
    </div>
  );
}
