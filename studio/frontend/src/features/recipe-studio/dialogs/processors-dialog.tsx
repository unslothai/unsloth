// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Switch } from "@/components/ui/switch";
import { Textarea } from "@/components/ui/textarea";
import { VisuallyHidden } from "radix-ui";
import { type ReactElement, useMemo } from "react";
import type {
  JsonDocumentScoreProcessorConfig,
  RecipeProcessorConfig,
  SchemaTransformProcessorConfig,
} from "../types";
import {
  buildDefaultJsonDocumentScore,
  buildDefaultSchemaTransform,
} from "../utils/processors";
import { AvailableVariables } from "./shared/available-variables";
import { FieldLabel } from "./shared/field-label";

type ProcessorsDialogProps = {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  processors: RecipeProcessorConfig[];
  onProcessorsChange: (processors: RecipeProcessorConfig[]) => void;
  container?: HTMLDivElement | null;
};

export function ProcessorsDialog({
  open,
  onOpenChange,
  processors,
  onProcessorsChange,
  container,
}: ProcessorsDialogProps): ReactElement {
  // Schema Transform
  const schemaIndex = useMemo(
    () =>
      processors.findIndex(
        (processor) => processor.processor_type === "schema_transform",
      ),
    [processors],
  );
  const schemaProcessor =
    schemaIndex >= 0
      ? (processors[schemaIndex] as SchemaTransformProcessorConfig)
      : null;

  // Document Score
  const scoreIndex = useMemo(
    () =>
      processors.findIndex(
        (processor) => processor.processor_type === "json_document_score",
      ),
    [processors],
  );
  const scoreProcessor =
    scoreIndex >= 0
      ? (processors[scoreIndex] as JsonDocumentScoreProcessorConfig)
      : null;

  const setSchemaEnabled = (enabled: boolean) => {
    if (enabled) {
      if (schemaProcessor) return;
      onProcessorsChange([...processors, buildDefaultSchemaTransform()]);
      return;
    }
    onProcessorsChange(
      processors.filter(
        (processor) => processor.processor_type !== "schema_transform",
      ),
    );
  };

  const setScoreEnabled = (enabled: boolean) => {
    if (enabled) {
      if (scoreProcessor) return;
      onProcessorsChange([...processors, buildDefaultJsonDocumentScore()]);
      return;
    }
    onProcessorsChange(
      processors.filter(
        (processor) => processor.processor_type !== "json_document_score",
      ),
    );
  };

  const updateSchema = (patch: Partial<SchemaTransformProcessorConfig>) => {
    if (!schemaProcessor) return;
    const next = [...processors];
    next[schemaIndex] = { ...schemaProcessor, ...patch };
    onProcessorsChange(next);
  };

  const updateScore = (patch: Partial<JsonDocumentScoreProcessorConfig>) => {
    if (!scoreProcessor) return;
    const next = [...processors];
    next[scoreIndex] = { ...scoreProcessor, ...patch };
    onProcessorsChange(next);
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent
        container={container}
        position="absolute"
        overlayPosition="absolute"
        overlayClassName="bg-transparent"
        className="corner-squircle max-h-[650px] overflow-auto sm:max-w-2xl shadow-border"
      >
        <VisuallyHidden.Root>
          <DialogTitle>Processors</DialogTitle>
        </VisuallyHidden.Root>
        <div className="space-y-4">
          {/* Schema Transform ---------------------------------------------- */}
          <div className="flex items-center justify-between gap-3 corner-squircle rounded-2xl border border-border/60 px-3 py-2">
            <div>
              <p className="text-sm font-semibold">Schema transform</p>
              <p className="text-xs text-muted-foreground">
                Transform final rows to target schema (post-batch).
              </p>
            </div>
            <Switch
              checked={Boolean(schemaProcessor)}
              onCheckedChange={setSchemaEnabled}
            />
          </div>

          {schemaProcessor && (
            <div className="space-y-3">
              <AvailableVariables configId="" />
              <div className="grid gap-1.5">
                <FieldLabel
                  label="Name"
                  htmlFor={`${schemaProcessor.id}-name`}
                  hint="Processor name shown in graph and payload."
                />
                <Input
                  id={`${schemaProcessor.id}-name`}
                  className="nodrag"
                  value={schemaProcessor.name}
                  onChange={(event) => updateSchema({ name: event.target.value })}
                />
              </div>
              <div className="grid gap-1.5">
                <FieldLabel
                  label="Template (JSON)"
                  htmlFor={`${schemaProcessor.id}-template`}
                  hint="Target output schema template using Jinja references."
                />
                <Textarea
                  id={`${schemaProcessor.id}-template`}
                  className="corner-squircle nodrag min-h-[220px]"
                  value={schemaProcessor.template}
                  onChange={(event) =>
                    updateSchema({ template: event.target.value })
                  }
                />
                <p className="text-xs text-muted-foreground">
                  Use Jinja refs like {"{{ customer_review }}"} in values.
                </p>
              </div>
            </div>
          )}

          {/* Document Score ----------------------------------------------- */}
          <div className="flex items-center justify-between gap-3 corner-squircle rounded-2xl border border-border/60 px-3 py-2">
            <div>
              <p className="text-sm font-semibold">Document score</p>
              <p className="text-xs text-muted-foreground">
                Score a JSON prediction column against a reference column
                (post-batch). Adds a `score` column to the output dataset.
              </p>
            </div>
            <Switch
              checked={Boolean(scoreProcessor)}
              onCheckedChange={setScoreEnabled}
            />
          </div>

          {scoreProcessor && (
            <div className="space-y-3">
              <div className="grid gap-1.5">
                <FieldLabel
                  label="Name"
                  htmlFor={`${scoreProcessor.id}-name`}
                  hint="Processor name shown in graph and payload."
                />
                <Input
                  id={`${scoreProcessor.id}-name`}
                  className="nodrag"
                  value={scoreProcessor.name}
                  onChange={(event) => updateScore({ name: event.target.value })}
                />
              </div>
              <div className="grid grid-cols-2 gap-3">
                <div className="grid gap-1.5">
                  <FieldLabel
                    label="Prediction column"
                    htmlFor={`${scoreProcessor.id}-pred`}
                    hint="Column with the model's JSON output."
                  />
                  <Input
                    id={`${scoreProcessor.id}-pred`}
                    className="nodrag"
                    value={scoreProcessor.prediction_column}
                    onChange={(event) =>
                      updateScore({ prediction_column: event.target.value })
                    }
                  />
                </div>
                <div className="grid gap-1.5">
                  <FieldLabel
                    label="Reference column"
                    htmlFor={`${scoreProcessor.id}-ref`}
                    hint="Column with the ground-truth JSON."
                  />
                  <Input
                    id={`${scoreProcessor.id}-ref`}
                    className="nodrag"
                    value={scoreProcessor.reference_column}
                    onChange={(event) =>
                      updateScore({ reference_column: event.target.value })
                    }
                  />
                </div>
              </div>
              <div className="grid gap-1.5">
                <FieldLabel
                  label="Schema (optional, JSON)"
                  htmlFor={`${scoreProcessor.id}-schema`}
                  hint="JSON Schema or studio field→comparator map. Leave empty to apply the default comparator to every leaf."
                />
                <Textarea
                  id={`${scoreProcessor.id}-schema`}
                  className="corner-squircle nodrag min-h-[140px]"
                  value={scoreProcessor.schema}
                  onChange={(event) =>
                    updateScore({ schema: event.target.value })
                  }
                />
              </div>
              <div className="grid grid-cols-3 gap-3">
                <div className="grid gap-1.5">
                  <FieldLabel
                    label="Default comparator"
                    htmlFor={`${scoreProcessor.id}-cmp`}
                    hint="Fallback comparator when no schema is given."
                  />
                  <Input
                    id={`${scoreProcessor.id}-cmp`}
                    className="nodrag"
                    value={scoreProcessor.default_comparator}
                    onChange={(event) =>
                      updateScore({ default_comparator: event.target.value })
                    }
                  />
                </div>
                <div className="grid gap-1.5">
                  <FieldLabel
                    label="Score column"
                    htmlFor={`${scoreProcessor.id}-score`}
                    hint="Output column name for the per-row float score."
                  />
                  <Input
                    id={`${scoreProcessor.id}-score`}
                    className="nodrag"
                    value={scoreProcessor.score_column}
                    onChange={(event) =>
                      updateScore({ score_column: event.target.value })
                    }
                  />
                </div>
                <div className="grid gap-1.5">
                  <FieldLabel
                    label="Breakdown column (optional)"
                    htmlFor={`${scoreProcessor.id}-bd`}
                    hint="Empty = no breakdown. If set, writes per-field score JSON to this column."
                  />
                  <Input
                    id={`${scoreProcessor.id}-bd`}
                    className="nodrag"
                    value={scoreProcessor.breakdown_column}
                    onChange={(event) =>
                      updateScore({ breakdown_column: event.target.value })
                    }
                  />
                </div>
              </div>
            </div>
          )}
        </div>
        <DialogFooter>
          <Button
            type="button"
            variant="outline"
            onClick={() => onOpenChange(false)}
          >
            Done
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
