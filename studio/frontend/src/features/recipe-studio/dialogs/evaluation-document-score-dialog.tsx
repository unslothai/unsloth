// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import type { ReactElement } from "react";
import type { EvaluationDocumentScoreConfig } from "../types";
import { FieldLabel } from "./shared/field-label";

type EvaluationDocumentScoreDialogProps = {
  config: EvaluationDocumentScoreConfig;
  onUpdate: (patch: Partial<EvaluationDocumentScoreConfig>) => void;
};

export function EvaluationDocumentScoreDialog({
  config,
  onUpdate,
}: EvaluationDocumentScoreDialogProps): ReactElement {
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
      <div className="grid grid-cols-2 gap-3">
        <div className="grid gap-1.5">
          <FieldLabel
            label="Prediction column"
            htmlFor={`${config.id}-pred`}
            hint="Column with the model's JSON output."
          />
          <Input
            id={`${config.id}-pred`}
            className="nodrag"
            value={config.prediction_column}
            onChange={(event) =>
              onUpdate({ prediction_column: event.target.value })
            }
          />
        </div>
        <div className="grid gap-1.5">
          <FieldLabel
            label="Reference column"
            htmlFor={`${config.id}-ref`}
            hint="Column with the ground-truth JSON."
          />
          <Input
            id={`${config.id}-ref`}
            className="nodrag"
            value={config.reference_column}
            onChange={(event) =>
              onUpdate({ reference_column: event.target.value })
            }
          />
        </div>
      </div>
      <div className="grid gap-1.5">
        <FieldLabel
          label="Schema (optional, JSON)"
          htmlFor={`${config.id}-schema`}
          hint="JSON Schema or studio field→comparator map. Leave empty to apply the default comparator to every leaf."
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
          <Input
            id={`${config.id}-cmp`}
            className="nodrag"
            value={config.default_comparator}
            onChange={(event) =>
              onUpdate({ default_comparator: event.target.value })
            }
          />
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
