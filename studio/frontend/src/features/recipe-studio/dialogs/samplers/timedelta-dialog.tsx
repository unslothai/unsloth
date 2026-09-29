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
import type { ReactElement } from "react";
import type { SamplerConfig } from "../../types";
import { NameField } from "../shared/name-field";
import { FieldLabel } from "../shared/field-label";

const TIMEDELTA_UNITS: Array<"D" | "h" | "m" | "s"> = ["D", "h", "m", "s"];
const NONE_VALUE = "__none";

type TimedeltaDialogProps = {
  config: SamplerConfig;
  datetimeOptions: string[];
  onUpdate: (patch: Partial<SamplerConfig>) => void;
};

export function TimedeltaDialog({
  config,
  datetimeOptions,
  onUpdate,
}: TimedeltaDialogProps): ReactElement {
  const dtMinId = `${config.id}-timedelta-min`;
  const dtMaxId = `${config.id}-timedelta-max`;
  const unitId = `${config.id}-timedelta-unit`;
  const referenceId = `${config.id}-timedelta-reference`;
  const updateField = <K extends keyof SamplerConfig>(
    key: K,
    value: SamplerConfig[K],
  ) => {
    onUpdate({ [key]: value } as Partial<SamplerConfig>);
  };
  return (
    <div className="space-y-4">
      <NameField
        value={config.name}
        onChange={(value) => onUpdate({ name: value })}
      />
      <div className="grid gap-3 sm:grid-cols-2">
        <div className="grid gap-1.5">
          <FieldLabel
            label="dt_min"
            htmlFor={dtMinId}
            hint="Minimum offset from reference datetime."
          />
          <Input
            id={dtMinId}
            type="number"
            className="nodrag"
            value={config.dt_min ?? ""}
            onChange={(event) => updateField("dt_min", event.target.value)}
          />
        </div>
        <div className="grid gap-1.5">
          <FieldLabel
            label="dt_max"
            htmlFor={dtMaxId}
            hint="Maximum offset from reference datetime."
          />
          <Input
            id={dtMaxId}
            type="number"
            className="nodrag"
            value={config.dt_max ?? ""}
            onChange={(event) => updateField("dt_max", event.target.value)}
          />
        </div>
      </div>
      <div className="grid gap-1.5">
        <FieldLabel
          label="Unit"
          htmlFor={unitId}
          hint="Offset unit. D/h/m/s."
        />
        <Select
          value={config.timedelta_unit ?? "D"}
          onValueChange={(value) =>
            updateField("timedelta_unit", value as "D" | "h" | "m" | "s")
          }
        >
          <SelectTrigger className="nodrag w-full" id={unitId}>
            <SelectValue placeholder="Select unit" />
          </SelectTrigger>
          <SelectContent>
            {TIMEDELTA_UNITS.map((unit) => (
              <SelectItem key={unit} value={unit}>
                {unit}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>
      <div className="grid gap-1.5">
        <FieldLabel
          label="Reference datetime column"
          htmlFor={referenceId}
          hint="Datetime column used as anchor before offset."
        />
        <Select
          value={config.reference_column_name?.trim() || NONE_VALUE}
          onValueChange={(value) =>
            updateField("reference_column_name", value === NONE_VALUE ? "" : value)
          }
        >
          <SelectTrigger className="nodrag w-full" id={referenceId}>
            <SelectValue placeholder="Select datetime column" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value={NONE_VALUE}>None</SelectItem>
            {datetimeOptions.map((name) => (
              <SelectItem key={name} value={name}>
                {name}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>
    </div>
  );
}
