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

type UniformDialogProps = {
  config: SamplerConfig;
  onUpdate: (patch: Partial<SamplerConfig>) => void;
};

export function UniformDialog({
  config,
  onUpdate,
}: UniformDialogProps): ReactElement {
  const lowId = `${config.id}-uniform-low`;
  const highId = `${config.id}-uniform-high`;
  const convertId = `${config.id}-uniform-convert`;
  return (
    <div className="space-y-4">
      <NameField
        value={config.name}
        onChange={(value) => onUpdate({ name: value })}
      />
      <div className="grid gap-3 sm:grid-cols-2">
        <div className="grid gap-1.5">
          <FieldLabel
            label="Low"
            htmlFor={lowId}
            hint="Minimum sampled value."
          />
          <Input
            id={lowId}
            type="number"
            className="nodrag"
            value={config.low ?? ""}
            onChange={(event) => onUpdate({ low: event.target.value })}
          />
        </div>
        <div className="grid gap-1.5">
          <FieldLabel
            label="High"
            htmlFor={highId}
            hint="Maximum sampled value."
          />
          <Input
            id={highId}
            type="number"
            className="nodrag"
            value={config.high ?? ""}
            onChange={(event) => onUpdate({ high: event.target.value })}
          />
        </div>
      </div>
      <div className="grid gap-1.5">
        <FieldLabel
          label="Convert to"
          htmlFor={convertId}
          hint="Optionally cast sampled values before output."
        />
        <Select
          value={config.convert_to ?? "none"}
          onValueChange={(value) =>
            onUpdate({
              // biome-ignore lint/style/useNamingConvention: api schema
              convert_to: value === "none" ? undefined : (value as "int" | "float" | "str"),
            })
          }
        >
          <SelectTrigger className="nodrag w-full" id={convertId}>
            <SelectValue placeholder="No conversion" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="none">None</SelectItem>
            <SelectItem value="int">int</SelectItem>
            <SelectItem value="float">float</SelectItem>
            <SelectItem value="str">str</SelectItem>
          </SelectContent>
        </Select>
      </div>
    </div>
  );
}
