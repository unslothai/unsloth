// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Input } from "@/components/ui/input";
import type { ReactElement } from "react";
import type { SamplerConfig } from "../../types";
import { NameField } from "../shared/name-field";
import { FieldLabel } from "../shared/field-label";

type BernoulliDialogProps = {
  config: SamplerConfig;
  onUpdate: (patch: Partial<SamplerConfig>) => void;
};

export function BernoulliDialog({
  config,
  onUpdate,
}: BernoulliDialogProps): ReactElement {
  const pId = `${config.id}-bernoulli-p`;
  return (
    <div className="space-y-4">
      <NameField
        value={config.name}
        onChange={(value) => onUpdate({ name: value })}
      />
      <div className="grid gap-1.5">
        <FieldLabel
          label="Probability (p)"
          htmlFor={pId}
          hint="Success probability in [0, 1]."
        />
        <Input
          id={pId}
          type="number"
          min="0"
          max="1"
          step="0.01"
          className="nodrag"
          value={config.p ?? ""}
          onChange={(event) => onUpdate({ p: event.target.value })}
        />
      </div>
    </div>
  );
}
