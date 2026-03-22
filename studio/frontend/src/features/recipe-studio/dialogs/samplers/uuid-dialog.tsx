// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Input } from "@/components/ui/input";
import type { ReactElement } from "react";
import type { SamplerConfig } from "../../types";
import { NameField } from "../shared/name-field";
import { FieldLabel } from "../shared/field-label";

type UuidDialogProps = {
  config: SamplerConfig;
  onUpdate: (patch: Partial<SamplerConfig>) => void;
};

export function UuidDialog({
  config,
  onUpdate,
}: UuidDialogProps): ReactElement {
  const uuidId = `${config.id}-uuid-format`;
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
      <div className="grid gap-1.5">
        <FieldLabel
          label="UUID format (optional)"
          htmlFor={uuidId}
          hint="Optional formatter e.g. prefix:, short, uppercase."
        />
        <Input
          id={uuidId}
          className="nodrag"
          value={config.uuid_format ?? ""}
          onChange={(event) => updateField("uuid_format", event.target.value)}
        />
      </div>
    </div>
  );
}
