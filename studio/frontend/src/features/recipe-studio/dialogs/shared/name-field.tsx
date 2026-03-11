// SPDX-License-Identifier: AGPL-3.0-only - See /studio/LICENSE.AGPL-3.0
// Copyright © 2025 Unsloth AI

import { Input } from "@/components/ui/input";
import { type ReactElement, useId } from "react";
import { FieldLabel } from "./field-label";

type NameFieldProps = {
  id?: string;
  value: string;
  onChange: (value: string) => void;
  label?: string;
  hint?: string;
};

export function NameField({
  id,
  value,
  onChange,
  label,
  hint,
}: NameFieldProps): ReactElement {
  const fallbackId = useId();
  const inputId = id ?? fallbackId;
  return (
    <div className="grid gap-2">
      <FieldLabel
        label={label ?? "Column name"}
        htmlFor={inputId}
        hint={
          hint ??
          "Unique field name used in templates and final dataset output."
        }
      />
      <Input
        id={inputId}
        className="nodrag"
        value={value}
        onChange={(event) => onChange(event.target.value)}
      />
    </div>
  );
}
