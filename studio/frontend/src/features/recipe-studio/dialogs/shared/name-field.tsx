// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

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
    <div className="grid gap-1.5">
      <FieldLabel
        label={label ?? "Field name"}
        htmlFor={inputId}
        hint={
          hint ??
          "This name is used in prompts and in the final dataset."
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
