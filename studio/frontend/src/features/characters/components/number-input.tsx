// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Input } from "@/components/ui/input";
import { useEffect, useState } from "react";
import { clampNumber } from "../draft";

type ClampedNumberInputProps = {
  value: number;
  min?: number;
  max?: number;
  step?: number;
  className?: string;
  "aria-label"?: string;
  onChange: (value: number) => void;
};

/** A number field that clamps when left, so clearing it to type a new value does not snap back mid-edit. */
export function ClampedNumberInput({ value, min, max, step, className, onChange, ...rest }: ClampedNumberInputProps) {
  const [text, setText] = useState(String(value));
  useEffect(() => setText(String(value)), [value]);
  return (
    <Input
      {...rest}
      type="number"
      min={min}
      max={max}
      step={step}
      value={text}
      className={className}
      onChange={(event) => {
        setText(event.target.value);
        const parsed = Number(event.target.value);
        if (event.target.value.trim() && Number.isFinite(parsed) && parsed === clampNumber(parsed, min, max)) {
          onChange(parsed);
        }
      }}
      onBlur={() => {
        const next = clampNumber(text.trim() ? Number(text) : Number.NaN, min, max, value);
        setText(String(next));
        if (next !== value) onChange(next);
      }}
    />
  );
}
