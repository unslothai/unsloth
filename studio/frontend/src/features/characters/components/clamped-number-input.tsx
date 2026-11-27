// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Input } from "@/components/ui/input";
import { useEffect, useRef, useState } from "react";
import { clampNumber } from "../draft";

type ClampedNumberInputProps = {
  value: number;
  min?: number;
  max?: number;
  step?: number;
  className?: string;
  "aria-label"?: string;
  /** Report only when the field is left, so each keystroke is not a save. */
  commitOnBlur?: boolean;
  onChange: (value: number) => void;
};

/** A number field that clamps when left, so clearing it to type a new value does not snap back mid-edit. */
export function ClampedNumberInput({
  value,
  min,
  max,
  step,
  className,
  commitOnBlur = false,
  onChange,
  ...rest
}: ClampedNumberInputProps) {
  const [text, setText] = useState(String(value));
  // Committing on blur, typed text wins over a stored value arriving meanwhile.
  const typed = useRef(false);
  useEffect(() => {
    if (!commitOnBlur || !typed.current) setText(String(value));
  }, [value, commitOnBlur]);
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
        typed.current = true;
        setText(event.target.value);
        if (commitOnBlur) return;
        const parsed = Number(event.target.value);
        if (event.target.value.trim() && Number.isFinite(parsed) && parsed === clampNumber(parsed, min, max)) {
          onChange(parsed);
        }
      }}
      onBlur={() => {
        if (commitOnBlur && !typed.current) return;
        typed.current = false;
        const next = clampNumber(text.trim() ? Number(text) : Number.NaN, min, max, value);
        setText(String(next));
        if (next !== value) onChange(next);
      }}
    />
  );
}
