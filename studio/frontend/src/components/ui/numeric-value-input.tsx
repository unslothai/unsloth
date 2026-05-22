// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { cn } from "@/lib/utils";
import { snapToStep } from "@/lib/snap-to-step";
import { useEffect, useRef, useState } from "react";

const NUMERIC_PATTERN = /^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$/;

/**
 * Editable numeric value that looks like text until focused. Unfocused it
 * shows `displayValue ?? value` (so labels like "Off"/"Max" render); on focus
 * it swaps to the raw value, selects it, and accepts free text. Commit happens
 * on blur or Enter (clamped + snapped to step); Escape reverts.
 */
export function NumericValueInput({
  value,
  min,
  max,
  step,
  onChange,
  displayValue,
  className,
  ariaLabel,
  size,
}: {
  value: number;
  min?: number;
  max?: number;
  step: number;
  onChange: (v: number) => void;
  displayValue?: string;
  className?: string;
  ariaLabel?: string;
  size?: number;
}) {
  const [focused, setFocused] = useState(false);
  const [draft, setDraft] = useState("");
  const cancelBlurCommitRef = useRef(false);
  const syncedValueRef = useRef(value);

  // While focused an external value change (e.g. a store-driven LR/batch
  // recompute on model/method change) must win over the stale draft, otherwise
  // blur commits the old number back and silently undoes the recompute.
  useEffect(() => {
    if (focused && value !== syncedValueRef.current) {
      syncedValueRef.current = value;
      setDraft(String(value));
    }
  }, [focused, value]);

  const commit = (raw: string) => {
    const trimmed = raw.trim();
    if (!NUMERIC_PATTERN.test(trimmed)) return;
    const parsed = Number.parseFloat(trimmed);
    if (!Number.isFinite(parsed)) return;
    const final = snapToStep(parsed, step, min, max);
    if (final !== value) onChange(final);
  };

  return (
    <input
      type="text"
      inputMode="decimal"
      size={size}
      value={focused ? draft : (displayValue ?? String(value))}
      aria-label={ariaLabel}
      onFocus={(e) => {
        cancelBlurCommitRef.current = false;
        syncedValueRef.current = value;
        setDraft(String(value));
        setFocused(true);
        const target = e.currentTarget;
        requestAnimationFrame(() => target.select());
      }}
      onBlur={() => {
        if (cancelBlurCommitRef.current) {
          cancelBlurCommitRef.current = false;
        } else {
          commit(draft);
        }
        setFocused(false);
      }}
      onChange={(e) => setDraft(e.target.value)}
      onKeyDown={(e) => {
        if (e.key === "Enter") {
          e.currentTarget.blur();
        } else if (e.key === "Escape") {
          cancelBlurCommitRef.current = true;
          setDraft(String(value));
          e.currentTarget.blur();
        }
      }}
      className={cn("panel-number-input", className)}
    />
  );
}
