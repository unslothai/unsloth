// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { cn } from "@/lib/utils";
import {
  forwardRef,
  useEffect,
  useImperativeHandle,
  useRef,
  useState,
} from "react";

export function snapToStep(
  value: number,
  step: number,
  min?: number,
  max?: number,
): number {
  const lo = min ?? Number.NEGATIVE_INFINITY;
  const hi = max ?? Number.POSITIVE_INFINITY;
  const clamped = Math.min(Math.max(value, lo), hi);
  const stepStr = String(step);
  const decimals = stepStr.includes(".") ? stepStr.split(".")[1].length : 0;
  const base = Number.isFinite(lo) ? lo : 0;
  const snapped = base + Math.round((clamped - base) / step) * step;
  const reclamped = Math.min(Math.max(snapped, lo), hi);
  return Number(reclamped.toFixed(decimals));
}

function sanitizeNumeric(raw: string, allowNegative: boolean): string {
  const sign = allowNegative && raw.startsWith("-") ? "-" : "";
  const [head, ...rest] = raw.replace(/[^\d.]/g, "").split(".");
  const tail = rest.length > 0 ? `.${rest.join("")}` : "";
  return `${sign}${head}${tail}`;
}

export type NumericValueInputHandle = {
  /** Commit a valid focused/same-click draft; null when none is pending. */
  commit: () => number | null;
};

export const NumericValueInput = forwardRef<
  NumericValueInputHandle,
  {
    value: number;
    min?: number;
    max?: number;
    step: number;
    onChange: (v: number) => void;
    displayValue?: string;
    className?: string;
    ariaLabel?: string;
    size?: number;
    disabled?: boolean;
  }
>(function NumericValueInput(
  {
    value,
    min,
    max,
    step,
    onChange,
    displayValue,
    className,
    ariaLabel,
    size: sizeAttr,
    disabled = false,
  },
  ref,
) {
  const [focused, setFocused] = useState(false);
  const [draft, setDraft] = useState("");
  const cancelBlurCommitRef = useRef(false);
  const draftRef = useRef("");
  const dirtyRef = useRef(false);
  // Same-click Load: blur commits via onChange and clears dirtyRef before the
  // button onClick runs, while parent `value` is still stale. Keep the blur
  // result for one imperative commit(); clear when `value` catches up or on
  // focus / external edits (Reset, slider).
  const lastBlurCommittedRef = useRef<number | null>(null);

  useEffect(() => {
    lastBlurCommittedRef.current = null;
  }, [value]);

  const commitDraft = (raw: string): number | null => {
    const parsed = Number.parseFloat(raw);
    if (!Number.isFinite(parsed)) {
      return null;
    }
    const final = snapToStep(parsed, step, min, max);
    if (final !== value) {
      onChange(final);
    }
    return final;
  };

  useImperativeHandle(
    ref,
    () => ({
      commit: () => {
        if (dirtyRef.current) {
          const raw = draftRef.current;
          const final = commitDraft(raw);
          dirtyRef.current = false;
          lastBlurCommittedRef.current = null;
          if (final == null) {
            draftRef.current = String(value);
          }
          if (focused) {
            setFocused(false);
          }
          return final;
        }
        const blurCommitted = lastBlurCommittedRef.current;
        if (blurCommitted != null) {
          lastBlurCommittedRef.current = null;
          if (focused) {
            setFocused(false);
          }
          return blurCommitted;
        }
        if (focused) {
          setFocused(false);
        }
        return null;
      },
    }),
    [draft, focused, max, min, onChange, step, value],
  );

  const displayed = focused ? draft : (displayValue ?? String(value));

  return (
    <input
      type="text"
      inputMode="decimal"
      disabled={disabled}
      size={sizeAttr}
      style={{
        boxSizing: "content-box",
        width: `calc(${Math.max(displayed.length, 4)}ch + 2px)`,
      }}
      value={displayed}
      aria-label={ariaLabel}
      onFocus={(e) => {
        cancelBlurCommitRef.current = false;
        dirtyRef.current = false;
        lastBlurCommittedRef.current = null;
        const next = String(value);
        draftRef.current = next;
        setDraft(next);
        setFocused(true);
        const target = e.currentTarget;
        requestAnimationFrame(() => target.select());
      }}
      onBlur={() => {
        if (cancelBlurCommitRef.current) {
          cancelBlurCommitRef.current = false;
          lastBlurCommittedRef.current = null;
        } else if (dirtyRef.current) {
          const final = commitDraft(draftRef.current);
          dirtyRef.current = false;
          if (final == null) {
            draftRef.current = String(value);
            lastBlurCommittedRef.current = null;
          } else {
            draftRef.current = String(final);
            lastBlurCommittedRef.current = final;
          }
        }
        setFocused(false);
      }}
      onChange={(e) => {
        dirtyRef.current = true;
        lastBlurCommittedRef.current = null;
        const next = sanitizeNumeric(e.target.value, (min ?? 0) < 0);
        draftRef.current = next;
        setDraft(next);
      }}
      onKeyDown={(e) => {
        if (e.key === "Enter") {
          e.currentTarget.blur();
        } else if (e.key === "Escape") {
          cancelBlurCommitRef.current = true;
          dirtyRef.current = false;
          lastBlurCommittedRef.current = null;
          const next = String(value);
          draftRef.current = next;
          setDraft(next);
          e.currentTarget.blur();
        }
      }}
      className={cn(className)}
    />
  );
});
