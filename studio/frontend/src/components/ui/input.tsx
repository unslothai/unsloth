// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type * as React from "react";

import { cn } from "@/lib/utils";

const BASE_CLASSES =
  // White fill with a subtle border,
  // fully-rounded pill for single-row controls.
  "bg-background border-border dark:border-transparent dark:bg-white/[0.06] focus-visible:border-ring dark:focus-visible:border-transparent dark:focus-visible:bg-white/[0.12] aria-invalid:ring-destructive/20 dark:aria-invalid:ring-destructive/40 aria-invalid:border-destructive dark:aria-invalid:border-destructive/50 h-9 rounded-full border px-3.5 py-1 text-base transition-colors file:h-7 file:text-sm file:font-medium aria-invalid:ring-[3px] md:text-sm file:text-foreground placeholder:text-muted-foreground w-full min-w-0 outline-none file:inline-flex file:border-0 file:bg-transparent disabled:pointer-events-none disabled:cursor-not-allowed disabled:opacity-50";

function stepNumberInput(input: HTMLInputElement, direction: 1 | -1): void {
  if (input.disabled || input.readOnly) {
    return;
  }
  const step = input.step === "" ? 1 : Number(input.step) || 1;
  const min = input.min === "" ? null : Number(input.min);
  const max = input.max === "" ? null : Number(input.max);
  // An empty field steps from its placeholder (the effective default).
  const current =
    input.value === "" ? Number(input.placeholder) : Number(input.value);
  let next: number;
  if (Number.isFinite(current)) {
    // Snap to the step grid (anchored at min, like the native spinner) rather
    // than adding step to an off-grid typed value, which would leave a
    // step-invalid result. Mirrors HTMLInputElement.stepUp/stepDown.
    const base = min ?? 0;
    const pos = (current - base) / step;
    const rounded = Math.round(pos);
    const onGrid = Math.abs(pos - rounded) < 1e-9;
    const nextPos = onGrid
      ? rounded + direction
      : direction === 1
        ? Math.ceil(pos)
        : Math.floor(pos);
    next = base + nextPos * step;
  } else {
    // Stepping an empty, non-numeric field starts from the minimum, matching
    // the native spinner.
    next = min ?? direction * step;
  }
  if (min !== null) {
    next = Math.max(min, next);
  }
  if (max !== null) {
    next = Math.min(max, next);
  }
  // Trim float noise from fractional steps.
  const decimals = (String(step).split(".")[1] ?? "").length;
  const value = String(Number(next.toFixed(decimals)));
  // Write through the native setter and emit "input" so the React onChange
  // of controlled callers fires.
  const setter = Object.getOwnPropertyDescriptor(
    HTMLInputElement.prototype,
    "value",
  )?.set;
  setter?.call(input, value);
  input.dispatchEvent(new Event("input", { bubbles: true }));
}

function StepperArrow({ direction }: { direction: 1 | -1 }) {
  return (
    <svg
      viewBox="0 0 10 6"
      aria-hidden="true"
      className={cn("h-[5px] w-2", direction === -1 && "rotate-180")}
    >
      <path d="M5 0.5 9 5.5H1Z" fill="currentColor" />
    </svg>
  );
}

function StepperButton({ direction }: { direction: 1 | -1 }) {
  return (
    <button
      type="button"
      tabIndex={-1}
      onMouseDown={(event) => {
        // Keep focus on the field itself.
        event.preventDefault();
      }}
      onClick={(event) => {
        const input = event.currentTarget
          .closest('[data-slot="number-input"]')
          ?.querySelector("input");
        if (input) {
          input.focus({ preventScroll: true });
          stepNumberInput(input, direction);
        }
      }}
      className="flex h-1/2 w-full cursor-default items-center justify-center text-foreground/55 hover:bg-black/10 active:bg-black/15 dark:text-white/70 dark:hover:bg-white/10 dark:active:bg-white/15"
    >
      <StepperArrow direction={direction} />
    </button>
  );
}

function Input({ className, type, ...props }: React.ComponentProps<"input">) {
  const field = (
    <input
      type={type}
      data-slot="input"
      className={cn(BASE_CLASSES, type === "number" && "pr-8", className)}
      {...props}
    />
  );
  if (type !== "number") {
    return field;
  }
  // Number fields swap the native spinner (hidden globally in index.css) for
  // a shared grey stepper.
  return (
    <span
      data-slot="number-input"
      className={cn(
        // The wrapper is now the flex/grid item, so mirror the field's width:
        // default to the full width the bare input used, and let an explicit
        // w-*/max-w-* from the caller win so the stepper stays on the field edge.
        // Also carry React Flow interaction classes (nodrag/nopan/nowheel) so
        // the stepper buttons, which live in the wrapper, don't drag the node.
        "group/number relative inline-flex items-center w-full min-w-0",
        (className ?? "")
          .split(/\s+/)
          .filter(
            (c) =>
              /^(?:min-w|max-w|w)-/.test(c) ||
              c === "nodrag" ||
              c === "nopan" ||
              c === "nowheel",
          ),
      )}
    >
      {field}
      <span
        aria-hidden="true"
        className="absolute top-1/2 right-1.5 flex h-[21px] w-4 -translate-y-1/2 flex-col overflow-hidden rounded-[5px] bg-black/[0.07] group-has-[input:disabled]/number:pointer-events-none group-has-[input:disabled]/number:opacity-40 dark:bg-white/[0.12]"
      >
        <StepperButton direction={1} />
        <StepperButton direction={-1} />
      </span>
    </span>
  );
}

export { Input };
