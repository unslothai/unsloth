// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { ReactNode, Ref } from "react";
import { InfoHint } from "@/components/ui/info-hint";
import { Slider } from "@/components/ui/slider";
import { useSliderDraft } from "@/hooks/use-slider-draft";
import {
  NumericValueInput,
  type NumericValueInputHandle,
} from "./numeric-value-input";

const ROW_CLASS = "flex min-h-8 items-center justify-between gap-3";
const LABEL_CLASS =
  "min-w-0 truncate text-ui-13 font-medium leading-[1.25] tracking-nav text-foreground";
const NUMBER_INPUT_CLASS =
  "panel-field h-8 w-[calc(84px*var(--ui-space-scale,1))] shrink-0";

export function AdvancedGpuSlider({
  label,
  value,
  min,
  max,
  onChange,
  onDraft,
  format,
  info,
  inputRef,
  step = 1,
  disabled = false,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  onChange: (value: number) => void;
  onDraft?: (value: number | null) => void;
  format?: (value: number) => string | undefined;
  info?: ReactNode;
  inputRef?: Ref<NumericValueInputHandle>;
  step?: number;
  disabled?: boolean;
}) {
  const { draft, sliderProps } = useSliderDraft(value, onChange, onDraft);
  const shown = draft ?? value;
  return (
    <div className="space-y-2">
      <div className={ROW_CLASS}>
        <div className="flex min-w-0 items-center gap-1.5">
          <span className={LABEL_CLASS}>{label}</span>
          {info && <InfoHint>{info}</InfoHint>}
        </div>
        <NumericValueInput
          ref={inputRef}
          value={shown}
          min={min}
          max={max}
          step={step}
          onChange={onChange}
          displayValue={format?.(shown)}
          ariaLabel={label}
          className={NUMBER_INPUT_CLASS}
          fixedWidth={true}
          size={8}
          disabled={disabled}
        />
      </div>
      <Slider
        min={min}
        max={max}
        step={step}
        {...sliderProps}
        aria-label={label}
        disabled={disabled}
      />
    </div>
  );
}
