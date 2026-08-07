


import { Slider } from "@/components/ui/slider";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { InformationCircleIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import type { ReactElement, ReactNode } from "react";

export function ParamsRow({
  label,
  tooltip,
  children,
}: {
  label: string;
  tooltip?: ReactNode;
  children: ReactNode;
}): ReactElement {
  return (
    // Match the tallest control so slider and select rows share one rhythm.
    <div className="flex min-h-9 items-center justify-between">
      <span className="flex items-center gap-1.5 text-xs font-medium text-muted-foreground">
        {label}
        {tooltip && (
          <Tooltip>
            <TooltipTrigger asChild={true}>
              <button
                type="button"
                className="text-foreground/70 hover:text-foreground"
              >
                <HugeiconsIcon
                  icon={InformationCircleIcon}
                  className="size-3"
                />
              </button>
            </TooltipTrigger>
            <TooltipContent>{tooltip}</TooltipContent>
          </Tooltip>
        )}
      </span>
      {children}
    </div>
  );
}

export function ParamsSliderRow({
  label,
  tooltip,
  value,
  onChange,
  min,
  max,
  step,
  format,
}: {
  label: string;
  tooltip?: ReactNode;
  value: number;
  onChange: (value: number) => void;
  min: number;
  max: number;
  step: number;
  format?: (value: number) => string;
}): ReactElement {
  return (
    <ParamsRow label={label} tooltip={tooltip}>
      <div className="flex items-center gap-3">
        <Slider
          value={[value]}
          onValueChange={([nextValue]) => onChange(nextValue)}
          min={min}
          max={max}
          step={step}
          className="w-32"
        />
        <input
          type="number"
          value={format ? format(value) : value}
          onChange={(event) => onChange(Number(event.target.value))}
          min={min}
          max={max}
          step={step}
          className="w-12 text-right font-mono text-xs font-medium bg-muted/50 border border-border rounded-lg px-1.5 py-0.5 focus:outline-none focus:ring-1 focus:ring-ring [&::-webkit-inner-spin-button]:appearance-none"
        />
      </div>
    </ParamsRow>
  );
}
