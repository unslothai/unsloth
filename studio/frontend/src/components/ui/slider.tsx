// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Slider as SliderPrimitive } from "radix-ui";
import * as React from "react";

import { cn } from "@/lib/utils";

const THUMB_SIZE_PX = 16;

function getThumbInBoundsOffset(width: number, percent: number) {
  const halfWidth = width / 2;
  const halfPercent = 50;

  if (percent <= 0) return halfWidth;
  if (percent >= 100) return -halfWidth;

  return halfWidth - (percent / halfPercent) * halfWidth;
}

function Slider({
  className,
  defaultValue,
  value,
  min = 0,
  max = 100,
  orientation = "horizontal",
  onValueChange,
  ...props
}: React.ComponentProps<typeof SliderPrimitive.Root>) {
  const isControlled = Array.isArray(value);
  const [uncontrolledValues, setUncontrolledValues] =
    React.useState<number[]>(() =>
      Array.isArray(defaultValue) ? defaultValue : [min, max],
    );

  const values = isControlled ? value : uncontrolledValues;
  const handleValueChange = React.useCallback(
    (nextValues: number[]) => {
      if (!isControlled) {
        setUncontrolledValues(nextValues);
      }
      onValueChange?.(nextValues);
    },
    [isControlled, onValueChange],
  );
  const isSingleThumbHorizontal =
    values.length === 1 && orientation === "horizontal";
  const fillPercent = isSingleThumbHorizontal
    ? Math.min(
        100,
        Math.max(
          0,
          max === min ? 0 : (((values[0] ?? min) - min) / (max - min)) * 100,
        ),
      )
    : null;
  const fillWidth =
    fillPercent === null
      ? undefined
      : fillPercent <= 0
        ? "0%"
        : `calc(${fillPercent}% + ${getThumbInBoundsOffset(THUMB_SIZE_PX, fillPercent)}px)`;

  return (
    <SliderPrimitive.Root
      data-slot="slider"
      defaultValue={defaultValue}
      value={value}
      min={min}
      max={max}
      orientation={orientation}
      onValueChange={handleValueChange}
      className={cn(
        "data-vertical:min-h-40 relative flex w-full touch-none items-center select-none data-disabled:opacity-50 data-vertical:h-full data-vertical:w-auto data-vertical:flex-col",
        className,
      )}
      {...props}
    >
      <SliderPrimitive.Track
        data-slot="slider-track"
        className="bg-black/10 dark:bg-black/12 rounded-4xl data-horizontal:h-2 data-horizontal:w-full data-vertical:h-full data-vertical:w-2 relative grow overflow-hidden cursor-pointer"
      >
        <SliderPrimitive.Range
          data-slot="slider-range"
          className={cn(
            "bg-primary absolute select-none data-horizontal:h-full data-vertical:w-full",
            isSingleThumbHorizontal && "opacity-0",
          )}
        />
        {isSingleThumbHorizontal && (
          <div
            aria-hidden={true}
            className={cn(
              "absolute inset-y-0 left-0 bg-primary pointer-events-none",
              fillPercent === 100 ? "rounded-4xl" : "rounded-l-4xl",
            )}
            style={{ width: fillWidth }}
          />
        )}
      </SliderPrimitive.Track>
      {Array.from({ length: values.length }, (_, index) => (
        <SliderPrimitive.Thumb
          data-slot="slider-thumb"
          key={index}
          className="ring-ring/50 relative z-10 size-4 rounded-4xl bg-white shadow-sm block shrink-0 select-none cursor-pointer disabled:pointer-events-none disabled:opacity-50 transition-transform duration-100 ease-out hover:scale-110 hover:ring-4 active:scale-95 focus-visible:ring-4 focus-visible:outline-hidden"
        />
      ))}
    </SliderPrimitive.Root>
  );
}

export { Slider };
