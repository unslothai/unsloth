// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Slider as SliderPrimitive } from "radix-ui";
import * as React from "react";

import { cn } from "@/lib/utils";

const THUMB_SIZE_PX = 20;

function getThumbInBoundsOffset(width: number, percent: number) {
  const halfWidth = width / 2;
  const halfPercent = 50;

  if (percent <= 0) return halfWidth;
  if (percent >= 100) return -halfWidth;

  return halfWidth - (percent / halfPercent) * halfWidth;
}

type SliderProps = React.ComponentProps<typeof SliderPrimitive.Root> & {
  /** Radix does not synthesise `aria-valuetext`; e.g. the context slider's leftmost means Auto. */
  thumbValueText?: (value: number, index: number) => string;
};

function Slider({
  className,
  defaultValue,
  value,
  min = 0,
  max = 100,
  orientation = "horizontal",
  onValueChange,
  thumbValueText,
  ...props
}: SliderProps) {
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
  // The thumb scales with the UI size, so the fill measures it.
  const thumbRef = React.useRef<HTMLSpanElement | null>(null);
  const [thumbWidth, setThumbWidth] = React.useState(THUMB_SIZE_PX);
  React.useLayoutEffect(() => {
    const thumb = thumbRef.current;
    if (!thumb) return;
    const measure = () => {
      const width = thumb.offsetWidth;
      if (width > 0) {
        setThumbWidth((current) => (current === width ? current : width));
      }
    };
    measure();
    if (typeof ResizeObserver === "undefined") return;
    const observer = new ResizeObserver(measure);
    observer.observe(thumb);
    return () => observer.disconnect();
  }, []);
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
        : `calc(${fillPercent}% + ${getThumbInBoundsOffset(thumbWidth, fillPercent)}px)`;

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
        "data-horizontal:min-h-(--slider-thumb-size) data-vertical:min-h-40 data-vertical:min-w-(--slider-thumb-size) relative flex w-full touch-none items-center select-none data-disabled:opacity-50 data-vertical:h-full data-vertical:w-auto data-vertical:flex-col",
        className,
      )}
      {...props}
    >
      <SliderPrimitive.Track
        data-slot="slider-track"
        className="bg-(--slider-track) rounded-full data-horizontal:h-(--slider-track-height) data-horizontal:w-full data-vertical:h-full data-vertical:w-(--slider-track-height) relative grow overflow-hidden cursor-pointer"
      >
        <SliderPrimitive.Range
          data-slot="slider-range"
          className={cn(
            "bg-(--slider-fill) absolute select-none data-horizontal:h-full data-vertical:w-full",
            isSingleThumbHorizontal && "opacity-0",
          )}
        />
        {isSingleThumbHorizontal && (
          <div
            aria-hidden={true}
            className={cn(
              "absolute inset-y-0 left-0 bg-(--slider-fill) pointer-events-none",
              fillPercent === 100 ? "rounded-full" : "rounded-l-full",
            )}
            style={{ width: fillWidth }}
          />
        )}
      </SliderPrimitive.Track>
      {Array.from({ length: values.length }, (_, index) => (
        <SliderPrimitive.Thumb
          data-slot="slider-thumb"
          key={index}
          ref={index === 0 ? thumbRef : undefined}
          // The thumb carries role="slider", so the name and spoken value belong here, not on Root.
          aria-label={props["aria-label"]}
          aria-labelledby={props["aria-labelledby"]}
          aria-valuetext={thumbValueText?.(values[index] ?? min, index)}
          className="ring-(--slider-fill)/30 relative z-10 size-(--slider-thumb-size) rounded-full bg-(--slider-thumb) shadow-(--slider-thumb-shadow) block shrink-0 select-none cursor-pointer disabled:pointer-events-none disabled:opacity-50 focus-visible:ring-4 focus-visible:outline-hidden"
        />
      ))}
    </SliderPrimitive.Root>
  );
}

export { Slider };
