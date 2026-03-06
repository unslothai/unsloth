import { Slider as SliderPrimitive } from "radix-ui";
import * as React from "react";

import { cn } from "@/lib/utils";

function Slider({
  className,
  defaultValue,
  value,
  min = 0,
  max = 100,
  ...props
}: React.ComponentProps<typeof SliderPrimitive.Root>) {
  const _values = React.useMemo(
    () =>
      Array.isArray(value)
        ? value
        : Array.isArray(defaultValue)
          ? defaultValue
          : [min, max],
    [value, defaultValue, min, max],
  );

  // For single-thumb horizontal sliders, render the fill bar as a sibling of
  // the track (outside its overflow-hidden container) so it can align flush
  // with the thumb center without being clipped. The Range inside the track
  // is hidden in this case to avoid double-painting.
  const isSingleThumb = _values.length === 1;
  const fillPercent = isSingleThumb
    ? Math.min(100, Math.max(0, (((_values[0] ?? min) - min) / (max - min)) * 100))
    : null;

  return (
    <SliderPrimitive.Root
      data-slot="slider"
      defaultValue={defaultValue}
      value={value}
      min={min}
      max={max}
      className={cn(
        "data-vertical:min-h-40 relative flex w-full touch-none items-center select-none data-disabled:opacity-50 data-vertical:h-full data-vertical:w-auto data-vertical:flex-col",
        className,
      )}
      {...props}
    >
      <SliderPrimitive.Track
        data-slot="slider-track"
        className="bg-muted rounded-4xl data-horizontal:h-3 data-horizontal:w-full data-vertical:h-full data-vertical:w-3 bg-muted relative grow overflow-hidden data-horizontal:w-full data-vertical:h-full cursor-pointer"
      >
        <SliderPrimitive.Range
          data-slot="slider-range"
          className={cn(
            "bg-primary absolute select-none data-horizontal:h-full data-vertical:w-full",
            isSingleThumb && "opacity-0",
          )}
        />
      </SliderPrimitive.Track>
      {isSingleThumb && (
        <div
          aria-hidden={true}
          className="absolute inset-y-0 left-0 my-auto h-3 rounded-4xl bg-primary pointer-events-none"
          style={{ width: `${fillPercent}%` }}
        />
      )}
      {Array.from({ length: _values.length }, (_, index) => (
        <SliderPrimitive.Thumb
          data-slot="slider-thumb"
          key={index}
          className="border-primary ring-ring/50 size-4 rounded-4xl border bg-white shadow-sm block shrink-0 select-none cursor-pointer disabled:pointer-events-none disabled:opacity-50 transition-transform duration-100 ease-out hover:scale-110 hover:ring-4 active:scale-95 focus-visible:ring-4 focus-visible:outline-hidden"
        />
      ))}
    </SliderPrimitive.Root>
  );
}

export { Slider };
