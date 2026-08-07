
import { cn } from "@/lib/utils";
import type { ReactNode } from "react";
import { SegmentedControlIndicator } from "./segmented-control";
import {
  type SegmentedSize,
  segmentedSegmentLabel,
  segmentedTrackHeight,
} from "./segmented-control-styles";
import { TabsList, TabsTrigger } from "./ui/tabs";

export interface SegmentedTabOption<T extends string> {
  value: T;
  label: ReactNode;
  disabled?: boolean;
}

export function SegmentedTabsList<T extends string>({
  value,
  options,
  ariaLabel,
  size = "default",
  className,
}: {
  value: T;
  options: readonly [
    SegmentedTabOption<T>,
    SegmentedTabOption<T>,
    ...SegmentedTabOption<T>[],
  ];
  ariaLabel: string;
  size?: SegmentedSize;
  className?: string;
}) {
  const activeIndex = Math.max(
    0,
    options.findIndex((option) => option.value === value),
  );

  return (
    <TabsList
      unstyled={true}
      aria-label={ariaLabel}
      className={cn(
        "hub-menu-trigger hub-tab-toggle relative inline-flex w-full shrink-0 items-center rounded-full",
        segmentedTrackHeight[size],
        className,
      )}
    >
      <SegmentedControlIndicator
        activeIndex={activeIndex}
        optionCount={options.length}
        className="inset-y-0"
      />
      {options.map((option) => {
        const active = option.value === value;
        return (
          <TabsTrigger
            key={option.value}
            value={option.value}
            disabled={option.disabled}
            indicatorClassName="hidden"
            className={cn(
              "relative z-10 h-full min-w-0 flex-1 cursor-pointer rounded-full border-0 py-0",
              segmentedSegmentLabel,
              active
                ? "text-foreground"
                : "text-muted-foreground hover:text-foreground",
            )}
          >
            {option.label}
          </TabsTrigger>
        );
      })}
    </TabsList>
  );
}
