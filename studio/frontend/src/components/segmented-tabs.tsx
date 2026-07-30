// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { cn } from "@/lib/utils";
import type { ReactNode } from "react";
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
  size?: "compact" | "default";
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
        size === "compact" ? "h-8" : "h-9",
        className,
      )}
    >
      <span
        aria-hidden="true"
        className="hub-tab-toggle-pill pointer-events-none absolute inset-y-0 left-0 rounded-full transition-transform duration-200 ease-out"
        style={{
          width: `${100 / options.length}%`,
          transform: `translateX(${activeIndex * 100}%)`,
        }}
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
              "relative z-10 h-full min-w-0 flex-1 cursor-pointer rounded-full border-0 px-3 py-0 text-ui-12p5",
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
