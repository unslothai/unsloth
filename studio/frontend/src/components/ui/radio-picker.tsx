// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { HugeiconsIcon } from "@hugeicons/react";
import { RadioGroup as RadioGroupPrimitive } from "radix-ui";
import { useState, type ReactNode } from "react";
import { Button } from "@/components/ui/button";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { ChevronDownStandardIcon } from "@/lib/chevron-icons";
import { Tick02Icon } from "@/lib/tick-icon";
import { cn } from "@/lib/utils";

export type RadioPickerOption<T extends string> = {
  value: T;
  label: ReactNode;
  disabled?: boolean;
};

/** A visual-settings picker that stays open while options are compared. */
export function RadioPicker<T extends string>({
  value,
  options,
  onValueChange,
  ariaLabel,
  disabled = false,
  className,
  contentClassName,
  align = "end",
}: {
  value: T;
  options: ReadonlyArray<RadioPickerOption<T>>;
  onValueChange: (value: T) => void;
  ariaLabel: string;
  disabled?: boolean;
  className?: string;
  contentClassName?: string;
  align?: "start" | "center" | "end";
}) {
  const [open, setOpen] = useState(false);
  const current = options.find((option) => option.value === value);
  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild={true}>
        <Button
          variant="picker"
          disabled={disabled}
          aria-label={ariaLabel}
          className={cn("w-52 justify-between font-normal", className)}
        >
          <span className="min-w-0 truncate">{current?.label ?? null}</span>
          <HugeiconsIcon
            icon={ChevronDownStandardIcon}
            strokeWidth={2}
            className="size-4 shrink-0 text-muted-foreground"
          />
        </Button>
      </PopoverTrigger>
      <PopoverContent
        align={align}
        sideOffset={6}
        className={cn("w-52 gap-0 rounded-lg p-1", contentClassName)}
      >
        <RadioGroupPrimitive.Root
          value={value}
          onValueChange={(next) => onValueChange(next as T)}
          orientation="vertical"
          disabled={disabled}
          aria-label={ariaLabel}
        >
          {options.map((option) => {
            const selected = option.value === value;
            return (
              <RadioGroupPrimitive.Item
                key={option.value}
                value={option.value}
                disabled={option.disabled}
                className={cn(
                  "relative flex w-full cursor-pointer items-center rounded-md py-2 pr-9 pl-3 text-left text-sm outline-none transition-colors hover:bg-accent focus-visible:bg-accent disabled:pointer-events-none disabled:opacity-50",
                  selected && "bg-accent/60",
                )}
              >
                <span className="min-w-0 truncate">{option.label}</span>
                {selected ? (
                  <HugeiconsIcon
                    icon={Tick02Icon}
                    className="absolute right-3 size-4 text-foreground"
                  />
                ) : null}
              </RadioGroupPrimitive.Item>
            );
          })}
        </RadioGroupPrimitive.Root>
      </PopoverContent>
    </Popover>
  );
}
