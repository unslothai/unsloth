// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { ChevronDownStandardIcon } from "@/lib/chevron-icons";
import { Tick02Icon } from "@/lib/tick-icon";
import { cn } from "@/lib/utils";
import { HugeiconsIcon } from "@hugeicons/react";
import {
  type KeyboardEvent,
  type ReactNode,
  useCallback,
  useId,
  useMemo,
  useRef,
  useState,
} from "react";

export interface HubOption<T extends string> {
  value: T;
  label: ReactNode;
  triggerLabel?: ReactNode;
  checkClassName?: string;
}

export function HubOptionMenu<T extends string>({
  value,
  options,
  onValueChange,
  className,
  contentClassName,
  ariaLabel,
  align = "start",
  showChevron = true,
  title,
  triggerContent,
  footer,
}: {
  value: T;
  options: readonly HubOption<T>[];
  onValueChange: (value: T) => void;
  className?: string;
  contentClassName?: string;
  ariaLabel: string;
  align?: "start" | "center" | "end";
  showChevron?: boolean;
  title?: string;
  triggerContent?: ReactNode;
  /** Rendered under the options behind a separator; clicks keep the menu open. */
  footer?: ReactNode;
}) {
  const [open, setOpen] = useState(false);
  // -1 = nothing highlighted (no hover, no keyboard nav yet).
  const [activeIndex, setActiveIndex] = useState(-1);
  const triggerRef = useRef<HTMLButtonElement | null>(null);
  const listboxRef = useRef<HTMLDivElement | null>(null);
  const idBase = useId();
  const listboxId = `${idBase}-listbox`;

  const selectedIndex = useMemo(() => {
    const index = options.findIndex((option) => option.value === value);
    return index >= 0 ? index : 0;
  }, [options, value]);
  const selected = options[selectedIndex];
  const resolvedActiveIndex =
    options.length === 0 || activeIndex < 0
      ? -1
      : Math.min(activeIndex, options.length - 1);
  const activeOptionId =
    resolvedActiveIndex >= 0
      ? `${idBase}-option-${resolvedActiveIndex}`
      : undefined;

  const activateIndex = useCallback((index: number) => {
    setActiveIndex((current) => (current === index ? current : index));
  }, []);

  const closeAndRestoreFocus = useCallback(() => {
    setOpen(false);
    requestAnimationFrame(() => triggerRef.current?.focus());
  }, []);

  const selectIndex = useCallback(
    (index: number) => {
      const option = options[index];
      if (!option) return;
      onValueChange(option.value);
      closeAndRestoreFocus();
    },
    [closeAndRestoreFocus, onValueChange, options],
  );

  const handleOpenChange = useCallback(
    (nextOpen: boolean) => {
      setOpen(nextOpen);
      if (nextOpen) {
        // Nothing highlighted until the user hovers or uses the keyboard;
        // keyboard nav anchors on the selected option (handleContentKeyDown).
        activateIndex(-1);
        requestAnimationFrame(() => listboxRef.current?.focus());
      }
    },
    [activateIndex],
  );

  const handleContentKeyDown = useCallback(
    (event: KeyboardEvent<HTMLDivElement>) => {
      if (options.length === 0) return;
      const currentIndex =
        resolvedActiveIndex >= 0 ? resolvedActiveIndex : selectedIndex;

      if (event.key === "Escape") {
        event.preventDefault();
        closeAndRestoreFocus();
        return;
      }
      if (event.key === "ArrowDown") {
        event.preventDefault();
        // First arrow press highlights the selected option, then steps.
        setActiveIndex(
          resolvedActiveIndex < 0
            ? selectedIndex
            : (currentIndex + 1) % options.length,
        );
        return;
      }
      if (event.key === "ArrowUp") {
        event.preventDefault();
        setActiveIndex(
          resolvedActiveIndex < 0
            ? selectedIndex
            : (currentIndex - 1 + options.length) % options.length,
        );
        return;
      }
      if (event.key === "Home") {
        event.preventDefault();
        setActiveIndex(0);
        return;
      }
      if (event.key === "End") {
        event.preventDefault();
        setActiveIndex(options.length - 1);
        return;
      }
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        selectIndex(currentIndex);
      }
    },
    [
      closeAndRestoreFocus,
      options,
      resolvedActiveIndex,
      selectIndex,
      selectedIndex,
    ],
  );

  return (
    <Popover open={open} onOpenChange={handleOpenChange}>
      <PopoverTrigger asChild={true}>
        <button
          ref={triggerRef}
          type="button"
          aria-haspopup="listbox"
          aria-expanded={open}
          aria-controls={open ? listboxId : undefined}
          aria-label={ariaLabel}
          title={title}
          className={cn(
            "field-trigger hub-menu-trigger field-soft field-filter inline-flex h-9 shrink-0 cursor-pointer items-center justify-between gap-2.5 rounded-full pl-3 pr-2.5 text-[12.5px] transition-colors",
            "focus-visible:border-border focus-visible:ring-0 focus-visible:ring-offset-0",
            className,
          )}
          onKeyDown={(event) => {
            if (event.key === "ArrowDown" || event.key === "ArrowUp") {
              event.preventDefault();
              handleOpenChange(true);
            }
          }}
        >
          <span className="min-w-0 flex-1 truncate text-left">
            {triggerContent ??
              selected?.triggerLabel ??
              selected?.label ??
              value}
          </span>
          {showChevron && (
            <HugeiconsIcon
              icon={ChevronDownStandardIcon}
              className="size-3.5 shrink-0 text-muted-foreground"
            />
          )}
        </button>
      </PopoverTrigger>
      <PopoverContent
        align={align}
        side="bottom"
        sideOffset={8}
        collisionPadding={12}
        onCloseAutoFocus={(event) => event.preventDefault()}
        className={cn(
          "hub-menu-instant menu-soft-surface w-max min-w-[var(--radix-popover-trigger-width)] max-w-[min(var(--radix-popover-content-available-width),calc(100vw-1rem))] rounded-[14px] p-1 ring-0",
          contentClassName,
        )}
      >
        <div
          ref={listboxRef}
          id={listboxId}
          role="listbox"
          aria-label={ariaLabel}
          aria-activedescendant={activeOptionId}
          tabIndex={0}
          onKeyDown={handleContentKeyDown}
          onPointerLeave={() => activateIndex(-1)}
          className="outline-none"
        >
          {options.map((option, index) => {
            const selectedOption = option.value === value;
            const activeOption = index === resolvedActiveIndex;
            return (
              <div
                key={option.value}
                id={`${idBase}-option-${index}`}
                role="option"
                aria-selected={selectedOption}
                data-highlighted={activeOption || undefined}
                onClick={() => {
                  selectIndex(index);
                }}
                onPointerEnter={() => activateIndex(index)}
                className={cn(
                  // 14px menu radius minus the 4px padding, so the hover nests
                  // cleanly into the dropdown's corners.
                  "relative flex w-full min-w-0 cursor-pointer select-none items-center rounded-[10px] py-2 pr-8 pl-3 text-left text-sm leading-snug outline-none transition-colors",
                )}
              >
                <span className="flex min-w-0 flex-1 items-center gap-2.5 overflow-hidden whitespace-normal break-words">
                  {option.label}
                </span>
                {selectedOption && (
                  <span className="pointer-events-none absolute right-2 flex size-4 items-center justify-center">
                    <HugeiconsIcon
                      icon={Tick02Icon}
                      strokeWidth={2}
                      className={cn("size-4", option.checkClassName)}
                    />
                  </span>
                )}
              </div>
            );
          })}
        </div>
        {footer && (
          // -mt-3 cancels the surface's 16px flex gap down to 4px. No side
          // padding: the footer label carries the same padding as the options
          // so its checkbox lines up with the option text.
          <div className="-mt-3 border-t border-border/60 pt-1">{footer}</div>
        )}
      </PopoverContent>
    </Popover>
  );
}
