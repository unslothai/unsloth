// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { SegmentedControlIndicator } from "@/components/segmented-control";
import { cn } from "@/lib/utils";
import { PICKER_OPTION_FOCUS_VISIBLE_CLASS } from "./picker-focus";
import { pickerTabId } from "./picker-tab-state";

function nextPickerTab<T extends string>(
  key: string,
  current: T,
  first: T,
  second: T,
): T | null {
  if (key === "ArrowLeft" || key === "ArrowRight") {
    return current === first ? second : first;
  }
  if (key === "Home") {
    return first;
  }
  if (key === "End") {
    return second;
  }
  return null;
}

export function PickerTabToggle<T extends string>({
  tab,
  options,
  onTabChange,
  idBase,
  panelId,
}: {
  tab: T;
  options: readonly [{ value: T; label: string }, { value: T; label: string }];
  onTabChange: (tab: T) => void;
  idBase: string;
  panelId: string;
}) {
  const [first, second] = options;
  return (
    <div
      role="tablist"
      className="hub-menu-trigger hub-tab-toggle relative inline-flex h-9 w-full select-none items-center rounded-full"
      onKeyDown={(event) => {
        const target = nextPickerTab(event.key, tab, first.value, second.value);
        if (target === null) {
          return;
        }
        event.preventDefault();
        if (target !== tab) {
          onTabChange(target);
        }
        document.getElementById(pickerTabId(idBase, target))?.focus();
      }}
    >
      <SegmentedControlIndicator
        activeIndex={tab === second.value ? 1 : 0}
        optionCount={options.length}
        className="inset-y-0 start-0"
      />
      {options.map((entry) => {
        const selected = tab === entry.value;
        return (
          <button
            key={entry.value}
            id={pickerTabId(idBase, entry.value)}
            type="button"
            role="tab"
            aria-selected={selected}
            aria-controls={panelId}
            tabIndex={selected ? 0 : -1}
            onClick={() => onTabChange(entry.value)}
            className={cn(
              "relative z-10 inline-flex h-full flex-1 select-none items-center justify-center rounded-full px-3 text-ui-12p5 font-medium transition-colors",
              PICKER_OPTION_FOCUS_VISIBLE_CLASS,
              selected
                ? "text-foreground"
                : "text-muted-foreground hover:text-foreground",
            )}
          >
            {entry.label}
          </button>
        );
      })}
    </div>
  );
}
