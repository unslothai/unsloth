// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { PlusSignIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useMemo, useRef, useState } from "react";

import { Button } from "@/components/ui/button";
import {
  Command,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
  CommandSeparator,
} from "@/components/ui/command";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { ChevronDownStandardIcon } from "@/lib/chevron-icons";

import type { DiffusionLoraInfo, LoraSpecInput } from "./api";
import { customLoraId, loraAddGroups } from "./lib/lora-add-options";

const EMPTY_ROW = "__empty-row__";
const CUSTOM_ROW = "__custom-row__";

/**
 * Add LoRA: a searchable list of the catalog when there is one, else a plain button for an empty row.
 * `onAdd`'s `focusRow` asks the caller to focus the new row's id field, which the popover then leaves alone.
 */
export function LoraAddMenu({
  catalog,
  takenIds,
  onAdd,
}: {
  catalog: readonly DiffusionLoraInfo[];
  takenIds: ReadonlySet<string>;
  onAdd: (spec: LoraSpecInput, focusRow: boolean) => void;
}) {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  // Set when the pick needs its typed id field focused rather than the trigger.
  const keepFocusOnRow = useRef(false);
  const groups = useMemo(
    () => loraAddGroups(catalog, takenIds, query),
    [catalog, takenIds, query],
  );
  const custom = customLoraId(query, catalog, takenIds);

  if (catalog.length === 0) {
    return (
      <Button
        type="button"
        variant="secondary"
        size="sm"
        className="flex-1"
        onClick={() => onAdd({ id: "", weight: 1 }, true)}
      >
        <HugeiconsIcon icon={PlusSignIcon} className="size-3.5" />
        Add LoRA
      </Button>
    );
  }

  const openChange = (next: boolean) => {
    setOpen(next);
    if (!next) setQuery("");
  };
  const add = (spec: LoraSpecInput, focusRow: boolean) => {
    keepFocusOnRow.current = focusRow;
    onAdd(spec, focusRow);
    openChange(false);
  };

  return (
    <Popover open={open} onOpenChange={openChange}>
      <PopoverTrigger asChild={true}>
        <Button type="button" variant="secondary" size="sm" className="flex-1">
          <HugeiconsIcon icon={PlusSignIcon} className="size-3.5" />
          Add LoRA
          <HugeiconsIcon
            icon={ChevronDownStandardIcon}
            strokeWidth={2}
            className="size-3.5 opacity-60"
          />
        </Button>
      </PopoverTrigger>
      <PopoverContent
        align="start"
        sideOffset={4}
        collisionPadding={12}
        // Capped at the space Radix measures; only the catalog gives way, so search and footer stay reachable.
        className="w-[max(var(--radix-popover-trigger-width),20rem)] gap-0 overflow-hidden rounded-lg p-1"
        onCloseAutoFocus={(e) => {
          if (keepFocusOnRow.current) e.preventDefault();
          keepFocusOnRow.current = false;
        }}
      >
        {/* Own filtering: the list is small and grouped, and the custom-id row must not be filtered away. */}
        <Command
          shouldFilter={false}
          className="min-h-0 rounded-none bg-transparent p-0"
        >
          <CommandInput
            value={query}
            onValueChange={setQuery}
            aria-label="Search LoRAs"
            placeholder="Search LoRAs or paste owner/name"
          />
          {/* Only the catalog scrolls, and it shrinks first in a short window, so the footer row stays in view.
              The flex column runs through cmdk's own sizer div. */}
          <CommandList className="flex max-h-none min-h-0 flex-col overflow-hidden [&>[cmdk-list-sizer]]:flex [&>[cmdk-list-sizer]]:min-h-0 [&>[cmdk-list-sizer]]:flex-col">
            {/* Not CommandEmpty: the Hugging Face row below is always an item, so cmdk never counts zero. */}
            {groups.length === 0 && !custom && (
              <p className="px-3 py-4 text-center text-sm text-muted-foreground">
                No LoRA matches.
              </p>
            )}
            <div className="hover-scrollbar max-h-64 min-h-0 overflow-y-auto overscroll-contain">
              {groups.map((group) => (
                <CommandGroup key={group.heading} heading={group.heading}>
                  {group.options.map(({ lora, added }) => (
                    <CommandItem
                      key={lora.id}
                      value={lora.id}
                      disabled={added}
                      data-checked={added}
                      title={lora.id}
                      onSelect={() =>
                        add(
                          { id: lora.id, weight: lora.weight_default || 1 },
                          false,
                        )
                      }
                      className="cursor-pointer"
                    >
                      {/* File-stem names run long with no spaces, so wrap anywhere rather than cut. */}
                      <span className="line-clamp-2 min-w-0 flex-1 [overflow-wrap:anywhere]">
                        {lora.display_name}
                      </span>
                      {added && (
                        <span className="shrink-0 text-xs text-muted-foreground">
                          Added
                        </span>
                      )}
                    </CommandItem>
                  ))}
                </CommandGroup>
              ))}
            </div>
            {groups.length > 0 && <CommandSeparator className="shrink-0" />}
            <CommandGroup className="shrink-0">
              {custom ? (
                <CommandItem
                  value={CUSTOM_ROW}
                  disabled={custom.added}
                  data-checked={custom.added}
                  onSelect={() => add({ id: custom.id, weight: 1 }, false)}
                  className="cursor-pointer"
                >
                  <span className="min-w-0 flex-1 [overflow-wrap:anywhere]">
                    Use <span className="font-mono text-xs">{custom.id}</span>
                  </span>
                  {custom.added && (
                    <span className="shrink-0 text-xs text-muted-foreground">
                      Added
                    </span>
                  )}
                </CommandItem>
              ) : (
                <CommandItem
                  value={EMPTY_ROW}
                  onSelect={() => add({ id: "", weight: 1 }, true)}
                  className="cursor-pointer"
                >
                  <span className="min-w-0 flex-1 truncate text-muted-foreground">
                    Enter a Hugging Face repo or file…
                  </span>
                </CommandItem>
              )}
            </CommandGroup>
          </CommandList>
        </Command>
      </PopoverContent>
    </Popover>
  );
}
