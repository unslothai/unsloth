// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  Combobox,
  ComboboxContent,
  ComboboxEmpty,
  ComboboxInput,
  ComboboxItem,
  ComboboxList,
} from "@/components/ui/combobox";
import { Spinner } from "@/components/ui/spinner";
import { useDebouncedValue, useHfDatasetSearch } from "@/hooks";
import { type ReactElement, useEffect, useMemo, useRef, useState } from "react";

type HfDatasetComboboxProps = {
  value: string;
  onValueChange: (value: string) => void;
  accessToken?: string;
  inputId?: string;
  placeholder?: string;
  className?: string;
};

export function HfDatasetCombobox({
  value,
  onValueChange,
  accessToken,
  inputId,
  placeholder = "Search datasets...",
  className,
}: HfDatasetComboboxProps): ReactElement {
  const [inputValue, setInputValue] = useState(value);
  const selectingRef = useRef(false);
  const anchorRef = useRef<HTMLDivElement>(null);
  const debouncedQuery = useDebouncedValue(inputValue);

  useEffect(() => {
    setInputValue(value);
  }, [value]);

  const { results, isLoading, error } = useHfDatasetSearch(debouncedQuery, {
    accessToken,
  });

  const items = useMemo(() => {
    const ids = results.map((item) => item.id);
    const selected = value.trim();
    if (selected && !ids.includes(selected)) {
      ids.push(selected);
    }
    return ids;
  }, [results, value]);

  return (
    <div
      ref={anchorRef}
      className={className}
      onKeyDown={(event) => {
        if (event.key !== "Enter") return;
        if (!(event.target instanceof HTMLInputElement)) return;
        event.preventDefault();
        if (items.length > 0) {
          onValueChange(items[0]);
          return;
        }
        const typed = event.target.value.trim();
        if (typed) {
          onValueChange(typed);
        }
      }}
    >
      <Combobox
        items={items}
        filteredItems={items}
        filter={null}
        value={value.trim() ? value : null}
        onValueChange={(next) => onValueChange(next ?? "")}
        onInputValueChange={(next) => {
          if (selectingRef.current) {
            selectingRef.current = false;
            return;
          }
          setInputValue(next);
        }}
        itemToStringValue={(item) => item}
        autoHighlight={true}
      >
        <ComboboxInput
          id={inputId}
          className="nodrag w-full"
          placeholder={placeholder}
        />
        <ComboboxContent anchor={anchorRef}>
          {isLoading ? (
            <div className="flex items-center gap-2 px-2 py-3 text-xs text-muted-foreground">
              <Spinner className="size-3.5" />
              Searching...
            </div>
          ) : (
            <ComboboxEmpty>No datasets found</ComboboxEmpty>
          )}
          <ComboboxList>
            {(id: string) => (
              <ComboboxItem
                key={id}
                value={id}
                onPointerDown={() => {
                  selectingRef.current = true;
                }}
              >
                {id}
              </ComboboxItem>
            )}
          </ComboboxList>
        </ComboboxContent>
      </Combobox>
      {error && (
        <p className="mt-1 text-xs text-destructive">
          {error}
        </p>
      )}
    </div>
  );
}
