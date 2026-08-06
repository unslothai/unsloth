// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Search-as-you-type pickers for the Audio Train panel, replacing a fixed Select
// of seven bases and a bare text input for the dataset. Curated entries stay
// pinned above the live Hub rows, so the known-good picks are still one click away.

import { type ReactElement, useEffect, useMemo, useRef, useState } from "react";

import {
  Combobox,
  ComboboxContent,
  ComboboxEmpty,
  ComboboxInput,
  ComboboxItem,
  ComboboxList,
} from "@/components/ui/combobox";
import { Spinner } from "@/components/ui/spinner";
import { useDebouncedValue } from "@/hooks";
import { useHubDatasetSearch } from "@/features/hub/hooks/use-hub-dataset-search";
import { useHubModelSearch } from "@/features/hub/hooks/use-hub-model-search";
import { AUDIO_GEN_TASKS } from "@/features/model-picker/components/model-selector/pickers";

/** A text field that is also a list. Typing still wins, so an unindexed repo id
 *  submits as the bare Input allowed. */
function HubCombobox({
  value,
  onValueChange,
  items,
  isLoading,
  placeholder,
  disabled,
  emptyLabel,
  renderItem,
}: {
  value: string;
  onValueChange: (value: string) => void;
  items: string[];
  isLoading: boolean;
  placeholder: string;
  disabled?: boolean;
  emptyLabel: string;
  renderItem?: (id: string) => ReactElement | string;
}): ReactElement {
  const [inputValue, setInputValue] = useState(value);
  const selectingRef = useRef(false);
  const anchorRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    setInputValue(value);
  }, [value]);

  return (
    <div
      ref={anchorRef}
      onKeyDown={(event) => {
        if (event.key !== "Enter") return;
        if (!(event.target instanceof HTMLInputElement)) return;
        event.preventDefault();
        // A typed "owner/name" beats the first row: it is already exact.
        const typed = event.target.value.trim();
        if (typed.includes("/")) {
          onValueChange(typed);
          return;
        }
        if (items.length > 0) onValueChange(items[0]);
        else if (typed) onValueChange(typed);
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
        disabled={disabled}
      >
        <ComboboxInput
          className="w-full"
          placeholder={placeholder}
          disabled={disabled}
        />
        <ComboboxContent anchor={anchorRef}>
          {isLoading ? (
            <div className="flex items-center gap-2 px-2 py-3 text-ui-11p5 text-muted-foreground">
              <Spinner className="size-3.5" />
              Searching the Hub…
            </div>
          ) : (
            <ComboboxEmpty>{emptyLabel}</ComboboxEmpty>
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
                {renderItem ? renderItem(id) : id}
              </ComboboxItem>
            )}
          </ComboboxList>
        </ComboboxContent>
      </Combobox>
      {/* The input is debounced, so it can trail the committed pick for a tick. */}
      {inputValue.trim() && inputValue.trim() !== value.trim() ? (
        <p className="mt-1 text-ui-11p5 text-muted-foreground">
          Selected: {value || "none"}
        </p>
      ) : null}
    </div>
  );
}

/** Curated bases first, then any Hub model with a TTS/ASR pipeline tag. GGUF is
 *  excluded: adapters need attachable weights, and the trainer rejects it. */
export function AudioBaseCombobox({
  value,
  onValueChange,
  curated,
  labelFor,
  disabled,
  accessToken,
}: {
  value: string;
  onValueChange: (value: string) => void;
  curated: readonly string[];
  labelFor: (repoId: string) => string | undefined;
  disabled?: boolean;
  accessToken?: string;
}): ReactElement {
  const [query, setQuery] = useState("");
  const debounced = useDebouncedValue(query);
  const { results, isLoading } = useHubModelSearch(debounced, {
    task: [...AUDIO_GEN_TASKS],
    ownerScope: "all",
    excludeGguf: true,
    accessToken,
    sortBy: "trendingScore",
    sortDirection: "desc",
  });

  const items = useMemo(() => {
    const q = debounced.trim().toLowerCase();
    const matchesQuery = (id: string) => !q || id.toLowerCase().includes(q);
    const ids = curated.filter(matchesQuery);
    const seen = new Set(ids.map((id) => id.toLowerCase()));
    for (const result of results) {
      const key = result.id.toLowerCase();
      // A GGUF repo past the tag filter is still untrainable.
      if (seen.has(key) || result.isGguf || /-gguf$/i.test(result.id)) continue;
      seen.add(key);
      ids.push(result.id);
    }
    if (value.trim() && !seen.has(value.trim().toLowerCase())) ids.push(value);
    return ids;
  }, [curated, results, debounced, value]);

  return (
    <div onInput={(event) => setQuery((event.target as HTMLInputElement).value)}>
      <HubCombobox
        value={value}
        onValueChange={onValueChange}
        items={items}
        isLoading={isLoading}
        disabled={disabled}
        placeholder="Search audio models, or paste owner/model"
        emptyLabel="No audio models found"
        renderItem={(id) => labelFor(id) ?? id}
      />
    </div>
  );
}

/** Dataset picker over the Hub's dataset index, the example pinned first. */
export function AudioDatasetCombobox({
  value,
  onValueChange,
  curated,
  disabled,
  accessToken,
}: {
  value: string;
  onValueChange: (value: string) => void;
  curated: readonly string[];
  disabled?: boolean;
  accessToken?: string;
}): ReactElement {
  const [query, setQuery] = useState("");
  const debounced = useDebouncedValue(query);
  const { results, isLoading } = useHubDatasetSearch(debounced, {
    accessToken,
    sortBy: "trendingScore",
    sortDirection: "desc",
  });

  const items = useMemo(() => {
    const q = debounced.trim().toLowerCase();
    const ids = curated.filter((id) => !q || id.toLowerCase().includes(q));
    const seen = new Set(ids.map((id) => id.toLowerCase()));
    for (const result of results) {
      const key = result.id.toLowerCase();
      if (seen.has(key)) continue;
      seen.add(key);
      ids.push(result.id);
    }
    if (value.trim() && !seen.has(value.trim().toLowerCase())) ids.push(value);
    return ids;
  }, [curated, results, debounced, value]);

  return (
    <div onInput={(event) => setQuery((event.target as HTMLInputElement).value)}>
      <HubCombobox
        value={value}
        onValueChange={onValueChange}
        items={items}
        isLoading={isLoading}
        disabled={disabled}
        placeholder="Search datasets, or paste owner/dataset"
        emptyLabel="No datasets found"
      />
    </div>
  );
}
