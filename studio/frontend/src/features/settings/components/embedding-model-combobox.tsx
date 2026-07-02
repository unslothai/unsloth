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
import type { PipelineType } from "@huggingface/hub";
import { useHubModelSearch } from "@/features/hub/hooks/use-hub-model-search";
import { useDebouncedValue } from "@/hooks";
import { type ReactElement, useMemo, useRef } from "react";

// HF pipeline filter for embedding models; matches the backend's
// is_embedding_model signals (sentence-similarity / feature-extraction).
const EMBEDDING_TASKS: readonly PipelineType[] = [
  "sentence-similarity",
  "feature-extraction",
];

type EmbeddingModelComboboxProps = {
  value: string;
  /** Fires on typing, selection, and Enter with the current text. */
  onChange: (value: string) => void;
  accessToken?: string;
  disabled?: boolean;
  placeholder?: string;
  ariaLabel?: string;
  className?: string;
};

export function EmbeddingModelCombobox({
  value,
  onChange,
  accessToken,
  disabled,
  placeholder,
  ariaLabel,
  className,
}: EmbeddingModelComboboxProps): ReactElement {
  const selectingRef = useRef(false);
  const anchorRef = useRef<HTMLDivElement>(null);
  // Fully controlled: the parent updates value on every keystroke, so the
  // prop itself is the search query.
  const debouncedQuery = useDebouncedValue(value);

  const { results, isLoading } = useHubModelSearch(debouncedQuery, {
    task: EMBEDDING_TASKS,
    accessToken,
    excludeGguf: true,
    enabled: !disabled,
    // Curated unsloth listing when empty (the global top-downloads page holds
    // no unsloth mirrors to float); a typed query searches the whole Hub.
    ownerScope: debouncedQuery.trim() ? "all" : "unsloth",
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
        const typed = event.target.value.trim();
        if (typed) {
          onChange(typed);
        } else if (items.length > 0) {
          onChange(items[0]);
        }
      }}
    >
      <Combobox
        items={items}
        filteredItems={items}
        filter={null}
        value={value.trim() ? value : null}
        onValueChange={(next) => onChange(next ?? "")}
        onInputValueChange={(next) => {
          if (selectingRef.current) {
            selectingRef.current = false;
            return;
          }
          onChange(next);
        }}
        itemToStringValue={(item) => item}
        autoHighlight={true}
      >
        <ComboboxInput
          className="h-8 w-full font-mono [&_input]:text-[11px]"
          placeholder={placeholder}
          aria-label={ariaLabel}
          disabled={disabled}
        />
        <ComboboxContent anchor={anchorRef}>
          {isLoading ? (
            <div className="flex items-center gap-2 px-2 py-3 text-xs text-muted-foreground">
              <Spinner className="size-3.5" />
              Searching...
            </div>
          ) : (
            <ComboboxEmpty>No embedding models found</ComboboxEmpty>
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
                <span className="truncate font-mono text-[11px]">{id}</span>
              </ComboboxItem>
            )}
          </ComboboxList>
        </ComboboxContent>
      </Combobox>
    </div>
  );
}
