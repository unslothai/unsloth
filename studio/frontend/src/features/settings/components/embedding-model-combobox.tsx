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
import { type ReactElement, useMemo, useRef, useState } from "react";

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
  const anchorRef = useRef<HTMLDivElement>(null);
  // Only text the user typed is a query. A saved or selected value searches for
  // itself, which leaves the list holding that one model instead of the others.
  const [typed, setTyped] = useState<string | null>(null);
  // Anything the parent set (selection, reset, reload) clears the query. Saving
  // writes the same string back, which this cannot see, so closing clears it too.
  const query = typed !== null && typed === value ? typed : "";
  const debouncedQuery = useDebouncedValue(query);

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
        onOpenChange={(open) => {
          // The search lasts as long as the list is open, as the model picker
          // in Settings -> Agents does.
          if (!open) setTyped(null);
        }}
        value={value.trim() ? value : null}
        onValueChange={(next) => onChange(next ?? "")}
        onInputValueChange={(next, details) => {
          // The settings load and a pick both write the input too, and arrive
          // with another reason; only typing is a search.
          if (details.reason !== "input-change") {
            setTyped(null);
            return;
          }
          setTyped(next);
          onChange(next);
        }}
        itemToStringValue={(item) => item}
        autoHighlight={true}
      >
        <ComboboxInput
          className="h-8 w-full font-mono [&_input]:text-ui-11"
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
              <ComboboxItem key={id} value={id}>
                <span className="truncate font-mono text-ui-11">{id}</span>
              </ComboboxItem>
            )}
          </ComboboxList>
        </ComboboxContent>
      </Combobox>
    </div>
  );
}
