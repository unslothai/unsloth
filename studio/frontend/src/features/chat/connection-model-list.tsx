// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import { cn } from "@/lib/utils";
import { useMemo, useSyncExternalStore } from "react";
import {
  modelCatalogVersion,
  resolveModelCatalogEntry,
  subscribeModelCatalog,
} from "./model-catalog";
import {
  providerModelDisplayName,
  useModelDisplayNamesVersion,
} from "./model-display-names";

export function formatContextWindow(tokens: number): string {
  if (tokens >= 1_000_000) {
    return `${(tokens / 1_000_000).toFixed(1).replace(/\.0$/, "")}M`;
  }
  return tokens >= 1_000 ? `${Math.round(tokens / 1_000)}K` : String(tokens);
}

export function modelFacts(providerType: string, modelId: string): string {
  const entry = resolveModelCatalogEntry(providerType, modelId);
  if (!entry) return "";
  const facts: string[] = [];
  if (entry.contextLength) facts.push(formatContextWindow(entry.contextLength));
  if (entry.inputModalities?.includes("image")) facts.push("Vision");
  if (entry.reasoning) facts.push("Reasoning");
  return facts.join(" · ");
}

export function ConnectionModelList({
  providerType,
  panelKey,
  models,
  totalCount,
  selectedIds,
  query,
  onQueryChange,
  selectedOnly,
  onSelectedOnlyChange,
  onToggle,
  onSelectAll,
  onClear,
}: {
  providerType: string;
  panelKey: string;
  /** Rows after the search and the Selected filter. */
  models: string[];
  totalCount: number;
  selectedIds: string[];
  query: string;
  onQueryChange: (query: string) => void;
  selectedOnly: boolean;
  onSelectedOnlyChange: (selectedOnly: boolean) => void;
  onToggle: (modelId: string) => void;
  onSelectAll: () => void;
  onClear: () => void;
}) {
  // Subscribed so the names and facts follow a catalog refresh or a fresh listing.
  useSyncExternalStore(subscribeModelCatalog, modelCatalogVersion);
  useModelDisplayNamesVersion();
  const selected = useMemo(() => new Set(selectedIds), [selectedIds]);
  const rows = models.map((model) => ({
    id: model,
    name: providerModelDisplayName(providerType, model),
    facts: modelFacts(providerType, model),
  }));
  const filtered = query.trim().length > 0 || selectedOnly;
  return (
    <>
      <div className="grid grid-cols-[minmax(90px,auto)_minmax(0,1fr)_auto] items-center gap-3 @max-[520px]:grid-cols-1">
        <span className="whitespace-nowrap text-xs font-medium text-muted-foreground">
          {filtered
            ? `${models.length} of ${totalCount} models`
            : `${totalCount} models`}
        </span>
        <Input
          id={`provider-model-search-${panelKey}`}
          type="search"
          value={query}
          onChange={(event) => onQueryChange(event.target.value)}
          placeholder="Search"
          aria-label="Search models"
          className="h-8 w-full bg-background/55 text-xs placeholder:text-muted-foreground/65 focus-visible:border-border focus-visible:ring-0"
        />
        <div className="flex items-center justify-end gap-1">
          <Button
            type="button"
            variant="ghost"
            size="sm"
            aria-pressed={selectedOnly}
            className={cn(
              "h-8 px-2 text-xs font-medium text-foreground/80 hover:bg-muted/45",
              selectedOnly && "bg-muted/60 text-foreground",
            )}
            onClick={() => onSelectedOnlyChange(!selectedOnly)}
            title="Show only the models you ticked"
          >
            Selected
          </Button>
          <Button
            type="button"
            variant="ghost"
            size="sm"
            className="h-8 px-2 text-xs font-medium text-foreground/80 hover:bg-muted/45"
            onClick={onSelectAll}
          >
            Select all
          </Button>
          <Button
            type="button"
            variant="ghost"
            size="sm"
            className="h-8 px-2 text-xs font-medium text-foreground/80 hover:bg-muted/45"
            onClick={onClear}
          >
            Clear
          </Button>
        </div>
      </div>
      <ul
        className={cn(
          // Full-bleed rows of the Models card; the list scrolls only past 10 rows.
          "-mx-4 divide-y divide-border/60 border-border/60 border-y last:-mb-4 last:border-b-0",
          rows.length > 10 &&
            "max-h-[calc(var(--spacing)*90+10px)] overflow-y-auto",
        )}
      >
        {rows.length === 0 ? (
          <li className="px-4 py-2.5 text-xs text-muted-foreground">
            {selectedOnly && !query.trim()
              ? "No models selected"
              : "No matching models"}
          </li>
        ) : (
          rows.map((row, index) => (
            <li key={row.id}>
              <label
                htmlFor={`provider-model-${panelKey}-${index}`}
                className="flex min-h-9 cursor-pointer items-center gap-2.5 px-4 py-1.5 hover:bg-muted/35"
              >
                <Checkbox
                  id={`provider-model-${panelKey}-${index}`}
                  checked={selected.has(row.id)}
                  onCheckedChange={() => onToggle(row.id)}
                />
                <span className="min-w-0 flex-1 break-all text-sm leading-tight">
                  {row.name}
                  {row.name === row.id ? null : (
                    <span className="ml-2 text-xs text-muted-foreground">
                      {row.id}
                    </span>
                  )}
                </span>
                {row.facts ? (
                  <span className="shrink-0 whitespace-nowrap text-ui-11 text-muted-foreground/80">
                    {row.facts}
                  </span>
                ) : null}
              </label>
            </li>
          ))
        )}
      </ul>
    </>
  );
}
