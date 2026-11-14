// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import { cn } from "@/lib/utils";
import { Search01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
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
  onToggle,
  onSelectAll,
  onClear,
}: {
  providerType: string;
  panelKey: string;
  /** Rows after the search. */
  models: string[];
  totalCount: number;
  selectedIds: string[];
  query: string;
  onQueryChange: (query: string) => void;
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
  const filtered = query.trim().length > 0;
  return (
    <>
      <div className="flex flex-wrap items-center gap-2">
        <div className="relative min-w-40 flex-1">
          <HugeiconsIcon
            icon={Search01Icon}
            className="pointer-events-none absolute top-1/2 left-3 size-4 -translate-y-1/2 text-muted-foreground"
            aria-hidden="true"
          />
          <Input
            id={`provider-model-search-${panelKey}`}
            value={query}
            onChange={(event) => onQueryChange(event.target.value)}
            placeholder={`Search ${totalCount} models`}
            aria-label="Search models"
            className="h-control pl-9"
          />
        </div>
        <div className="flex items-center justify-end gap-1">
          <Button
            type="button"
            variant="ghost"
            size="sm"
            className="h-8 rounded-full px-3 text-xs font-medium text-muted-foreground hover:text-foreground"
            onClick={onSelectAll}
          >
            Select all
          </Button>
          <Button
            type="button"
            variant="ghost"
            size="sm"
            className="h-8 rounded-full px-3 text-xs font-medium text-muted-foreground hover:text-foreground"
            onClick={onClear}
          >
            Clear
          </Button>
        </div>
      </div>
      <ul
        className={cn(
          // Rows hang into the gutter so their hover pill lines text up with the page; scrolls past 10 rows.
          "-mx-3 flex flex-col gap-0.5",
          rows.length > 10 &&
            "max-h-[calc(var(--spacing)*90+10px)] overflow-y-auto",
        )}
      >
        {rows.length === 0 ? (
          <li className="px-3 py-2.5 text-xs text-muted-foreground">
            No matching models
          </li>
        ) : (
          rows.map((row, index) => (
            <li key={row.id}>
              <label
                htmlFor={`provider-model-${panelKey}-${index}`}
                className="flex min-h-9 cursor-pointer items-center gap-2.5 rounded-full px-3 py-1.5 transition-colors hover:bg-muted/45"
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
      {filtered && rows.length > 0 ? (
        <p className="text-xs text-muted-foreground">
          {`Showing ${models.length} of ${totalCount} models`}
        </p>
      ) : null}
    </>
  );
}
