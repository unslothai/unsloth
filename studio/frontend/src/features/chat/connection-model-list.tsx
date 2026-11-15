// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import { ChevronDownStandardIcon } from "@/lib/chevron-icons";
import { cn } from "@/lib/utils";
import { Search01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useMemo, useState, useSyncExternalStore } from "react";
import { createPortal } from "react-dom";
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

const INPUT_LABELS: Record<string, string> = {
  text: "Text",
  image: "Image",
  audio: "Audio",
  video: "Video",
  pdf: "PDF",
};

/** Label and value pairs for a model's hover card; empty when the catalog does not know it. */
export function modelDetails(
  providerType: string,
  modelId: string,
): [string, string][] {
  const entry = resolveModelCatalogEntry(providerType, modelId);
  if (!entry) return [];
  const details: [string, string][] = [];
  const inputs = (entry.inputModalities ?? []).map(
    (input) => INPUT_LABELS[input] ?? input,
  );
  if (inputs.length > 0) details.push(["Inputs", inputs.join(", ")]);
  const efforts = entry.efforts
    .filter((effort) => effort !== "none")
    .map((effort) =>
      effort === "xhigh"
        ? "XHigh"
        : effort.charAt(0).toUpperCase() + effort.slice(1),
    );
  details.push([
    "Reasoning",
    efforts.length > 0 ? efforts.join(", ") : entry.reasoning ? "Yes" : "No",
  ]);
  if (entry.contextLength) {
    details.push(["Context", entry.contextLength.toLocaleString("en-US")]);
  }
  return details;
}

/** Fades the list's top or bottom edge while more rows lie beyond it. */
function fadeScrollEdges(list: HTMLElement | null) {
  if (!list) return;
  const update = () => {
    const above = list.scrollTop > 1;
    const below = list.scrollTop + list.clientHeight < list.scrollHeight - 1;
    const mask =
      above || below
        ? `linear-gradient(to bottom, ${above ? "transparent" : "#000"} 0, #000 32px, #000 calc(100% - 32px), ${below ? "transparent" : "#000"} 100%)`
        : "";
    list.style.maskImage = mask;
    list.style.setProperty("-webkit-mask-image", mask);
  };
  update();
  list.addEventListener("scroll", update, { passive: true });
  // Rows come and go with the search and the Older models fold.
  const observer = new MutationObserver(update);
  observer.observe(list, { childList: true });
  return () => {
    list.removeEventListener("scroll", update);
    observer.disconnect();
  };
}

export function ConnectionModelList({
  providerType,
  panelKey,
  models,
  totalCount,
  selectedIds,
  query,
  onQueryChange,
  olderCount,
  showOlder,
  onShowOlderChange,
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
  /** Models released over a year ago; folded away unless ticked, searched for or shown. */
  olderCount: number;
  showOlder: boolean;
  onShowOlderChange: (showOlder: boolean) => void;
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
    details: modelDetails(providerType, model),
  }));
  const filtered = query.trim().length > 0;
  const [card, setCard] = useState<{
    name: string;
    details: [string, string][];
    left: number;
    top: number;
  } | null>(null);
  // Beside the name, 24px clear of the text.
  const showCard = (
    row: { name: string; details: [string, string][] },
    label: HTMLElement,
  ) => {
    if (row.details.length === 0) return setCard(null);
    const text = label
      .querySelector("[data-model-name]")
      ?.getBoundingClientRect();
    if (!text) return;
    const width = 190;
    setCard({
      name: row.name,
      details: row.details,
      left: Math.max(
        8,
        Math.min(text.right + 24, window.innerWidth - width - 8),
      ),
      top: text.top - 8,
    });
  };
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
      {card
        ? createPortal(
            // One card for the whole list, drawn outside it and deaf to the pointer: a tooltip per row
            // sat under the cursor and fought the row for hover.
            <div
              role="tooltip"
              style={{ left: card.left, top: card.top }}
              className="tooltip-rich pointer-events-none fixed z-[999999] flex w-[calc(190px*var(--ui-space-scale,1))] flex-col gap-1 rounded-xl px-3 py-2 text-ui-11"
            >
              <div className="flex min-w-0 items-center gap-3">
                <span className="shrink-0 text-muted-foreground">Model</span>
                <span className="ml-auto min-w-0 truncate text-right">
                  {card.name}
                </span>
              </div>
              {card.details.map(([name, value]) => (
                <div key={name} className="flex min-w-0 items-start gap-3">
                  <span className="shrink-0 text-muted-foreground">{name}</span>
                  <span className="ml-auto min-w-0 text-right">{value}</span>
                </div>
              ))}
            </div>,
            document.body,
          )
        : null}
      <div
        className="overflow-hidden rounded-2xl border border-border/60 py-1 pl-1"
        onScrollCapture={() => setCard(null)}
      >
        <ul
          ref={fadeScrollEdges}
          className={cn(
            // Scrolls past 10 rows.
            "hover-scrollbar flex flex-col gap-0.5 pr-1",
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
                  onPointerEnter={(event) => showCard(row, event.currentTarget)}
                  onPointerLeave={() => setCard(null)}
                  className="flex min-h-9 cursor-pointer items-center gap-2.5 rounded-full px-3 py-1.5 transition-colors hover:bg-accent"
                >
                  <Checkbox
                    id={`provider-model-${panelKey}-${index}`}
                    // Solid fill and a firmer edge: the default box vanishes into the row's hover pill.
                    className="border-muted-foreground/60 bg-background dark:bg-background"
                    checked={selected.has(row.id)}
                    onCheckedChange={() => onToggle(row.id)}
                  />
                  <span
                    data-model-name={true}
                    className="min-w-0 break-all text-sm leading-tight"
                  >
                    {row.name}
                    {row.name === row.id ? null : (
                      <span className="ml-2 text-xs text-muted-foreground">
                        {row.id}
                      </span>
                    )}
                  </span>
                </label>
              </li>
            ))
          )}
        </ul>
      </div>
      {olderCount > 0 && !filtered ? (
        <button
          type="button"
          aria-expanded={showOlder}
          onClick={() => onShowOlderChange(!showOlder)}
          className="flex items-center gap-1 self-start rounded-md py-1 text-xs font-medium text-muted-foreground transition-colors hover:text-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
        >
          {`Older models (${olderCount})`}
          <HugeiconsIcon
            icon={ChevronDownStandardIcon}
            className={cn(
              "size-3.5 transition-transform",
              !showOlder && "-rotate-90",
            )}
          />
        </button>
      ) : null}
      {filtered && rows.length > 0 ? (
        <p className="text-xs text-muted-foreground">
          {`Showing ${models.length} of ${totalCount} models`}
        </p>
      ) : null}
    </>
  );
}
