// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Cancel01Icon, Clock01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";

/**
 * Recent-searches suggestion panel beneath the search field. Presentational only
 * (parent owns visibility and store). Suppresses its own mousedown so clicks fire
 * without blurring the focused input and closing the panel.
 */
export function RecentSearches({
  searches,
  onSelect,
  onRemove,
  onClear,
}: {
  searches: string[];
  onSelect: (query: string) => void;
  onRemove: (query: string) => void;
  onClear: () => void;
}) {
  if (searches.length === 0) {
    return null;
  }
  return (
    <div
      className="hub-recent-panel menu-soft-surface absolute inset-x-0 top-full z-50 mt-2 overflow-hidden rounded-[16px] p-1.5"
      aria-label="Recent searches"
      onMouseDown={(event) => event.preventDefault()}
    >
      <div className="flex items-center justify-between gap-2 px-2.5 pb-1.5 pt-1">
        <span className="text-[11px] font-semibold uppercase tracking-[0.04em] text-muted-foreground/70">
          Recent searches
        </span>
        <button
          type="button"
          onClick={onClear}
          className="hub-recent-clear rounded-full px-2 py-0.5 text-[11.5px] font-medium text-muted-foreground transition-colors hover:text-foreground"
        >
          Clear all
        </button>
      </div>
      <ul className="flex flex-col">
        {searches.map((query) => (
          <li key={query}>
            <div className="hub-recent-item group/recent flex items-center rounded-[10px] pr-1.5">
              <button
                type="button"
                onClick={() => onSelect(query)}
                className="flex min-w-0 flex-1 items-center gap-2.5 rounded-l-[10px] py-2 pl-2.5 text-left"
              >
                <HugeiconsIcon
                  icon={Clock01Icon}
                  strokeWidth={1.75}
                  className="size-4 shrink-0 text-muted-foreground/70"
                />
                <span className="truncate text-[13px] text-foreground">
                  {query}
                </span>
              </button>
              <button
                type="button"
                aria-label={`Remove “${query}” from recent searches`}
                onClick={() => onRemove(query)}
                className="hub-recent-remove ml-1 inline-flex size-6 shrink-0 items-center justify-center rounded-full text-muted-foreground opacity-0 transition-[opacity,color,background-color] hover:text-foreground focus-visible:opacity-100 group-hover/recent:opacity-100"
              >
                <HugeiconsIcon
                  icon={Cancel01Icon}
                  strokeWidth={1.75}
                  className="size-3.5"
                />
              </button>
            </div>
          </li>
        ))}
      </ul>
    </div>
  );
}
