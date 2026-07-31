// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Spinner } from "@/components/ui/spinner";
import { useT } from "@/i18n";
import { cn } from "@/lib/utils";
import { RefreshIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import type { Ref } from "react";
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

export function RetryButton({ onRetry }: { onRetry: () => void }) {
  const t = useT();
  return (
    <button
      type="button"
      onClick={onRetry}
      className={cn(
        "mt-1 inline-flex items-center gap-1.5 rounded-full border border-border/70 px-3 py-1 text-ui-11 font-medium text-foreground transition-colors hover:bg-foreground/[0.05]",
        PICKER_OPTION_FOCUS_VISIBLE_CLASS,
      )}
    >
      <HugeiconsIcon icon={RefreshIcon} strokeWidth={1.75} className="size-3" />
      {t("picker.retry")}
    </button>
  );
}

function PickerLoadMoreButton({
  disabled,
  onLoadMore,
}: {
  disabled: boolean;
  onLoadMore: () => void;
}) {
  const t = useT();
  return (
    <div className="flex justify-center py-2">
      <button
        type="button"
        disabled={disabled}
        onClick={onLoadMore}
        className={cn(
          "inline-flex h-7 items-center rounded-full border border-border/70 px-3 text-ui-11 font-medium text-foreground transition-colors hover:bg-foreground/[0.05] disabled:cursor-not-allowed disabled:opacity-50",
          PICKER_OPTION_FOCUS_VISIBLE_CLASS,
        )}
      >
        {t("picker.loadMore")}
      </button>
    </div>
  );
}

export function PickerHubPaginationFooter({
  isLoading,
  isLoadingMore,
  onLoadMore,
  sentinelRef,
  showLoadMore,
}: {
  isLoading: boolean;
  isLoadingMore: boolean;
  onLoadMore: () => void;
  sentinelRef: Ref<HTMLDivElement>;
  showLoadMore: boolean;
}) {
  return (
    <>
      <div ref={sentinelRef} className="h-px" />
      {isLoadingMore && (
        <div className="flex items-center justify-center py-2">
          <Spinner className="size-3.5 text-muted-foreground" />
        </div>
      )}
      {showLoadMore && (
        <PickerLoadMoreButton
          disabled={isLoading || isLoadingMore}
          onLoadMore={onLoadMore}
        />
      )}
    </>
  );
}

export function PickerSearchError({
  title,
  detail,
  onRetry,
  compact = false,
}: {
  title: string;
  detail: string;
  onRetry: () => void;
  compact?: boolean;
}) {
  return (
    <div
      role="alert"
      className={
        compact
          ? "flex items-start justify-between gap-3 border-t border-border/60 px-2.5 py-2"
          : "flex flex-col items-center gap-1.5 px-4 py-8 text-center"
      }
    >
      <div className={cn("min-w-0", compact && "text-left")}>
        <p className="text-ui-12p5 font-medium text-foreground">{title}</p>
        <p className="text-ui-11 leading-snug text-muted-foreground">
          {detail}
        </p>
      </div>
      <RetryButton onRetry={onRetry} />
    </div>
  );
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
      className="hub-menu-trigger hub-tab-toggle relative inline-flex h-8 w-full select-none items-center rounded-full p-0.5"
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
      <span
        aria-hidden="true"
        className={cn(
          "hub-tab-toggle-pill pointer-events-none absolute left-0.5 top-0.5 bottom-0.5 w-[calc(50%-2px)] rounded-full transition-transform duration-200 ease-out",
          tab === second.value ? "translate-x-full" : "translate-x-0",
        )}
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
              "relative z-10 inline-flex h-7 flex-1 select-none items-center justify-center rounded-full px-3 text-ui-12p5 transition-colors",
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
