// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { cn } from "@/lib/utils";
import { RefreshIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";

const HF_AUTH_ERROR_RE =
  /\b401\b|unauthorized|invalid.*token|invalid.*credential|authentication|forbidden|\b403\b/i;

export function isHfAuthError(message: string | null | undefined): boolean {
  return !!message && HF_AUTH_ERROR_RE.test(message);
}

/**
 * Heuristic for "this query looks like a filesystem path, not a Hugging Face
 * id" so the Train pickers can offer a "Use as local path" affordance. Shared
 * by the model and dataset pickers.
 */
export function looksLikeLocalPath(query: string): boolean {
  return (
    /^[/.~]/.test(query) ||
    query.includes("\\") ||
    /^[a-zA-Z]:[\\/]/.test(query)
  );
}

/**
 * Compact retry affordance for the Train pickers' Hugging Face error state,
 * shared by the model and dataset pickers. Calls the search hook's retry(),
 * which restarts the listing from the first page.
 */
export function RetryButton({ onRetry }: { onRetry: () => void }) {
  return (
    <button
      type="button"
      onClick={onRetry}
      className="mt-1 inline-flex items-center gap-1.5 rounded-full border border-border/70 px-3 py-1 text-[11px] font-medium text-foreground transition-colors hover:bg-foreground/[0.05]"
    >
      <HugeiconsIcon icon={RefreshIcon} strokeWidth={1.75} className="size-3" />
      Retry
    </button>
  );
}

export type PickerTab = "device" | "hub";

export const PICKER_TABS = [
  { value: "device", label: "On Device" },
  { value: "hub", label: "Hugging Face" },
] as const satisfies readonly [
  { value: PickerTab; label: string },
  { value: PickerTab; label: string },
];

/**
 * Two-option pill toggle shared by the Train page's model + dataset pickers.
 * Generic over the tab key string so callers can use their own enum.
 */
export function PickerTabToggle<T extends string>({
  tab,
  options,
  onTabChange,
}: {
  tab: T;
  options: readonly [
    { value: T; label: string },
    { value: T; label: string },
  ];
  onTabChange: (tab: T) => void;
}) {
  const [, second] = options;
  return (
    <div
      role="tablist"
      className="menu-trigger tab-toggle relative inline-flex h-8 w-full select-none items-center rounded-full p-0.5"
    >
      <span
        aria-hidden="true"
        className={cn(
          "tab-toggle-pill pointer-events-none absolute left-0.5 top-0.5 bottom-0.5 w-[calc(50%-2px)] rounded-full transition-transform duration-200 ease-out",
          tab === second.value ? "translate-x-full" : "translate-x-0",
        )}
      />
      {options.map((entry) => (
        <button
          key={entry.value}
          type="button"
          role="tab"
          aria-selected={tab === entry.value}
          onClick={() => onTabChange(entry.value)}
          className={cn(
            "relative z-10 inline-flex h-7 flex-1 select-none items-center justify-center rounded-full px-3 text-[12.5px] transition-colors",
            tab === entry.value
              ? "text-foreground"
              : "text-muted-foreground hover:text-foreground",
          )}
        >
          {entry.label}
        </button>
      ))}
    </div>
  );
}
