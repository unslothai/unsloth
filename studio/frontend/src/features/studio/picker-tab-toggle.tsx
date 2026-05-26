// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { isHuggingFaceOffline } from "@/lib/network";
import { cn } from "@/lib/utils";
import { RefreshIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
export { looksLikeLocalPath } from "@/lib/local-path";

const HF_AUTH_ERROR_RE =
  /\b401\b|unauthorized|invalid.*token|invalid.*credential|authentication|forbidden|\b403\b/i;

export function isHfAuthError(message: string | null | undefined): boolean {
  return !!message && HF_AUTH_ERROR_RE.test(message);
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

export const PICKER_TAB = {
  DEVICE: "device",
  HUB: "hub",
} as const;

export const PICKER_TAB_VALUES = [PICKER_TAB.DEVICE, PICKER_TAB.HUB] as const;
export type PickerTab = (typeof PICKER_TAB_VALUES)[number];

export const PICKER_TABS = [
  { value: PICKER_TAB.DEVICE, label: "On Device" },
  { value: PICKER_TAB.HUB, label: "Hugging Face" },
] as const satisfies readonly [
  { value: PickerTab; label: string },
  { value: PickerTab; label: string },
];

function isPickerTab(value: unknown): value is PickerTab {
  return (
    typeof value === "string" &&
    (PICKER_TAB_VALUES as readonly string[]).includes(value)
  );
}

export function readPickerTabPreference(storageKey: string): PickerTab | null {
  if (typeof window === "undefined") return null;
  try {
    const value = window.localStorage.getItem(storageKey);
    return isPickerTab(value) ? value : null;
  } catch {
    return null;
  }
}

export function defaultPickerTab(): PickerTab {
  return isHuggingFaceOffline() ? PICKER_TAB.DEVICE : PICKER_TAB.HUB;
}

export function writePickerTabPreference(
  storageKey: string,
  tab: PickerTab,
): void {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(storageKey, tab);
  } catch {
    void 0;
  }
}

export function pickerTabId(idBase: string, value: string): string {
  return `${idBase}-tab-${value}`;
}

/**
 * Two-option pill toggle shared by the Train page's model + dataset pickers.
 * Generic over the tab key string so callers can use their own enum.
 *
 * `idBase` and `panelId` wire up the WAI-ARIA tablist/tabpanel contract: each
 * tab gets a stable id derived from `idBase` and points at the panel via
 * `aria-controls`, so screen readers announce the relationship.
 */
export function PickerTabToggle<T extends string>({
  tab,
  options,
  onTabChange,
  idBase,
  panelId,
}: {
  tab: T;
  options: readonly [
    { value: T; label: string },
    { value: T; label: string },
  ];
  onTabChange: (tab: T) => void;
  idBase: string;
  panelId: string;
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
              "relative z-10 inline-flex h-7 flex-1 select-none items-center justify-center rounded-full px-3 text-[12.5px] transition-colors",
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
