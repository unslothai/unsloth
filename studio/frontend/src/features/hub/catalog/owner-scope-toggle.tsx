// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { cn } from "@/lib/utils";

export type OwnerScope = "unsloth" | "all";

const OPTIONS: { value: OwnerScope; label: string }[] = [
  { value: "unsloth", label: "Unsloth" },
  { value: "all", label: "All" },
];

/**
 * Compact "Unsloth / All" segmented control. Lives in the list header beside the
 * view-mode toggle so it only appears while browsing a model list (never on the
 * hub homepage feed). Switches the discover scope between the unsloth org and
 * the whole Hub.
 *
 * Buttons size to their own label (like the HTTP/Xet transport toggle) rather
 * than splitting the width evenly — "Unsloth" and "All" are very different
 * lengths, so equal halves would crowd one and leave a gap in the other.
 */
export function OwnerScopeToggle({
  value,
  onChange,
}: {
  value: OwnerScope;
  onChange: (value: OwnerScope) => void;
}) {
  return (
    <div
      className="hub-tab-toggle inline-flex h-8 shrink-0 items-center rounded-full"
      role="radiogroup"
      aria-label="Publisher scope"
    >
      {OPTIONS.map((opt) => {
        const active = value === opt.value;
        return (
          <button
            key={opt.value}
            type="button"
            role="radio"
            aria-checked={active}
            onClick={() => onChange(opt.value)}
            className={cn(
              "inline-flex h-8 items-center justify-center rounded-full px-3.5 text-[11.5px] transition-colors",
              active
                ? "hub-tab-toggle-pill text-foreground"
                : "cursor-pointer text-muted-foreground hover:text-foreground",
            )}
          >
            {opt.label}
          </button>
        );
      })}
    </div>
  );
}
