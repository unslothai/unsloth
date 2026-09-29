// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Light mode uses a soft border, dark mode background fill only. The border stays
* transparent so box size matches across themes. */
export function selectableOptionStateClassName(selected: boolean): string {
  return selected
    ? "border-ring-strong/50 bg-primary/5 dark:border-transparent dark:bg-emerald-200/10"
    : "border-border/50 bg-muted/40 hover:border-[color-mix(in_oklab,var(--foreground)_calc(20%*var(--contrast-edge-gain,1)),transparent)] hover:bg-muted/60 dark:border-transparent dark:bg-[rgb(255_255_255_/_calc(0.05*var(--contrast-wash-gain,1)))] dark:hover:bg-[rgb(255_255_255_/_calc(0.09*var(--contrast-wash-gain,1)))]";
}
