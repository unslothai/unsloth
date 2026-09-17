// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Light mode uses a soft border, dark mode background fill only. The border stays
* transparent so box size matches across themes. */
export function selectableOptionStateClassName(selected: boolean): string {
  return selected
    ? "border-ring-strong/50 bg-primary/5 dark:border-transparent dark:bg-emerald-200/10"
    : "border-border/50 bg-muted/40 hover:border-foreground/20 hover:bg-muted/60 dark:border-transparent dark:bg-white/[0.05] dark:hover:bg-white/[0.09]";
}
