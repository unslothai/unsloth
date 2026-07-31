// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export function selectableOptionStateClassName(selected: boolean): string {
  return selected
    ? "border-ring-strong bg-primary/5"
    : "border-border hover:border-foreground/20";
}
