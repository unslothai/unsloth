// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Card and Run button shared by every Benchmarks tab, so each tab reads as one page.

export const BENCH_CARD =
  "corner-squircle rounded-3xl bg-card ring-1 ring-[color-mix(in_oklab,var(--foreground)_calc(10%*var(--contrast-edge-gain,1)),transparent)]";

export const RUN_BUTTON =
  "h-11 w-full justify-center rounded-xl text-ui-13p5 font-semibold tracking-tight bg-primary text-primary-foreground shadow-sm hover:bg-primary/90 disabled:bg-[color-mix(in_oklab,var(--foreground)_calc(8%*var(--contrast-wash-gain,1)),transparent)] disabled:text-muted-foreground disabled:shadow-none dark:disabled:bg-[rgb(255_255_255_/_calc(0.06*var(--contrast-wash-gain,1)))] transition-colors duration-200";
