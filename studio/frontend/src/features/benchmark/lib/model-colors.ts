// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useTheme } from "@/features/settings";

// The Benchmarks page's validated categorical order, minus its neutral. Assigned by first
// appearance and never cycled: an eighth model is left to the history list.
const MODEL_COLORS = {
  light: [
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#4a3aa7",
  ],
  dark: [
    "#3987e5",
    "#d95926",
    "#199e70",
    "#c98500",
    "#d55181",
    "#008300",
    "#9085e9",
  ],
};

export function useModelPalette(): string[] {
  const { resolved } = useTheme();
  return MODEL_COLORS[resolved === "dark" ? "dark" : "light"];
}
