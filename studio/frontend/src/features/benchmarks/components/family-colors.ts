// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useTheme } from "@/features/settings";
import type { Family } from "../lib/bench-math";

// Validated categorical order (adjacent-pair CVD ΔE ≥ 8 in both modes), one slot per
// family and fixed: a mode keeps its colour whichever rows a run has. Off is the neutral.
const SERIES: Record<"light" | "dark", Record<Family, string>> = {
  light: {
    mtp: "#2a78d6",
    "mtp+ngram": "#eb6834",
    ngram: "#1baf7a",
    auto: "#eda100",
    dspark: "#e87ba4",
    dflash: "#008300",
    other: "#4a3aa7",
    off: "#9a9893",
  },
  dark: {
    mtp: "#3987e5",
    "mtp+ngram": "#d95926",
    ngram: "#199e70",
    auto: "#c98500",
    dspark: "#d55181",
    dflash: "#008300",
    other: "#9085e9",
    off: "#6f6e69",
  },
};

export function useFamilyColors(): Record<Family, string> {
  const { resolved } = useTheme();
  return SERIES[resolved === "dark" ? "dark" : "light"];
}
