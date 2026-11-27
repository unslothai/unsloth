// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { LoraModelOption } from "./types";

export type LocalGgufKind = "direct" | "variants" | null;

/** Repo-name heuristics never override an explicit one-artifact inventory source like Ollama. */
export function localGgufKindFor(
  option: Pick<LoraModelOption, "source" | "isDirectGguf">,
  looksLikeVariantRepo: boolean,
): LocalGgufKind {
  if (option.source !== "local") {
    return null;
  }
  if (option.isDirectGguf === true) {
    return "direct";
  }
  return looksLikeVariantRepo ? "variants" : null;
}
