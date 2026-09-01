// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { LoraModelOption } from "./types";

export type LocalGgufKind = "direct" | "variants" | null;

/** Resolve local GGUF interaction semantics without letting repo-name heuristics override an
 *  explicit one-artifact inventory source such as Ollama. */
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
