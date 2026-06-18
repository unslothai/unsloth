// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Pure rules for the source toggle (Hub models / Fine-tuned / Connected).

export type SourceTab = { value: string; label: string };

/** Local models (LM Studio, Ollama, custom folders) are not fine-tuned; they
 * live in the Hub tab's Downloaded / Custom sections. */
export function isFineTunedSource(source?: string): boolean {
  return source !== "local";
}

/** Build the source tabs. Fine-tuned and Connected models live as sections in
 * the Hub tab's toggle, so Hub is the only source and its strip stays hidden. */
export function buildSourceTabs(): SourceTab[] {
  return [{ value: "hub", label: "Hub models" }];
}
