// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Pure rules for the source toggle (Hub models / Fine-tuned / Connected).

export type SourceTab = { value: string; label: string };

/** Local models (LM Studio, Ollama, custom folders) are not fine-tuned; they
 * live in the Hub tab's Downloaded / Custom sections. */
export function isFineTunedSource(source?: string): boolean {
  return source !== "local";
}

/** Build the source tabs. Fine-tuned models live as a section in the Hub tab's
 * On Device view, so there is no Fine-tuned tab. Connected only appears with
 * external providers; with none, the lone Hub tab hides its own toggle. */
export function buildSourceTabs(opts: { hasExternal: boolean }): SourceTab[] {
  const list: SourceTab[] = [{ value: "hub", label: "Hub models" }];
  if (opts.hasExternal) {
    list.push({ value: "external", label: "Connected" });
  }
  return list;
}
