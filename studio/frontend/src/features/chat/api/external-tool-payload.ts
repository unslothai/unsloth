// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export interface ExternalToolSelection {
  studioToolExecution: boolean;
  webSearch: boolean;
  webFetch: boolean;
  codeExecution: boolean;
  imageGeneration: boolean;
}

/** Preserve hosted tools when adding Studio tools. */
export function buildExternalEnabledTools({
  studioToolExecution,
  webSearch,
  webFetch,
  codeExecution,
  imageGeneration,
}: ExternalToolSelection): string[] {
  return Array.from(
    new Set([
      ...(webSearch ? ["web_search"] : []),
      ...(webFetch ? ["web_fetch"] : []),
      ...(codeExecution
        ? studioToolExecution
          ? ["python", "terminal"]
          : ["code_execution"]
        : []),
      ...(imageGeneration ? ["image_generation"] : []),
    ]),
  );
}
