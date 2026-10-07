// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** `code_execution` runs in the provider's sandbox; `python`/`terminal` run on this machine.
 *  A connection with its own sandbox keeps it and never falls back to local execution. */

export interface CodeToolPlacementInput {
  codeToolsEnabled: boolean;
  /** This provider AND this model expose the provider's own code sandbox. */
  hostedCodeExecutionForThisTurn: boolean;
  /** This provider type ships a code sandbox at all, model aside. */
  providerHostsCodeExecution: boolean;
}

export interface CodeToolNames {
  /** Unsloth tool names, executed on this machine by the Unsloth tool loop. */
  local: string[];
  /** Provider builtin names, executed and billed by the provider. */
  hosted: string[];
}

export function selectCodeToolNames(input: CodeToolPlacementInput): CodeToolNames {
  if (!input.codeToolsEnabled) return { local: [], hosted: [] };
  if (input.hostedCodeExecutionForThisTurn) return { local: [], hosted: ["code_execution"] };
  if (input.providerHostsCodeExecution) return { local: [], hosted: [] };
  // edit_file is local-only: hosted execution keeps files in the provider's sandbox.
  return { local: ["python", "terminal", "edit_file", "view_image"], hosted: [] };
}

/** Derived from the placement so the pill is never offered where it would run nothing. */
export function codeToolCanRun(input: {
  hostedCodeExecutionForThisTurn: boolean;
  providerHostsCodeExecution: boolean;
  /** This provider AND model can run Unsloth's own tools through the loop. */
  supportsStudioTools: boolean;
}): boolean {
  const names = selectCodeToolNames({
    codeToolsEnabled: true,
    hostedCodeExecutionForThisTurn: input.hostedCodeExecutionForThisTurn,
    providerHostsCodeExecution: input.providerHostsCodeExecution,
  });
  if (names.hosted.length > 0) return true;
  return names.local.length > 0 && input.supportsStudioTools;
}
