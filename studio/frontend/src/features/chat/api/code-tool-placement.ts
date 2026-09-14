// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Which side of the connection the Code pill runs code on. `code_execution` runs in the
 *  provider's own sandbox and is billed by them; `python` / `terminal` are Unsloth's tools and
 *  run on the machine Unsloth is installed on. They are two different trust boundaries, so which
 *  one a stored toggle means must not change underneath the user. Until Unsloth's tool loop
 *  reached the general external providers, only openai_codex carried studio_tools, so the pill on
 *  an OpenAI or Anthropic connection always resolved to the provider's sandbox; now the same
 *  persisted `true` would resolve to local execution. The rule: a connection with its own sandbox
 *  keeps it, and a model there that cannot use it runs nothing rather than falling back to the
 *  user's machine. Unsloth's local tools are for connections with no sandbox at all. */

export interface CodeToolPlacementInput {
  /** The composer's Code pill (persisted as unsloth_chat_code_tools_enabled). */
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
  // edit_file is local-only: when the provider hosts execution the files live in its sandbox, so a
  // local editor would patch a copy nothing else sees.
  return { local: ["python", "terminal", "edit_file"], hosted: [] };
}

/** Whether the Code pill can do anything on this connection, so the composer can offer it and a
 *  stored preference can be restored. Read out of the placement itself rather than restated,
 *  because the two must not drift: a pill offered where the placement runs nothing is a toggle
 *  the user turns on to no effect. That is the case for a model on a sandbox-owning provider
 *  that cannot use it. */
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
