// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Master-off (#11671): no Search/Code/MCP/Skills/… pill may open the tool loop. */

export type ToolPillFlags = {
  toolsEnabled: boolean;
  codeToolsEnabled: boolean;
  artifactsEnabled: boolean;
  mcpEnabledForChat: boolean;
  ragOn: boolean;
  deepResearchArmed: boolean;
  hasEnabledSkills: boolean;
};

/** True when any pill would open the Unsloth tool loop (before master-off). */
export function anyToolPillOn(flags: ToolPillFlags): boolean {
  return (
    flags.toolsEnabled ||
    flags.codeToolsEnabled ||
    flags.artifactsEnabled ||
    flags.mcpEnabledForChat ||
    flags.ragOn ||
    flags.deepResearchArmed ||
    flags.hasEnabledSkills
  );
}

/** Final enable_tools for a Studio-tools request. */
export function resolveEnableTools(
  wantTools: boolean,
  allToolsOff: boolean,
): boolean {
  return wantTools && !allToolsOff;
}
