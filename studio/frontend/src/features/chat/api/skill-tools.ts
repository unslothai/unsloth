// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Skills from ~/.agents/skills and ~/.claude/skills are on by default with no pill, so they
// follow Code; otherwise a chat with every pill off still prompted for read_skill (#11671).

export interface SkillToolEntry {
  valid: boolean;
  shadowed: boolean;
  enabled: boolean;
}

export function skillToolsOffered(
  skills: readonly SkillToolEntry[],
  codeToolsOn: boolean,
): boolean {
  return (
    codeToolsOn &&
    skills.some((skill) => skill.valid && !skill.shadowed && skill.enabled)
  );
}
