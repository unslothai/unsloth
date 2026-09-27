// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Agent Skills are discovered from ~/.agents/skills and ~/.claude/skills (put there by other
 *  agents or `hf skills add`) and are on by default with no pill of their own, so with every
 *  pill off a chat still opened the tool loop for read_skill and prompted for it (#11671).
 *  Skills are instructions for running code, so they ride along with the Code pill. */

export interface SkillToolEntry {
  valid: boolean;
  shadowed: boolean;
  enabled: boolean;
}

/** Whether this request offers read_skill / create_skill. */
export function skillToolsOffered(
  skills: readonly SkillToolEntry[],
  codeToolsOn: boolean,
): boolean {
  return (
    codeToolsOn &&
    skills.some((skill) => skill.valid && !skill.shadowed && skill.enabled)
  );
}
