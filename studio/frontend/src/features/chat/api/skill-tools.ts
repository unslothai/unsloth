// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Home-dir skills are on by default with no pill of their own, so they follow Code (#11671).

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
