// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// mirror backend preload token boundaries, including at most one trailing sentence mark.
export const SKILL_MENTION_PATTERN =
  /(^|\s)@([a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?)(?=$|\s|[.,;:!?)](?:$|\s))/g;

export interface SkillToolEntry {
  name: string;
  valid: boolean;
  shadowed: boolean;
  enabled: boolean;
}

// with Code off, only an @mention offers read_skill; create_skill stays gated by Code (#11671).
export function skillToolNames(
  skills: readonly SkillToolEntry[],
  codeToolsOn: boolean,
  userTexts: readonly string[],
): string[] {
  const usable = new Set(
    skills
      .filter((skill) => skill.valid && !skill.shadowed && skill.enabled)
      .map((skill) => skill.name),
  );
  if (usable.size === 0) return [];
  if (codeToolsOn) return ["read_skill", "create_skill"];
  // earlier turns count because follow-ups must re-read SKILL.md contents that are not replayed.
  // do not mask code or quotes because frontend Markdown guesses can hide backend-literal mentions.
  const mentioned = userTexts.some((text) =>
    Array.from(text.matchAll(SKILL_MENTION_PATTERN)).some((match) =>
      usable.has(match[2] ?? ""),
    ),
  );
  return mentioned ? ["read_skill"] : [];
}
