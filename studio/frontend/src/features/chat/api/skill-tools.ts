// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Spec skill names only, ending at a word boundary: `@example.com`, `@3pm`, `@Probe` are not mentions.
// At most one trailing sentence mark, as the backend preload's _TOKEN, so `@name!!` loads on neither side.
export const SKILL_MENTION_PATTERN =
  /(^|\s)@([a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?)(?=$|\s|[.,;:!?)](?:$|\s))/g;

export interface SkillToolEntry {
  name: string;
  valid: boolean;
  shadowed: boolean;
  enabled: boolean;
}

// Home-dir skills are on by default with no pill of their own, so with Code off a plain chat must not
// open the tool loop for them (#11671). An @mention is the user asking for one, so it offers read_skill
// alone; create_skill writes files and stays with Code.
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
  // Any earlier turn counts too: the preloaded SKILL.md is not replayed, so a follow-up re-reads it.
  // Code and quotations are not masked here: a mention the backend treats as literal only opens the
  // loop with read_skill, while a client-side Markdown guess can hide a real one.
  const mentioned = userTexts.some((text) =>
    Array.from(text.matchAll(SKILL_MENTION_PATTERN)).some((match) =>
      usable.has(match[2] ?? ""),
    ),
  );
  return mentioned ? ["read_skill"] : [];
}
