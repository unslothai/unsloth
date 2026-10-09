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

const FENCE = /^ {0,3}(`{3,}|~{3,})/;

// Blank out what the backend preload treats as literal: fenced and indented code, blockquotes,
// inline code and balanced quotations. Approximate CommonMark; it only decides whether to offer read_skill.
function proseOnly(text: string): string {
  let fence: string | null = null;
  let previousBlank = true;
  const lines = text.split(/\r\n?|\n/).map((line) => {
    const opener = FENCE.exec(line)?.[1];
    if (fence !== null) {
      if (opener && opener[0] === fence[0] && opener.length >= fence.length) fence = null;
      return "";
    }
    if (opener) {
      fence = opener;
      return "";
    }
    const literal = /^ {0,3}>/.test(line) || (previousBlank && /^( {4}|\t)/.test(line));
    previousBlank = line.trim() === "";
    return literal ? "" : line;
  });
  return lines
    .join("\n")
    .replace(/(`+)[^`]*?\1/g, " ")
    .replace(/"(?:\\.|[^"])*"|“[^”]*”|‘[^’]*’|(?<!\w)'(?:\\.|(?<=\w)'(?=\w)|[^'])*'/g, " ");
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
  const mentioned = userTexts.some((text) =>
    Array.from(proseOnly(text).matchAll(SKILL_MENTION_PATTERN)).some((match) =>
      usable.has(match[2] ?? ""),
    ),
  );
  return mentioned ? ["read_skill"] : [];
}
