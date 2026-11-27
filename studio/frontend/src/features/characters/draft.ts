// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { CharacterPatch } from "./api";
import type { Character, CharacterDraft } from "./types";

// Mirrors the server's limits (cards.py), so the editor refuses what the save would reject.
export const MAX_TAGS = 64;
export const MAX_TAG_CHARS = 100;
export const MAX_ALTERNATE_GREETINGS = 64;

export function blankDraft(): CharacterDraft {
  return {
    name: "",
    description: "",
    personality: "",
    scenario: "",
    firstMes: "",
    alternateGreetings: [],
    mesExample: "",
    systemPrompt: "",
    postHistoryInstructions: "",
    creatorNotes: "",
    creator: "",
    characterVersion: "",
    tags: [],
    talkativeness: 0.5,
    favorite: false,
    depthPrompt: null,
    characterBook: null,
  };
}

export function draftOf(character: Character): CharacterDraft {
  const { name, description, personality, scenario, firstMes, alternateGreetings, mesExample, systemPrompt,
    postHistoryInstructions, creatorNotes, creator, characterVersion, tags, talkativeness, favorite, depthPrompt,
    characterBook } = character;
  return {
    name, description, personality, scenario, firstMes, alternateGreetings: [...alternateGreetings], mesExample,
    systemPrompt, postHistoryInstructions, creatorNotes, creator, characterVersion, tags: [...tags], talkativeness,
    favorite, depthPrompt, characterBook,
  };
}

/** Adds comma-separated tags, skipping ones already there in any letter case. */
export function addTags(tags: string[], raw: string): string[] {
  const seen = new Set(tags.map((tag) => tag.toLowerCase()));
  const next = [...tags];
  for (const part of raw.split(",")) {
    const tag = part.trim().slice(0, MAX_TAG_CHARS);
    if (!tag || seen.has(tag.toLowerCase())) continue;
    seen.add(tag.toLowerCase());
    next.push(tag);
  }
  return next.length === tags.length ? tags : next.slice(0, MAX_TAGS);
}

/** The text sent with every turn, for the editor's token estimate. */
export function permanentPromptText(draft: CharacterDraft): string {
  return [
    draft.description,
    draft.personality,
    draft.scenario,
    draft.mesExample,
    draft.systemPrompt,
    draft.postHistoryInstructions,
    draft.depthPrompt?.prompt ?? "",
    // Always-on lore rides every turn too; keyword entries only when they fire.
    ...(draft.characterBook?.entries ?? [])
      .filter((entry) => entry.constant && entry.enabled !== false)
      .map((entry) => entry.content ?? ""),
  ]
    .map((field) => field.trim())
    .filter(Boolean)
    .join("\n");
}

/** The create/update body; an empty note or book is cleared rather than saved empty. */
export function toSavePayload(draft: CharacterDraft): {
  fields: CharacterPatch & { name: string };
  clear: NonNullable<CharacterPatch["clear"]>;
} {
  const { depthPrompt, characterBook, ...rest } = draft;
  const keepNote = Boolean(depthPrompt?.prompt.trim());
  const keepBook = Boolean(characterBook?.entries.length);
  return {
    fields: {
      ...rest,
      name: draft.name.trim(),
      alternateGreetings: draft.alternateGreetings.filter((g) => g.trim()),
      ...(keepNote ? { depthPrompt } : {}),
      ...(keepBook ? { characterBook } : {}),
    },
    clear: [...(keepNote ? [] : (["depthPrompt"] as const)), ...(keepBook ? [] : (["characterBook"] as const))],
  };
}

/** Rounds into [min, max]; anything unparsable becomes `fallback`. */
export function clampNumber(value: number, min = -Infinity, max = Infinity, fallback = Number.isFinite(min) ? min : 0): number {
  if (!Number.isFinite(value)) return fallback;
  return Math.min(max, Math.max(min, Math.round(value)));
}
