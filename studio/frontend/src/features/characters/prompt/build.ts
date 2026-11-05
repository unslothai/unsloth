// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { DepthPrompt, Lorebook, RoleplayMessageMeta, SceneSettings } from "../types";
import { activateLorebook, type ActivatedLore } from "./lorebook.ts";
import { applyMacros, applyNameMacros } from "./macros.ts";

/**
 * Builds a character chat's prompt for a chat-template model.
 *
 * Everything that does not change between replies sits in one leading system message, so a
 * llama.cpp prompt cache keeps it. What does change (notes at a depth, post-history
 * instructions, whose turn it is) rides on user turns near the end: Studio's backend merges
 * every system message into the first, and strict chat templates reject system turns after it.
 * History stays append-only, including in groups, so each reply reprocesses only the new tail.
 */

export interface PromptCharacter {
  id: string;
  name: string;
  description: string;
  personality: string;
  scenario: string;
  mesExample: string;
  systemPrompt: string;
  postHistoryInstructions: string;
  depthPrompt: DepthPrompt | null;
  characterBook: Lorebook | null;
}

export interface HistoryEntry {
  role: "user" | "assistant";
  text: string;
  meta?: RoleplayMessageMeta | null;
}

export interface BuildInput {
  /** Every character in the chat, in cast order. */
  cast: PromptCharacter[];
  speakerId: string;
  persona: { name: string; description: string };
  scene: SceneSettings;
  history: HistoryEntry[];
  now?: Date;
}

export interface PromptSection {
  label: string;
  text: string;
}

export interface BuiltPrompt {
  system: string;
  /** One text per history entry, same order and length as the input. */
  texts: string[];
  stop: string[];
  speaker: PromptCharacter;
  lore: ActivatedLore;
  sections: PromptSection[];
}

export const SOLO_FRAME =
  "You are {{char}} in an ongoing roleplay with {{user}}. Write {{char}}'s next reply only. " +
  "Stay in character and keep the scene alive: react to what just happened, act, and bring in your own ideas, " +
  "feelings and small surprises instead of only asking {{user}} questions. " +
  "Leave {{user}}'s words, thoughts and actions to {{user}}.";

export const GROUP_FRAME =
  "This is a group roleplay between {{user}} and {{group}}. Each turn you are told which character speaks; " +
  "write only that character's next reply, in their own voice. Characters talk to each other as well as to {{user}}: " +
  "react to what was just said, tease, disagree, make plans and move the scene forward, rather than only asking {{user}} questions. " +
  "Leave {{user}}'s words, thoughts and actions to {{user}}.";

/**
 * Messages that ask the model to write the user's own next line ("Write my reply"): the same
 * prompt, then one more user turn asking for the persona's words instead of a character's.
 */
export function buildImpersonationMessages(input: Omit<BuildInput, "speakerId">): {
  system: string;
  messages: { role: "user" | "assistant"; content: string }[];
  stop: string[];
} {
  const cast = input.cast;
  const built = buildCharacterPrompt({ ...input, speakerId: cast[0]?.id ?? "" });
  const user = input.persona.name || "User";
  const messages = input.history.map((entry, index) => ({ role: entry.role, content: built.texts[index] }));
  // Group nudges ask a character to speak; this turn asks for the user instead.
  const last = messages[messages.length - 1];
  if (last?.role === "user") {
    const groupNudge = /\n*\[Write the next reply only as [^\]]*\.\]$/;
    last.content = last.content.replace(groupNudge, "");
  }
  const ask =
    `[Write ${user}'s next message now: only what ${user} says and does, a few sentences in ${user}'s own voice ` +
    `and in the style ${user} has used so far. Do not narrate the world or write for any other character.]`;
  if (last?.role === "user") last.content = `${last.content}\n\n${ask}`.trim();
  else messages.push({ role: "user", content: ask });
  const stop = cast.map((member) => `\n${member.name}:`);
  return { system: built.system, messages, stop };
}

export function nudgeText(name: string): string {
  return `[Write the next reply only as ${name}.]`;
}

/** A reply without the "Name:" a model sometimes starts with, which the chat already shows. */
export function stripSpeakerPrefix(text: string, name: string): string {
  if (!name) return text;
  const escaped = name.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  // Also "**Name:**" and a stray opening quote before the name ("Name: "Hi"), which models produce.
  return text.replace(new RegExp(`^\\s*["“]?(?:\\*\\*)?${escaped}(?:\\*\\*)?\\s*:(?:\\*\\*)?\\s*`, "i"), "");
}

function exampleBlock(raw: string, char: string, user: string, now: Date, group: string[]): string {
  const text = applyMacros(raw, { char, user, now, group }).replace(/\r\n/g, "\n");
  const blocks = text
    .split(/<START>/i)
    .map((block) => block.trim())
    .filter(Boolean);
  return blocks.join("\n\n---\n\n");
}

function characterSection(character: PromptCharacter, context: { user: string; now: Date; group: string[] }): string {
  const macro = (value: string) =>
    applyMacros(value, { char: character.name, user: context.user, now: context.now, group: context.group }).trim();
  const parts: string[] = [];
  const description = macro(character.description);
  if (description) parts.push(description);
  const personality = macro(character.personality);
  if (personality) parts.push(`${character.name}'s personality: ${personality}`);
  return parts.join("\n\n");
}

export function buildCharacterPrompt(input: BuildInput): BuiltPrompt {
  const now = input.now ?? new Date();
  const cast = input.cast.length ? input.cast : [];
  const speaker = cast.find((c) => c.id === input.speakerId) ?? cast[0];
  if (!speaker) throw new Error("A character chat needs at least one character.");
  const group = cast.length > 1;
  const names = cast.map((c) => c.name);
  const user = input.persona.name || "User";
  const macro = (value: string, char = speaker.name, original?: string) =>
    applyMacros(value, { char, user, now, group: names, original });

  // ── Lore: scanned over recent messages, from every member's book.
  const recent = [...input.history]
    .reverse()
    .filter((entry) => entry.meta?.kind !== "opening" && entry.meta?.kind !== "turn")
    .map((entry) => entry.text);
  const lore: ActivatedLore = { before: [], after: [], atDepth: [], fired: [] };
  for (const member of cast) {
    const activated = activateLorebook(member.characterBook, recent);
    lore.before.push(...activated.before.map((t) => macro(t, member.name)));
    lore.after.push(...activated.after.map((t) => macro(t, member.name)));
    lore.atDepth.push(...activated.atDepth.map((d) => ({ ...d, content: macro(d.content, member.name) })));
    lore.fired.push(...activated.fired);
  }

  // ── System block.
  const sections: PromptSection[] = [];
  const defaultFrame = macro(group ? GROUP_FRAME : SOLO_FRAME);
  const override = speaker.systemPrompt.trim() && !group ? speaker.systemPrompt : "";
  sections.push({
    label: "Instructions",
    text: override ? macro(override, speaker.name, defaultFrame).trim() : defaultFrame,
  });
  if (lore.before.length) sections.push({ label: "Lore", text: lore.before.join("\n\n") });
  for (const member of cast) {
    const body = characterSection(member, { user, now, group: names });
    if (body) sections.push({ label: member.name, text: body });
  }
  if (lore.after.length) sections.push({ label: "More lore", text: lore.after.join("\n\n") });
  // In a group these must not name whoever is speaking, or the system block would change every turn.
  const steadyChar = group ? cast[0].name : speaker.name;
  const personaText = macro(input.persona.description, steadyChar).trim();
  if (personaText) sections.push({ label: `${user} (you)`, text: personaText });
  const sceneOverride = input.scene.scenario?.trim();
  const scenario = sceneOverride
    ? macro(sceneOverride, steadyChar)
    : cast
        .map((member) => macro(member.scenario, member.name).trim())
        .filter(Boolean)
        .filter((value, index, all) => all.indexOf(value) === index)
        .join("\n\n");
  if (scenario) sections.push({ label: "Scenario", text: scenario });
  const examples = cast
    .map((member) => {
      const block = exampleBlock(member.mesExample, member.name, user, now, names);
      return block ? (group ? `${member.name}:\n${block}` : block) : "";
    })
    .filter(Boolean);
  if (examples.length) {
    sections.push({
      label: "Example dialogue",
      text: `These examples show how ${group ? "the characters speak" : `${speaker.name} speaks`}. They are not part of the story.\n\n${examples.join("\n\n")}`,
    });
  }
  const system = sections
    .map((section, index) => (index === 0 ? section.text : `## ${section.label}\n${section.text}`))
    .join("\n\n");

  // ── History, append-only.
  const history = input.history;
  const nameOf = (id: string | undefined, fallback: string) =>
    cast.find((c) => c.id === id)?.name ?? fallback;
  const nextReplyName = (index: number): string | null => {
    const next = history[index + 1];
    if (next?.role === "assistant" && next.meta?.kind === "reply") return nameOf(next.meta.speakerId, next.meta.speakerName);
    if (next?.role === "assistant") return speaker.name;
    return null;
  };
  const parts: string[][] = history.map((entry, index) => {
    const meta = entry.meta;
    const last = index === history.length - 1;
    // A marker asks the named character to speak again without a new user message.
    if (meta?.kind === "turn") return [nudgeText(nameOf(meta.speakerId, speaker.name))];
    if (meta?.kind === "opening") return [entry.text.trim() || "[Start a new chat]"];
    if (entry.role === "assistant") {
      const name = meta?.kind === "reply" ? nameOf(meta.speakerId, meta.speakerName) : speaker.name;
      const text = stripSpeakerPrefix(entry.text, name).trim();
      return [group ? `${name}: ${text}` : text];
    }
    // Names only: a re-rolled {{random}} or a fresh {{time}} would rewrite old turns on every send.
    const text = applyNameMacros(entry.text, steadyChar, user).trim();
    const lines = [group ? `${user}: ${text}` : text];
    const answeredBy = !last && group ? nextReplyName(index) : null;
    if (answeredBy) lines.push(nudgeText(answeredBy));
    return lines;
  });

  // Notes at a depth attach to the nearest user turn at or above that depth.
  const injections: { depth: number; text: string }[] = [];
  const note = input.scene.authorsNote;
  if (note?.text.trim()) injections.push({ depth: note.depth, text: macro(note.text) });
  if (speaker.depthPrompt?.prompt.trim()) {
    injections.push({ depth: speaker.depthPrompt.depth, text: macro(speaker.depthPrompt.prompt) });
  }
  injections.push(...lore.atDepth.map((entry) => ({ depth: entry.depth, text: entry.content })));
  for (const injection of injections) {
    let target = Math.max(0, history.length - 1 - Math.max(0, injection.depth));
    while (target > 0 && history[target]?.role !== "user") target -= 1;
    if (history[target]?.role !== "user") continue;
    const text = injection.text.trim();
    parts[target].push(text.startsWith("[") ? text : `[${text}]`);
  }
  // The final instructions ride the last user turn. After a reply (as when drafting the user's own
  // line) there is no such turn, so they are left out.
  if (history.length && history[history.length - 1].role === "user") {
    const lastParts = parts[history.length - 1];
    const post = speaker.postHistoryInstructions.trim();
    if (post) lastParts.push(macro(post, speaker.name, ""));
    const lastIsMarker = history[history.length - 1].meta?.kind === "turn";
    if (lastIsMarker) {
      // The nudge stays last, after any notes that landed on the marker.
      const nudge = lastParts.shift();
      if (nudge) lastParts.push(nudge);
    } else if (group) {
      lastParts.push(nudgeText(speaker.name));
    }
  }
  const texts = parts.map((lines) => lines.filter(Boolean).join("\n\n"));

  const stop = [`\n${user}:`];
  if (group) for (const member of cast) if (member.id !== speaker.id) stop.push(`\n${member.name}:`);

  return { system, texts, stop, speaker, lore, sections };
}
