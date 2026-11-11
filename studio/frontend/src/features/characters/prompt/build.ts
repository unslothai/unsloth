// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { DepthPrompt, Lorebook, RoleplayMessageMeta, SceneSettings } from "../types";
import { activateLorebook, type ActivatedLore } from "./lorebook.ts";
import { applyMacros, applyNameMacros } from "./macros.ts";

/**
 * Static text sits in one leading system message so the llama.cpp prompt cache keeps it; per-turn
 * text rides late user turns (the backend merges system messages, strict templates reject later ones).
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
  /** Whether each history entry is sent; a hidden turn followed by another user turn is not. */
  sent: boolean[];
  stop: string[];
  speaker: PromptCharacter;
  lore: ActivatedLore;
  sections: PromptSection[];
  /** History is written with speaker names: a group, or a chat that was one. */
  named: boolean;
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

/** "Write my reply": the same prompt plus a user turn asking for the persona's line. */
export function buildImpersonationMessages(input: Omit<BuildInput, "speakerId">): {
  system: string;
  messages: { role: "user" | "assistant"; content: string }[];
  stop: string[];
} {
  const cast = input.cast;
  const built = buildCharacterPrompt({ ...input, speakerId: cast[0]?.id ?? "" });
  const user = input.persona.name || "User";
  const messages = input.history
    .map((entry, index) => ({ role: entry.role, content: built.texts[index] }))
    .filter((_message, index) => built.sent[index]);
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

export function replayedReply(text: string, name: string, group: boolean, recorded = name): string {
  // The name the reply was written under goes too, or a renamed character replays as "Nova: Rin: Hello".
  const body = stripSpeakerPrefix(stripSpeakerPrefix(text, recorded), name).trim();
  return group ? `${name}: ${body}` : body;
}

export function replyNameOf(
  meta: RoleplayMessageMeta | null | undefined,
  cast: readonly { id: string; name: string }[],
  fallback: string,
): string {
  if (meta?.kind !== "reply") return fallback;
  return cast.find((c) => c.id === meta.speakerId)?.name ?? meta.speakerName;
}

type ContentPart = { type: string; text?: string };

/** Tool replay relies on a reply's text/tool order, so only its first text is reformatted. */
export function shapeReplayMessage<M extends { role: string; content: unknown }>(
  message: M,
  text: string,
  replyName: string,
  group: boolean,
  recordedName = replyName,
): M {
  const parts = message.content as readonly ContentPart[];
  const firstText = parts.findIndex((part) => part.type === "text");
  if (message.role === "assistant" && parts.filter((part) => part.type === "text").length > 1) {
    const content = parts.map((part, index) =>
      index === firstText ? { ...part, text: replayedReply(part.text ?? "", replyName, group, recordedName) } : part,
    );
    return { ...message, content };
  }
  const others = parts.filter((part) => part.type !== "text");
  const replacement = { type: "text", text };
  const content =
    firstText < 0 ? [replacement, ...others] : [...others.slice(0, firstText), replacement, ...others.slice(firstText)];
  return { ...message, content };
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

  const history = input.history;
  // An unanswered hidden turn (no greeting) would put two user turns in a row, which strict templates reject.
  const sent = history.map(
    (entry, index) =>
      !(entry.meta?.kind === "opening" || entry.meta?.kind === "turn") || history[index + 1]?.role !== "user",
  );
  const spoken = new Map<string, string>();
  for (const entry of history) if (entry.meta?.kind === "reply") spoken.set(entry.meta.speakerId, entry.meta.speakerName);
  // A chat that was a group keeps its history attributed after its other members leave.
  const named = group || history.some((entry) => entry.meta?.kind === "turn") || [...spoken.keys()].some((id) => id !== speaker.id);
  const nameOf = (id: string | undefined, fallback: string) =>
    cast.find((c) => c.id === id)?.name ?? (id ? spoken.get(id) : undefined) ?? fallback;
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
      const name = replyNameOf(meta, cast, speaker.name);
      return [replayedReply(entry.text, name, named, meta?.kind === "reply" ? meta.speakerName || name : name)];
    }
    // Names only: a re-rolled {{random}} or a fresh {{time}} would rewrite old turns on every send.
    const text = applyNameMacros(entry.text, steadyChar, user).trim();
    const lines = [named ? `${user}: ${text}` : text];
    const answeredBy = !last && named ? nextReplyName(index) : null;
    if (answeredBy) lines.push(nudgeText(answeredBy));
    return lines;
  });

  // Notes at a depth attach to the nearest user turn at or above that depth.
  const injections: { depth: number; text: string }[] = [];
  const note = input.scene.authorsNote;
  if (note?.text?.trim()) injections.push({ depth: note.depth ?? 4, text: macro(note.text) });
  if (speaker.depthPrompt?.prompt.trim()) {
    injections.push({ depth: speaker.depthPrompt.depth, text: macro(speaker.depthPrompt.prompt) });
  }
  injections.push(...lore.atDepth.map((entry) => ({ depth: entry.depth, text: entry.content })));
  for (const injection of injections) {
    const start = Math.max(0, history.length - 1 - Math.max(0, injection.depth));
    const fits = (index: number) => history[index]?.role === "user" && sent[index];
    let target = start;
    while (target > 0 && !fits(target)) target -= 1;
    // Nothing sent above it (only a dropped opening): the next user turn takes it instead.
    if (!fits(target)) for (target = start; target < history.length && !fits(target); ) target += 1;
    if (!fits(target)) continue;
    const text = injection.text.trim();
    parts[target].push(text.startsWith("[") ? text : `[${text}]`);
  }
  // Final instructions ride the last user turn; after a reply there is none, so they are dropped.
  if (history.length && history[history.length - 1].role === "user") {
    const lastParts = parts[history.length - 1];
    const post = speaker.postHistoryInstructions.trim();
    if (post) lastParts.push(macro(post, speaker.name, ""));
    const lastIsMarker = history[history.length - 1].meta?.kind === "turn";
    if (lastIsMarker) {
      // The nudge stays last, after any notes that landed on the marker.
      const nudge = lastParts.shift();
      if (nudge) lastParts.push(nudge);
    } else if (named) {
      lastParts.push(nudgeText(speaker.name));
    }
  }
  const texts = parts.map((lines) => lines.filter(Boolean).join("\n\n"));

  const stop = [`\n${user}:`];
  if (named) {
    const others = new Set([...cast.map((c) => c.name), ...spoken.values()]);
    others.delete(speaker.name);
    for (const name of others) stop.push(`\n${name}:`);
  }

  return { system, texts, sent, stop, speaker, lore, sections, named };
}
