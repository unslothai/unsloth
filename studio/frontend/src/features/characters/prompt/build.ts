// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { escapeRegExp } from "../../../lib/escape-regexp.ts";
import type { DepthPrompt, Lorebook, RoleplayMessageMeta, SceneSettings } from "../types";
import { activateLorebook, loreBudget, type ActivatedLore } from "./lorebook.ts";
import { applyMacros, applyNameMacros } from "./macros.ts";
import { seededRandom, shortName } from "./speakers.ts";

// Card text sits in one leading system message so the prompt cache can reuse it while the cast and
// fired lore stay the same; per-turn text rides late user turns (strict templates reject later system turns).

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
  /** Per-turn text ({{time}} in a note) reads this. */
  now?: Date;
  /** System-block text reads this instead, so its {{time}} and {{date}} hold still for the chat. */
  startedAt?: Date;
  /** Seeds {{random}}, {{pick}} and {{roll}}, so the system block rolls the same on every send of a chat. */
  seed?: string;
  /** The turn asks for the user's line: no post-history instructions, no character nudge at the end. */
  forUser?: boolean;
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
  /** Whether each history entry carries a note placed at a depth (author's note, character note, lore). */
  noted: boolean[];
  stop: string[];
  speaker: PromptCharacter;
  lore: ActivatedLore;
  sections: PromptSection[];
  /** History is written with speaker names: a group, or a chat that was one. */
  named: boolean;
  /** The user's name as turns and stop strings write it; "Rin (you)" when a character is also Rin. */
  user: string;
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
export function buildImpersonationMessages(input: Omit<BuildInput, "speakerId" | "forUser">): {
  system: string;
  messages: { role: "user" | "assistant"; content: string }[];
  stop: string[];
  user: string;
} {
  const cast = input.cast;
  const built = buildCharacterPrompt({ ...input, speakerId: cast[0]?.id ?? "", forUser: true });
  const user = built.user;
  const messages = input.history
    .map((entry, index) => ({ role: entry.role, content: built.texts[index] }))
    .filter((_message, index) => built.sent[index]);
  const last = messages[messages.length - 1];
  const ask =
    `[Write ${user}'s next message now: only what ${user} says and does, a few sentences in ${user}'s own voice ` +
    `and in the style ${user} has used so far. Do not narrate the world or write for any other character.]`;
  if (last?.role === "user") last.content = `${last.content}\n\n${ask}`.trim();
  else messages.push({ role: "user", content: ask });
  const stop = cast.map((member) => `\n${member.name}:`);
  return { system: built.system, messages, stop, user };
}

export function nudgeText(name: string): string {
  return `[Write the next reply only as ${name}.]`;
}

/** A reply without the "Name:" or "## Name" a model sometimes starts with, which the chat already shows. */
export function stripSpeakerPrefix(text: string, name: string): string {
  const full = name.trim();
  if (!full) return text;
  const names = [...new Set([full, shortName(full)])]
    .filter(Boolean)
    .sort((a, b) => b.length - a.length)
    .map(escapeRegExp)
    .join("|");
  const heading = new RegExp(`^\\s*#{1,6}[ \\t]*(?:\\*\\*)?(?:${names})(?:\\*\\*)?[ \\t]*:?[ \\t]*\\n+`, "i").exec(text);
  if (heading) return text.slice(heading[0].length);
  // Also "**Name:**", "**Name**:" and a quote opened before the name ("Mira: Hello.").
  const label = new RegExp(`^\\s*(["“]?)(?:\\*\\*)?(?:${names})(?:\\*\\*)?\\s*:(?:\\*\\*)?\\s*`, "i").exec(text);
  if (!label) return text;
  const body = text.slice(label[0].length);
  // That quote belongs to the line unless the line opens its own.
  return label[1] && !/^["“]/.test(body) ? `${label[1]}${body}` : body;
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

function exampleBlock(raw: string, expand: (value: string) => string): string {
  const blocks = expand(raw)
    .replace(/\r\n/g, "\n")
    .split(/<START>/i)
    .map((block) => block.trim())
    .filter(Boolean);
  return blocks.join("\n\n---\n\n");
}

function characterSection(character: PromptCharacter, expand: (value: string) => string): string {
  const parts: string[] = [];
  const description = expand(character.description).trim();
  if (description) parts.push(description);
  const personality = expand(character.personality).trim();
  if (personality) parts.push(`${character.name}'s personality: ${personality}`);
  return parts.join("\n\n");
}

const sameName = (a: string, b: string) => a.trim().toLowerCase() === b.trim().toLowerCase();

export function buildCharacterPrompt(input: BuildInput): BuiltPrompt {
  const now = input.now ?? new Date();
  const systemNow = input.startedAt ?? now;
  const cast = input.cast;
  const speaker = cast.find((c) => c.id === input.speakerId) ?? cast[0];
  if (!speaker) throw new Error("A character chat needs at least one character.");
  const group = cast.length > 1;
  const names = cast.map((c) => c.name);
  const history = input.history;
  const persona = input.persona.name || "User";
  const spoken = new Map<string, string>();
  for (const entry of history) if (entry.meta?.kind === "reply") spoken.set(entry.meta.speakerId, entry.meta.speakerName);
  // A persona sharing a character's name would read as that character in turns and stop strings.
  const user = [...names, ...spoken.values()].some((name) => sameName(name, persona)) ? `${persona} (you)` : persona;
  // In a group these must not name whoever is speaking, or the system block would change every turn.
  const steadyChar = group ? cast[0].name : speaker.name;
  const seed = input.seed ?? "";
  // System text rolls once per chat; per-turn text once per turn, so a retry reads the same.
  const expand = (value: string, options: { char?: string; original?: string; turn?: boolean } = {}) =>
    applyMacros(value, {
      char: options.char ?? speaker.name,
      user,
      now: options.turn ? now : systemNow,
      group: names,
      original: options.original,
      random: seededRandom(`${seed}\u0001${options.turn ? history.length : "system"}\u0001${value}`),
    });

  const recent = [...history]
    .reverse()
    .filter((entry) => entry.meta?.kind !== "opening" && entry.meta?.kind !== "turn")
    .map((entry) => entry.text);
  const lore: ActivatedLore = { before: [], after: [], atDepth: [], fired: [] };
  // One budget for the whole cast, or each member's book would add its own.
  const budget = loreBudget(cast.map((member) => member.characterBook));
  for (const member of cast) {
    const activated = activateLorebook(member.characterBook, recent, budget);
    lore.before.push(...activated.before.map((t) => expand(t, { char: member.name })));
    lore.after.push(...activated.after.map((t) => expand(t, { char: member.name })));
    lore.atDepth.push(...activated.atDepth.map((d) => ({ ...d, content: expand(d.content, { char: member.name, turn: true }) })));
    lore.fired.push(...activated.fired);
  }

  const sections: PromptSection[] = [];
  const defaultFrame = expand(group ? GROUP_FRAME : SOLO_FRAME);
  const override = speaker.systemPrompt.trim() && !group ? speaker.systemPrompt : "";
  sections.push({
    label: "Instructions",
    text: override ? expand(override, { original: defaultFrame }).trim() : defaultFrame,
  });
  if (lore.before.length) sections.push({ label: "Lore", text: lore.before.join("\n\n") });
  for (const member of cast) {
    const body = characterSection(member, (value) => expand(value, { char: member.name }));
    if (body) sections.push({ label: member.name, text: body });
  }
  if (lore.after.length) sections.push({ label: "More lore", text: lore.after.join("\n\n") });
  const personaText = expand(input.persona.description, { char: steadyChar }).trim();
  if (personaText) sections.push({ label: user === persona ? `${user} (you)` : user, text: personaText });
  const sceneOverride = input.scene.scenario?.trim();
  const scenario = sceneOverride
    ? expand(sceneOverride, { char: steadyChar })
    : cast
        .map((member) => expand(member.scenario, { char: member.name }).trim())
        .filter(Boolean)
        .filter((value, index, all) => all.indexOf(value) === index)
        .join("\n\n");
  if (scenario) sections.push({ label: "Scenario", text: scenario });
  const examples = cast
    .map((member) => {
      const block = exampleBlock(member.mesExample, (value) => expand(value, { char: member.name }));
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

  // An unanswered hidden turn (no greeting) would put two user turns in a row, which strict templates reject.
  const sent = history.map(
    (entry, index) =>
      !(entry.meta?.kind === "opening" || entry.meta?.kind === "turn") || history[index + 1]?.role !== "user",
  );
  // A chat that was a group keeps its history attributed after its other members leave.
  const named = group || history.some((entry) => entry.meta?.kind === "turn") || [...spoken.keys()].some((id) => id !== speaker.id);
  const nameOf = (id: string | undefined, fallback: string) =>
    cast.find((c) => c.id === id)?.name ?? (id ? spoken.get(id) : undefined) ?? fallback;
  // An untagged reply (from before the chat had a cast) keeps one author, not whoever speaks now.
  const nextReplyName = (index: number): string | null => {
    const next = history[index + 1];
    if (next?.role === "assistant" && next.meta?.kind === "reply") return nameOf(next.meta.speakerId, next.meta.speakerName);
    if (next?.role === "assistant") return steadyChar;
    return null;
  };
  const parts: string[][] = history.map((entry, index) => {
    const meta = entry.meta;
    const last = index === history.length - 1;
    // A marker asks the named character to speak again without a new user message.
    if (meta?.kind === "turn") return [nudgeText(nameOf(meta.speakerId, speaker.name))];
    if (meta?.kind === "opening") return [entry.text.trim() || "[Start a new chat]"];
    if (entry.role === "assistant") {
      const name = replyNameOf(meta, cast, steadyChar);
      return [replayedReply(entry.text, name, named, meta?.kind === "reply" ? meta.speakerName || name : name)];
    }
    // Names only: a re-rolled {{random}} or a fresh {{time}} would rewrite old turns on every send.
    const text = applyNameMacros(entry.text, steadyChar, persona).trim();
    const lines = [named ? `${user}: ${text}` : text];
    const answeredBy = !last && named ? nextReplyName(index) : null;
    if (answeredBy) lines.push(nudgeText(answeredBy));
    return lines;
  });

  // Notes at a depth attach to the nearest user turn at or above that depth.
  const injections: { depth: number; text: string }[] = [];
  const note = input.scene.authorsNote;
  if (note?.text?.trim()) injections.push({ depth: note.depth ?? 4, text: expand(note.text, { turn: true }) });
  if (speaker.depthPrompt?.prompt.trim()) {
    injections.push({ depth: speaker.depthPrompt.depth, text: expand(speaker.depthPrompt.prompt, { turn: true }) });
  }
  injections.push(...lore.atDepth.map((entry) => ({ depth: entry.depth, text: entry.content })));
  const noted = history.map(() => false);
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
    noted[target] = true;
  }
  // Final instructions ride the last user turn; after a reply there is none, so they are dropped.
  if (history.length && history[history.length - 1].role === "user") {
    const lastParts = parts[history.length - 1];
    const post = speaker.postHistoryInstructions.trim();
    if (post && !input.forUser) lastParts.push(expand(post, { original: "", turn: true }));
    const lastIsMarker = history[history.length - 1].meta?.kind === "turn";
    if (lastIsMarker) {
      // The nudge stays last, after any notes that landed on the marker; a turn for the user drops it.
      const nudge = lastParts.shift();
      if (nudge && !input.forUser) lastParts.push(nudge);
    } else if (named && !input.forUser) {
      lastParts.push(nudgeText(speaker.name));
    }
  }
  const texts = parts.map((lines) => lines.filter(Boolean).join("\n\n"));

  const stop = [`\n${user}:`];
  if (named) {
    const others = new Set([...names, ...spoken.values()]);
    others.delete(speaker.name);
    for (const name of others) stop.push(`\n${name}:`);
  }

  return { system, texts, sent, noted, stop, speaker, lore, sections, named, user };
}
