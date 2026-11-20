// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import {
  buildCharacterPrompt,
  buildImpersonationMessages,
  nudgeText,
  shapeReplayMessage,
  stripSpeakerPrefix,
  type PromptCharacter,
} from "../src/features/characters/prompt/build.ts";
import { activateLorebook, parseKeyList } from "../src/features/characters/prompt/lorebook.ts";
import { applyMacros, applyNameMacros } from "../src/features/characters/prompt/macros.ts";
import { hiddenRoleplayTurnKind } from "../src/features/characters/meta.ts";
import { tidyCharacterDisplay } from "../src/features/characters/prompt/display.ts";
import { nextInRound, resolveSpeaker, textOf } from "../src/features/characters/prompt/round.ts";
import { mentionedIds, planReplies, seededRandom, shortName } from "../src/features/characters/prompt/speakers.ts";
import type { Character, ChatScene, Lorebook, LorebookEntry } from "../src/features/characters/types.ts";

function character(overrides: Partial<PromptCharacter> & { id: string; name: string }): PromptCharacter {
  return {
    description: "",
    personality: "",
    scenario: "",
    mesExample: "",
    systemPrompt: "",
    postHistoryInstructions: "",
    depthPrompt: null,
    characterBook: null,
    ...overrides,
  };
}

function entry(overrides: Partial<LorebookEntry>): LorebookEntry {
  return {
    id: 1,
    keys: [],
    secondary_keys: [],
    content: "",
    comment: "",
    enabled: true,
    constant: false,
    selective: false,
    case_sensitive: false,
    insertion_order: 100,
    position: "before_char",
    depth: 4,
    selective_logic: "and_any",
    ...overrides,
  };
}

function book(entries: LorebookEntry[], overrides: Partial<Lorebook> = {}): Lorebook {
  return { name: "", scan_depth: 4, token_budget: 1024, recursive_scanning: false, extensions: {}, entries, ...overrides };
}

const NOW = new Date(2026, 10, 4, 9, 5);

test("macros cover the names, comments, trim, dice and picks cards use", () => {
  const ctx = { char: "Mira", user: "Pat", now: NOW, random: () => 0 };
  assert.equal(applyMacros("{{char}} greets {{USER}} and <BOT> <user>", ctx), "Mira greets Pat and Mira Pat");
  assert.equal(applyMacros("a{{// hidden note}}b", ctx), "ab");
  assert.equal(applyMacros("line\n{{trim}}\nnext", ctx), "linenext");
  assert.equal(applyMacros("{{random:red,green}} {{random::x, y::z}}", ctx), "red x, y");
  assert.equal(applyMacros("{{roll:2d6+1}}", ctx), "3");
  assert.equal(applyMacros("{{isodate}} {{isotime}} {{weekday}}", ctx), "2026-11-04 09:05 Wednesday");
  assert.equal(applyMacros("{{original}}!", { ...ctx, original: "base" }), "base!");
  assert.equal(applyMacros("{{unknown}} stays", ctx), "{{unknown}} stays");
  assert.equal(applyMacros("{{group}}", { ...ctx, group: ["Mira", "Bo"] }), "Mira, Bo");
});

test("lorebook entries fire on whole-word keys, honor secondary logic, and keep to the budget", () => {
  const lore = book([
    entry({ id: "cat", keys: ["cat"], content: "Cats rule." }),
    entry({ id: "sel", keys: ["tea"], selective: true, secondary_keys: ["cold"], selective_logic: "not_any", content: "Tea is warm." }),
    entry({ id: "const", constant: true, content: "Always here.", position: "after_char" }),
    entry({ id: "deep", keys: ["/dra(g|k)on/i"], content: "Dragons sleep.", position: "at_depth", depth: 2 }),
    entry({ id: "off", keys: ["cat"], enabled: false, content: "Disabled." }),
  ]);
  const fired = activateLorebook(lore, ["A category of tea, and a Drakon."]);
  assert.deepEqual(fired.before, ["Tea is warm."]);
  assert.deepEqual(fired.after, ["Always here."]);
  assert.deepEqual(fired.atDepth, [{ depth: 2, content: "Dragons sleep." }]);
  const cold = activateLorebook(lore, ["cold tea and a cat"]);
  assert.deepEqual(cold.before, ["Cats rule."]);
  // Only messages within scan depth count.
  const shallow = activateLorebook(book([entry({ keys: ["old"], content: "x" })], { scan_depth: 1 }), ["new", "old"]);
  assert.equal(shallow.fired.length, 0);
  const tight = activateLorebook(
    book([entry({ id: 1, constant: true, content: "a".repeat(40) }), entry({ id: 2, constant: true, content: "b".repeat(40) })], { token_budget: 12 }),
    [],
  );
  assert.equal(tight.fired.length, 1);
  const oversized = activateLorebook(book([entry({ constant: true, content: "a".repeat(400) })], { token_budget: 64 }), []);
  assert.equal(oversized.fired.length, 0, "an entry over the budget is left out, even the first");
});

test("speaker planning is deterministic per message and follows each reply order", () => {
  const cast = [
    { id: "a", name: "Ann Lee", talkativeness: 0.5, muted: false },
    { id: "b", name: "Bo", talkativeness: 1, muted: false },
    { id: "c", name: "Cy", talkativeness: 0, muted: true },
  ];
  assert.deepEqual(mentionedIds("hey bo, and ann?", cast), ["b", "a"]);
  assert.deepEqual(mentionedIds("bonfire", cast), []);
  assert.deepEqual(planReplies({ cast, order: "list", userText: "", lastSpeakerId: null, seed: "m1" }), ["a", "b"]);
  assert.deepEqual(planReplies({ cast, order: "manual", userText: "ann?", lastSpeakerId: null, seed: "m1" }), ["a"]);
  assert.deepEqual(planReplies({ cast, order: "manual", userText: "hi", lastSpeakerId: "a", seed: "m1" }), ["b"]);
  const natural = planReplies({ cast, order: "natural", userText: "Ann, look", lastSpeakerId: null, seed: "m1" });
  assert.equal(natural[0], "a");
  assert.ok(!natural.includes("c"), "muted members never reply");
  for (const seed of ["x", "y", "z"]) {
    assert.deepEqual(
      planReplies({ cast, order: "natural", userText: "", lastSpeakerId: "b", seed }),
      planReplies({ cast, order: "natural", userText: "", lastSpeakerId: "b", seed }),
    );
  }
  assert.notEqual(seededRandom("a")(), seededRandom("b")());
});

test("a one-on-one prompt keeps the system block stable and puts notes on user turns", () => {
  const mira = character({
    id: "m",
    name: "Mira",
    description: "{{char}} keeps a greenhouse for {{user}}.",
    personality: "patient",
    scenario: "Dusk.",
    mesExample: "<START>\n{{user}}: Hi\n{{char}}: Hello.",
    postHistoryInstructions: "Stay gentle.",
    depthPrompt: { prompt: "Mira hides a secret.", depth: 0, role: "system" },
    characterBook: book([entry({ keys: ["fern"], content: "Ferns are sacred to {{char}}." })]),
  });
  const built = buildCharacterPrompt({
    cast: [mira],
    speakerId: "m",
    persona: { name: "Pat", description: "A traveler." },
    scene: { authorsNote: { text: "Keep it short.", depth: 2 } },
    history: [
      { role: "user", text: "[Start a new chat]", meta: { kind: "opening" } },
      { role: "assistant", text: "Hello Pat.", meta: { kind: "reply", speakerId: "m", speakerName: "Mira" } },
      { role: "user", text: "Mira: what about the fern?" },
      { role: "assistant", text: "Mira: They matter.", meta: { kind: "reply", speakerId: "m", speakerName: "Mira" } },
      { role: "user", text: "Tell me more." },
    ],
    now: NOW,
  });
  assert.match(built.system, /^You are Mira in an ongoing roleplay with Pat\./);
  assert.match(built.system, /## Lore\nFerns are sacred to Mira\./);
  assert.match(built.system, /## Mira\nMira keeps a greenhouse for Pat\.\n\nMira's personality: patient/);
  assert.match(built.system, /## Pat \(you\)\nA traveler\./);
  assert.match(built.system, /## Scenario\nDusk\./);
  assert.match(built.system, /Pat: Hi\nMira: Hello\./);
  assert.equal(built.texts[0], "[Start a new chat]");
  assert.equal(built.texts[3], "They matter.", "a leading name is stripped in one-on-one chats");
  // Author's note at depth 2 lands on the user turn two above the end.
  assert.equal(built.texts[2], "Mira: what about the fern?\n\n[Keep it short.]");
  assert.deepEqual(built.noted, [false, false, true, false, true], "View prompt shows every noted turn");
  assert.equal(built.texts[4], "Tell me more.\n\n[Mira hides a secret.]\n\nStay gentle.");
  assert.deepEqual(built.stop, ["\nPat:"]);
});

test("a card's system prompt replaces the frame and can keep it with {{original}}", () => {
  const card = character({ id: "x", name: "X", systemPrompt: "{{original}} Speak in rhyme." });
  const built = buildCharacterPrompt({
    cast: [card],
    speakerId: "x",
    persona: { name: "Pat", description: "" },
    scene: {},
    history: [{ role: "user", text: "hi" }],
  });
  assert.match(built.system, /^You are X in an ongoing roleplay with Pat\..* Speak in rhyme\.$/s);
});

test("a group prompt names every turn and stays append-only across speakers", () => {
  const ann = character({ id: "a", name: "Ann", description: "A knight." });
  const bo = character({ id: "b", name: "Bo", description: "A bard.", postHistoryInstructions: "Sing." });
  const base = [
    { role: "user" as const, text: "[Start a new chat]", meta: { kind: "opening" as const } },
    { role: "assistant" as const, text: "I guard.", meta: { kind: "reply" as const, speakerId: "a", speakerName: "Ann" } },
    { role: "user" as const, text: "", meta: { kind: "turn" as const, speakerId: "b" } },
    { role: "assistant" as const, text: "Bo: I sing.", meta: { kind: "reply" as const, speakerId: "b", speakerName: "Bo" } },
    { role: "user" as const, text: "Who is hungry?" },
  ];
  const first = buildCharacterPrompt({
    cast: [ann, bo],
    speakerId: "a",
    persona: { name: "Pat", description: "" },
    scene: {},
    history: base,
  });
  assert.match(first.system, /^This is a group roleplay between Pat and Ann, Bo\./);
  assert.match(first.system, /## Ann\nA knight\.\n\n## Bo\nA bard\./);
  assert.deepEqual(first.texts, [
    "[Start a new chat]",
    "Ann: I guard.",
    nudgeText("Bo"),
    "Bo: I sing.",
    `Pat: Who is hungry?\n\n${nudgeText("Ann")}`,
  ]);
  assert.deepEqual(first.stop, ["\nPat:", "\nBo:"]);
  const next = buildCharacterPrompt({
    cast: [ann, bo],
    speakerId: "b",
    persona: { name: "Pat", description: "" },
    scene: {},
    history: [
      ...base,
      { role: "assistant", text: "Me.", meta: { kind: "reply", speakerId: "a", speakerName: "Ann" } },
      { role: "user", text: "", meta: { kind: "turn", speakerId: "b" } },
    ],
  });
  assert.equal(next.system, first.system, "the system block does not depend on the speaker");
  // Every earlier turn reads exactly as it was sent, so only the new tail is new to the cache.
  assert.deepEqual(next.texts.slice(0, 5), first.texts);
  assert.equal(next.texts[5], "Ann: Me.");
  assert.equal(next.texts[6], `Sing.\n\n${nudgeText("Bo")}`);
});

test("speaker prefixes are stripped in any common spelling", () => {
  assert.equal(stripSpeakerPrefix("Mira: hi", "Mira"), "hi");
  assert.equal(stripSpeakerPrefix("  **Mira**: hi", "Mira"), "hi");
  assert.equal(stripSpeakerPrefix("Miranda: hi", "Mira"), "Miranda: hi");
  assert.equal(stripSpeakerPrefix('"Mira: "Hm?"', "Mira"), '"Hm?"');
  assert.equal(stripSpeakerPrefix("**Mira:** hi", "Mira"), "hi");
});

function member(id: string, name: string): Character {
  return {
    id, name, talkativeness: 0.5, description: "", personality: "", scenario: "", firstMes: "", alternateGreetings: [],
    mesExample: "", systemPrompt: "", postHistoryInstructions: "", creatorNotes: "", creator: "", characterVersion: "",
    tags: [], favorite: false, depthPrompt: null, characterBook: null, extensions: {}, hasAvatar: false, avatarType: null,
    avatarRev: 0, createdAt: 0, updatedAt: 0, lastChatAt: null, chatCount: 0, cardSpec: null,
  };
}

test("a group round plans once per user message and goes on until everyone planned has spoken", () => {
  const cast = [member("a", "Ann"), member("b", "Bo")];
  const scene: ChatScene = {
    threadId: "t",
    personaId: null,
    settings: { replyOrder: "list" },
    cast: [
      { characterId: "a", position: 0, muted: false },
      { characterId: "b", position: 1, muted: false },
    ],
  };
  const user = { id: "u1", role: "user", content: [{ type: "text", text: "Hello" }] };
  const reply = (id: string, speakerId: string, plan?: string[]) => ({
    id,
    role: "assistant",
    content: [{ type: "text", text: "..." }],
    metadata: { custom: { roleplay: { kind: "reply", speakerId, speakerName: speakerId, ...(plan ? { plan } : {}) } } },
  });
  const marker = (id: string, speakerId: string) => ({
    id,
    role: "user",
    content: [{ type: "text", text: "" }],
    metadata: { custom: { roleplay: { kind: "turn", speakerId } } },
  });
  assert.deepEqual(resolveSpeaker({ scene, cast, messages: [user] }), { speakerId: "a", plan: ["a", "b"] });
  assert.equal(nextInRound([user, reply("r1", "a", ["a", "b"])], scene), "b");
  // The marker names the speaker, so retrying Bo's reply keeps it Bo's.
  assert.deepEqual(
    resolveSpeaker({ scene, cast, messages: [user, reply("r1", "a", ["a", "b"]), marker("m", "b")] }),
    { speakerId: "b" },
  );
  assert.equal(nextInRound([user, reply("r1", "a", ["a", "b"]), marker("m", "b"), reply("r2", "b")], scene), null);
  const muted: ChatScene = { ...scene, cast: [scene.cast[0], { ...scene.cast[1], muted: true }] };
  assert.equal(nextInRound([user, reply("r1", "a", ["a", "b"])], muted), null, "a muted member is skipped");
  // A stopped reply ends the round, even after a reply asked for by hand.
  const stopped = { ...reply("r1", "a", ["a", "b"]), status: { type: "incomplete" } };
  assert.equal(nextInRound([user, stopped, marker("m", "a"), reply("r2", "a")], scene), null);
  assert.deepEqual(resolveSpeaker({ scene, cast, messages: [user], requested: "b" }), { speakerId: "b" });
  // A one-on-one chat never plans a round.
  assert.deepEqual(resolveSpeaker({ scene, cast: [cast[0]], messages: [user] }), { speakerId: "a" });
  assert.equal(textOf({ content: [{ type: "text", text: "x" }, { type: "image" }, { type: "text", text: "y" }] }), "x\n\ny");
});

test("a card's own wrapper tags are hidden on screen so the block inside renders", () => {
  const text = "Look.\n\n<stats>\n```\nHP: 10\n```\n</stats>\n<details>\nkept\n</details>";
  assert.equal(tidyCharacterDisplay(text), "Look.\n\n\n```\nHP: 10\n```\n\n<details>\nkept\n</details>");
  assert.equal(tidyCharacterDisplay("a <b>bold</b> line"), "a <b>bold</b> line");
  const audio = 'Here.\n\n<audio-player src="data:audio/wav;base64,UklGRg==" />';
  assert.equal(tidyCharacterDisplay(audio), audio, "an audio-output reply keeps its player");
});

test("opening retries keep the first member, notes skip a trailing reply, code keeps its tags", () => {
  const cast = [member("a", "Ann"), member("b", "Bo")];
  const scene: ChatScene = {
    threadId: "t",
    personaId: null,
    settings: {},
    cast: [
      { characterId: "a", position: 0, muted: false },
      { characterId: "b", position: 1, muted: false },
    ],
  };
  const opening = {
    id: "o",
    role: "user",
    content: [{ type: "text", text: "[Start a new chat]" }],
    metadata: { custom: { roleplay: { kind: "opening" } } },
  };
  assert.deepEqual(resolveSpeaker({ scene, cast, messages: [opening] }), { speakerId: "a" });

  const built = buildCharacterPrompt({
    cast: [character({ id: "a", name: "Ann", postHistoryInstructions: "Stay brief." })],
    speakerId: "a",
    persona: { name: "Pat", description: "" },
    scene: {},
    history: [
      { role: "user", text: "Roll {{roll:d20}} for {{char}}" },
      { role: "assistant", text: "Done." },
    ],
  });
  assert.equal(built.texts[0], "Roll {{roll:d20}} for Ann", "history fixes only the names");
  assert.equal(built.texts[1], "Done.", "no final instructions on a reply");

  const code = "```html\n<template>\n</template>\n```\n<stats>\nok\n</stats>";
  assert.equal(tidyCharacterDisplay(code), "```html\n<template>\n</template>\n```\n\nok\n");
});

test("a long lorebook regex key is ignored rather than run", () => {
  const long = `/${"(a+)+".repeat(40)}$/`;
  const fired = activateLorebook(book([entry({ keys: [long], content: "never" })]), ["aaaa"]);
  assert.equal(fired.fired.length, 0);
});

test("lorebook regex keys run in linear time and match as JS regexes do", () => {
  const started = Date.now();
  for (const key of ["/(a+)+$/", "/a+a+a+a+a+a+$/", "/(a|a)*(a?){20}b/", "/(?:(?:(?:(?:(?:a{0}){100}){100}){100}){100}){100}/"]) {
    const fired = activateLorebook(book([entry({ keys: [key], content: "never" })]), [`${"a".repeat(3000)}!`]);
    assert.equal(fired.fired.length, 0, key);
  }
  assert.ok(Date.now() - started < 1500);
  const patterns: [string, string][] = [
    ["dra(g|k)ons?\\b", "i"],
    ["^\\s*(?:big |huge )?cat$", "m"],
    ["[^aeiou\\s]{3,}", ""],
    ["\\bsword(s|smith)?\\b", "i"],
    ["\\p{Lu}\\p{Ll}+", "u"],
    ["a.c", "s"],
    ["(?<who>Rin|Rhys):", ""],
    ["x{2}y{1,}z?$", ""],
    ["dragon", "y"],
    ["Two", "iy"],
    ["[(]+x", ""],
    ["\u{20BB7}野", "u"],
    ["\u{1F600}+!", "u"],
    ["[\u{1F600}]", "u"],
  ];
  const texts = ["\u{20BB7}野", "\u{1F600}\u{1F600}!", "\u{1F600}", "Two DRAKONS", "cat", "a big cat\nhuge cat", "strength", "swordsmith", "Rin: hi", "Hello", "a\nc", "xxyyy", "((x", "", "dragonfly"];
  for (const [source, flags] of patterns) {
    for (const text of texts) {
      const fired = activateLorebook(book([entry({ keys: [`/${source}/${flags}`], content: "c" })]), [text]);
      assert.equal(fired.fired.length === 1, new RegExp(source, flags).test(text), `/${source}/${flags} on ${JSON.stringify(text)}`);
    }
  }
  assert.equal(activateLorebook(book([entry({ keys: ["a".repeat(50_000)], content: "c" })]), ["hello"]).fired.length, 0);
  const v3 = activateLorebook(book([entry({ keys: ["dragon|wyrm"], use_regex: true, content: "c" })]), ["A WYRM lands"]);
  assert.equal(v3.fired.length, 1, "a V3 use_regex key is a pattern without slashes");
  const long = activateLorebook(book([entry({ keys: ["dragon"], content: "c" })], { scan_depth: 2 }), ["ok", `${"x".repeat(16_000)} A dragon appears.`]);
  assert.equal(long.fired.length, 1, "the newest text of a long reply is scanned");
  for (const key of ["/(a)\\1/", "/(?=a)a/", "/(?<!x)a/"]) {
    assert.equal(activateLorebook(book([entry({ keys: [key], content: "c" })]), ["aa"]).fired.length, 0, key);
  }
});

test("card text with long unclosed macros and tags stays linear", () => {
  const ctx = { char: "Bo", user: "Ann", now: new Date(0), group: [] };
  const spaces = " ".repeat(100_000);
  const started = Date.now();
  applyMacros(`{{${spaces}x`, ctx);
  applyMacros("{{//".repeat(20_000) + "tail", ctx);
  applyMacros(`{{roll:d1${spaces}x}}`, ctx);
  tidyCharacterDisplay(`<a${"b".repeat(100_000)}`);
  assert.ok(Date.now() - started < 1000);
  assert.equal(applyMacros("a{{// note }}b{{ char }}{{//open", ctx), "abBo{{//open");
  assert.equal(applyMacros("{{roll: 2d1 + 3 }}", ctx), "5");
  assert.equal(tidyCharacterDisplay("<stats>\nok\n</stats>"), "\nok\n");
});

test("names with $ patterns are inserted literally", () => {
  const ctx = { char: "$'", user: "A$&B", now: new Date(0), group: [] };
  assert.equal(applyMacros("<BOT> {{char}} <USER> {{user}}", ctx), "$' $' A$&B A$&B");
  assert.equal(applyMacros("<BOT>".repeat(16_000), ctx).length, 32_000);
});

test("a group reduced to one character keeps its history attributed", () => {
  const built = buildCharacterPrompt({
    cast: [character({ id: "ann", name: "Ann" })],
    speakerId: "ann",
    persona: { name: "Pat", description: "" },
    scene: {},
    history: [
      { role: "user", text: "[Start]", meta: { kind: "opening" } },
      { role: "assistant", text: "Hi.", meta: { kind: "reply", speakerId: "ann", speakerName: "Ann" } },
      { role: "user", text: "[Bob's turn]", meta: { kind: "turn", speakerId: "bob" } },
      { role: "assistant", text: "Yo.", meta: { kind: "reply", speakerId: "bob", speakerName: "Bob" } },
      { role: "user", text: "Hello" },
    ],
    now: NOW,
  });
  assert.equal(built.texts[1], "Ann: Hi.");
  assert.equal(built.texts[2], nudgeText("Bob"));
  assert.equal(built.texts[3], "Bob: Yo.");
  assert.ok(built.stop.includes("\nBob:") && !built.stop.includes("\nAnn:"));
  const solo = buildCharacterPrompt({
    cast: [character({ id: "ann", name: "Ann" })],
    speakerId: "ann",
    persona: { name: "Pat", description: "" },
    scene: {},
    history: [{ role: "assistant", text: "Hi.", meta: { kind: "reply", speakerId: "ann", speakerName: "Ann" } }],
    now: NOW,
  });
  assert.equal(solo.texts[0], "Hi.");
  assert.equal(solo.namesTurns, false);
});

test("a hidden turn with no reply after it is not sent", () => {
  const built = buildCharacterPrompt({
    cast: [character({ id: "ann", name: "Ann" })],
    speakerId: "ann",
    persona: { name: "Pat", description: "" },
    scene: { authorsNote: { text: "Rain.", depth: 1 } },
    history: [
      { role: "user", text: "[Start]", meta: { kind: "opening" } },
      { role: "user", text: "Hello" },
    ],
    now: NOW,
  });
  assert.deepEqual(built.sent, [false, true]);
  assert.ok(built.texts[1].startsWith("Hello") && built.texts[1].includes("[Rain.]"));
});

test("a renamed character's old replies replay under the new name only", () => {
  const built = buildCharacterPrompt({
    cast: [character({ id: "rin", name: "Nova" }), character({ id: "bo", name: "Bo" })],
    speakerId: "bo",
    persona: { name: "Pat", description: "" },
    scene: {},
    history: [{ role: "assistant", text: "Rin: Hello", meta: { kind: "reply", speakerId: "rin", speakerName: "Rin" } }],
    now: NOW,
  });
  assert.equal(built.texts[0], "Nova: Hello");
});

test("lorebook keys split on commas, except inside a regex key", () => {
  assert.deepEqual(parseKeyList(" sword, /a{2,4}/i , /[,/]x/,blade ,, "), ["sword", "/a{2,4}/i", "/[,/]x/", "blade"]);
});

test("a replayed reply with tools between its texts keeps their order", () => {
  const tool = { type: "tool-call", toolName: "web_search" };
  const reply = { role: "assistant", content: [{ type: "text", text: "Bo: Let me look." }, tool, { type: "text", text: "Found it." }] };
  assert.deepEqual(shapeReplayMessage(reply, "unused", "Bo", true).content, [
    { type: "text", text: "Bo: Let me look." },
    tool,
    { type: "text", text: "Found it." },
  ]);
  const plain = { role: "assistant", content: [tool, { type: "text", text: "Bo: Hi" }] };
  assert.deepEqual(shapeReplayMessage(plain, "Bo: Hi.", "Bo", true).content, [tool, { type: "text", text: "Bo: Hi." }]);
});

test("{{trim}} cleanup stays linear on a long whitespace run", () => {
  const ctx = { char: "Bo", user: "Ann", now: new Date(0), group: [] };
  assert.equal(applyMacros("a  \n{{trim}}\n  b {{ trim }}c", ctx), "abc");
  const started = Date.now();
  applyMacros(`x${" ".repeat(200_000)}y`, ctx);
  assert.ok(Date.now() - started < 500);
});

test("a title is not a name: mentions and labels skip Captain, Professor and the like", () => {
  assert.equal(shortName("Captain Rhys Calloway"), "Rhys");
  assert.equal(shortName("Dr. Moon"), "Moon");
  assert.equal(shortName("Elara Voss"), "Elara");
  assert.equal(shortName("Captain"), "Captain");
  const cast = [
    { id: "elara", name: "Elara Voss", talkativeness: 0.5, muted: false },
    { id: "rhys", name: "Captain Rhys Calloway", talkativeness: 0.5, muted: false },
  ];
  // "captain's coat" is not Rhys; "Elara" is the one named.
  assert.deepEqual(mentionedIds("The man in the captain's coat stared. Elara, where are we?", cast), ["elara"]);
  assert.deepEqual(mentionedIds("Rhys, and you, Elara?", cast), ["rhys", "elara"]);
  assert.deepEqual(mentionedIds("Ask Captain Rhys Calloway first.", cast), ["rhys"]);
  // A title only one character holds calls them when used as a form of address, and so does a surname.
  assert.deepEqual(mentionedIds("Captain, you've gone quiet.", cast), ["rhys"]);
  assert.deepEqual(mentionedIds("What do you think, Calloway? And you, Voss?", cast), ["rhys", "elara"]);
  assert.deepEqual(mentionedIds("I admire the captain's coat.", cast), []);
  const two = [...cast, { id: "mara", name: "Captain Mara Okafor", talkativeness: 0.5, muted: false }];
  assert.deepEqual(mentionedIds("Captain, a word?", two), [], "a shared title names nobody");
});

test("natural order: the named character answers, an open line gets one voice, rarely two", () => {
  const cast = [
    { id: "elara", name: "Elara Voss", talkativeness: 0.5, muted: false },
    { id: "rin", name: "Rin Takeda", talkativeness: 0.6, muted: false },
    { id: "rhys", name: "Captain Rhys Calloway", talkativeness: 0.45, muted: false },
  ];
  const seeds = Array.from({ length: 400 }, (_, i) => `msg-${i}`);
  const named = seeds.map((seed) => planReplies({ cast, order: "natural", userText: "Elara, where are we?", lastSpeakerId: "rin", seed }));
  assert.ok(named.every((plan) => plan[0] === "elara" && plan.length <= 2));
  const namedExtra = named.filter((plan) => plan.length === 2).length / seeds.length;
  assert.ok(namedExtra > 0.05 && namedExtra < 0.25, `a named answer is rarely joined (${namedExtra})`);

  const open = seeds.map((seed) => planReplies({ cast, order: "natural", userText: "What now?", lastSpeakerId: "rin", seed }));
  assert.ok(open.every((plan) => plan.length >= 1 && plan.length <= 2 && plan[0] !== "rin"));
  const openExtra = open.filter((plan) => plan.length === 2).length / seeds.length;
  assert.ok(openExtra > 0.18 && openExtra < 0.38, `an open line often gets a second voice (${openExtra})`);
  const leads = new Set(open.map((plan) => plan[0]));
  assert.deepEqual([...leads].sort(), ["elara", "rhys"], "anyone but the last speaker can lead");

  // Several named characters all answer, in the order named.
  assert.deepEqual(
    planReplies({ cast, order: "natural", userText: "Rhys and Elara, help!", lastSpeakerId: null, seed: "s" }).slice(0, 2),
    ["rhys", "elara"],
  );
  // Back to back only when allowed or named.
  const self = planReplies({ cast: cast.slice(0, 2), order: "natural", userText: "", lastSpeakerId: "elara", seed: "q" });
  assert.equal(self[0], "rin");
});

test("a card's regex keys share one matching budget per prompt", () => {
  const entries = Array.from({ length: 400 }, (_, i) =>
    entry({ id: `r${i}`, keys: [`/(x|y|z)*${i}q/`], content: `lore ${i}` }),
  );
  const lore = book([...entries, entry({ id: "word", keys: ["dragon"], content: "a keyword still fires" })], {
    token_budget: 1_000_000,
  });
  const text = `${"xyz".repeat(5000)} dragon ${Array.from({ length: 400 }, (_, i) => `${i}q`).join(" ")}`;
  const started = performance.now();
  const fired = activateLorebook(lore, [text]);
  const elapsed = performance.now() - started;
  const regexFired = fired.fired.filter((f) => String(f.id).startsWith("r")).length;
  assert.ok(regexFired > 0 && regexFired < 400, `${regexFired} regex keys fired`);
  assert.ok(fired.fired.some((f) => f.id === "word"));
  assert.ok(elapsed < 1_000, `took ${elapsed} ms`);
});

test("the regex budget counts matching work, not keys", () => {
  // Few keys, each one heavy: a key-count cap let 200 of these take seconds.
  const entries = Array.from({ length: 200 }, (_, i) =>
    entry({ id: `h${i}`, keys: [`/(?:.?){100}A${i}/s`], content: `lore ${i}` }),
  );
  const started = performance.now();
  activateLorebook(book(entries), ["x".repeat(16_000)]);
  const elapsed = performance.now() - started;
  assert.ok(elapsed < 1_000, `took ${elapsed} ms`);
});

test("keys and names in scripts without spaces fire inside a sentence; a card can turn whole words off", () => {
  assert.equal(activateLorebook(book([entry({ keys: ["東京"], content: "Tokyo" })]), ["私は東京に行く"]).fired.length, 1);
  const loose = entry({ keys: ["elf"], content: "Elves", extensions: { match_whole_words: false } });
  assert.equal(activateLorebook(book([loose]), ["an elfish grin"]).fired.length, 1);
  assert.equal(activateLorebook(book([entry({ keys: ["elf"], content: "Elves" })]), ["an elfish grin"]).fired.length, 0);
  const cast = [{ id: "x", name: "小明", talkativeness: 0.5, muted: false }];
  assert.deepEqual(mentionedIds("你好小明今天怎么样", cast), ["x"]);
});

test("recursive scanning lets fired lore trigger more lore, unless an entry opts out", () => {
  const entries = [
    entry({ id: "a", keys: ["castle"], content: "The castle has a dragon." }),
    entry({ id: "b", keys: ["dragon"], content: "The dragon sleeps." }),
    entry({ id: "c", keys: ["sleeps"], content: "Nobody wakes it.", extensions: { exclude_recursion: true } }),
  ];
  const ids = (on: boolean) => activateLorebook(book(entries, { recursive_scanning: on }), ["We reach the castle."]).fired.map((f) => f.id);
  assert.deepEqual(ids(false), ["a"]);
  assert.deepEqual(ids(true), ["a", "b"]);
});

test("over budget, higher priority stays; what stays is placed by insertion order", () => {
  const lore = book(
    [
      entry({ id: "low", keys: ["x"], content: "L".repeat(40), insertion_order: 1, priority: 1 }),
      entry({ id: "high", keys: ["x"], content: "H".repeat(40), insertion_order: 2, priority: 9 }),
      entry({ id: "mid", keys: ["x"], content: "M".repeat(4), insertion_order: 0, priority: 5 }),
    ],
    { token_budget: 12 },
  );
  assert.deepEqual(activateLorebook(lore, ["x"]).fired.map((f) => f.id), ["mid", "high"]);
});

test("SillyTavern exports match as SillyTavern does: regex only in slashes, substrings by default", () => {
  // SillyTavern sets use_regex on every exported entry and writes match_whole_words: null.
  const st = (keys: string[], extensions: Record<string, unknown> = {}) =>
    entry({ keys, use_regex: true, content: "c", extensions: { position: 0, display_index: 0, match_whole_words: null, ...extensions } });
  const fires = (e: LorebookEntry, text: string) => activateLorebook(book([e]), [text]).fired.length === 1;
  assert.equal(fires(st(["[Mira]"]), "Hello there."), false, "a bracketed name is not a character class");
  assert.equal(fires(st(["[Mira]"]), "Have you met [Mira]?"), true);
  assert.equal(fires(st(["C++"]), "The ship runs C++ code."), true);
  assert.equal(fires(st(["dragon"]), "The dragons attacked."), true, "substring match by default");
  assert.equal(fires(st(["dragon"], { match_whole_words: true }), "The dragons attacked."), false);
  assert.equal(fires(st(["/wyrm|drake/i"]), "A DRAKE lands"), true, "a slashed key is still a regex");
});

test("recursive scanning reads always-on lore, except entries that opt out", () => {
  const setting = (extensions = {}) => entry({ id: 0, constant: true, content: "The valley is guarded by a dragon.", extensions });
  const dragon = (extensions = {}) => entry({ id: 1, keys: ["dragon"], content: "The dragon is named Vex.", extensions });
  const fired = (entries: LorebookEntry[], recursive: boolean) =>
    activateLorebook(book(entries, { recursive_scanning: recursive }), ["hello"]).fired.map((f) => f.id);
  assert.deepEqual(fired([setting(), dragon()], true), [0, 1]);
  assert.deepEqual(fired([setting(), dragon()], false), [0]);
  assert.deepEqual(fired([setting({ prevent_recursion: true }), dragon()], true), [0]);
  assert.deepEqual(fired([setting(), dragon({ exclude_recursion: true })], true), [0]);
});

test("one lore budget covers the whole cast", () => {
  const member = (id: string) =>
    character({ id, name: id, characterBook: book([entry({ constant: true, content: "x".repeat(2000) })], { token_budget: 1024 }) });
  const built = buildCharacterPrompt({
    cast: [member("A"), member("B"), member("C")],
    speakerId: "A",
    persona: { name: "Pat", description: "" },
    scene: {},
    history: [{ role: "user", text: "hi" }],
    now: NOW,
  });
  assert.equal(built.lore.fired.length, 2, "1024 tokens fit two 500-token entries, not one per member");
});

test("a reply from before the chat had a cast keeps one author whoever speaks", () => {
  const mira = character({ id: "m", name: "Mira" });
  const bo = character({ id: "b", name: "Bo" });
  const history = [
    { role: "user" as const, text: "hello" },
    { role: "assistant" as const, text: "Hi, I'm Mira." },
    { role: "user" as const, text: "and you?" },
  ];
  const as = (speakerId: string) =>
    buildCharacterPrompt({ cast: [mira, bo], speakerId, persona: { name: "Pat", description: "" }, scene: {}, history, now: NOW });
  assert.equal(as("b").texts[1], "Mira: Hi, I'm Mira.");
  assert.deepEqual(as("b").texts.slice(0, 2), as("m").texts.slice(0, 2));
});

test("names that read like macros are inserted as written", () => {
  const ctx = { char: "{{random::A::B}}", user: "{{trim}}", now: NOW, random: () => 0.9 };
  assert.equal(applyMacros("<BOT> waves", ctx), "{{random::A::B}} waves");
  assert.equal(applyMacros("hi <USER> there", ctx), "hi {{trim}} there");
  assert.equal(applyNameMacros("{{char}} and {{user}}", "{{user}}", "Pat"), "{{user}} and Pat");
});

test("the system block rolls and dates the same on every send of a chat", () => {
  const card = character({ id: "m", name: "Mira", description: "Mood: {{random::calm::wild::sly}}, {{pick::a::b::c}}, at {{isotime}}." });
  const build = (now: Date, history: { role: "user" | "assistant"; text: string }[]) =>
    buildCharacterPrompt({ cast: [card], speakerId: "m", persona: { name: "Pat", description: "" }, scene: {}, history, now, startedAt: NOW, seed: "thread-1" });
  const first = build(new Date(2026, 10, 4, 9, 6), [{ role: "user", text: "hi" }]);
  const later = build(new Date(2026, 10, 5, 18, 0), [{ role: "user", text: "hi" }, { role: "assistant", text: "yo" }, { role: "user", text: "again" }]);
  assert.equal(first.system, later.system);
  assert.match(first.system, /at 09:05\./);
});

test("regex keys read legacy escapes as JS does and reject a quantifier with nothing to repeat", () => {
  const fires = (key: string, text: string) => activateLorebook(book([entry({ keys: [key], content: "c" })]), [text]).fired.length;
  assert.equal(fires("/\\xyz/", "xyz"), 1);
  assert.equal(fires("/\\p{L}/", "p{L}"), 1);
  assert.equal(fires("/a**/", "a*"), 0);
});

test("one at a time hands on past a muted last speaker; a broken talkativeness is ignored", () => {
  const member = (id: string, muted = false, talkativeness = 0.5) => ({ id, name: id, talkativeness, muted });
  const manual = planReplies({ cast: [member("a"), member("b", true), member("c")], order: "manual", userText: "", lastSpeakerId: "b", seed: "x" });
  assert.deepEqual(manual, ["c"]);
  const seen = new Set<string>();
  for (let i = 0; i < 40; i += 1) {
    const plan = planReplies({ cast: [member("a", false, NaN), member("b", false, NaN), member("c", false, NaN)], order: "natural", userText: "", lastSpeakerId: null, seed: `s${i}` });
    seen.add(plan[0]);
  }
  assert.ok(seen.size > 1, "NaN no longer always picks the last member");
});

test("display keeps code fences by length, indented code, and ordinary HTML", () => {
  assert.equal(tidyCharacterDisplay("````\n```\n<stats>\n````"), "````\n```\n<stats>\n````");
  assert.equal(tidyCharacterDisplay("    <stats>\n    x"), "    <stats>\n    x");
  assert.equal(tidyCharacterDisplay("<center>\nhi\n</center>"), "<center>\nhi\n</center>");
});

test("writing the user's line drops the character's final instructions and nudges", () => {
  const mira = character({ id: "m", name: "Mira", postHistoryInstructions: "Mira answers in rhyme." });
  const bo = character({ id: "b", name: "Bo" });
  const prompt = buildImpersonationMessages({
    cast: [mira, bo],
    persona: { name: "Pat", description: "" },
    scene: {},
    history: [{ role: "user", text: "hi" }],
    now: NOW,
  });
  const last = prompt.messages[prompt.messages.length - 1].content;
  assert.doesNotMatch(last, /rhyme|Write the next reply only as/);
  assert.match(last, /Write Pat's next message now/);
});

test("a persona named like a character is told apart in turns and stop strings", () => {
  const rin = character({ id: "r", name: "Rin" });
  const bo = character({ id: "b", name: "Bo" });
  const built = buildCharacterPrompt({
    cast: [rin, bo],
    speakerId: "r",
    persona: { name: "Rin", description: "A traveller." },
    scene: {},
    history: [{ role: "user", text: "hello {{user}}" }],
    now: NOW,
  });
  assert.equal(built.user, "Rin (you)");
  assert.match(built.texts[0], /^Rin \(you\): hello Rin/);
  assert.deepEqual(built.stop, ["\nRin (you):", "\nBo:"]);
  assert.ok(built.sections.some((section) => section.label === "Rin (you)"));
});

test("speaker prefixes: short names, headings, and a quote that belongs to the line", () => {
  assert.equal(stripSpeakerPrefix("Mara: Hello.", "Mara Quill"), "Hello.");
  assert.equal(stripSpeakerPrefix("**Mara**: Hello.", "Mara Quill"), "Hello.");
  assert.equal(stripSpeakerPrefix("## Mara Quill\nHello.", "Mara Quill"), "Hello.");
  assert.equal(stripSpeakerPrefix('"Mara: Hello there."', "Mara"), '"Hello there."');
  assert.equal(stripSpeakerPrefix('"Mara: "Hi," she says.', "Mara"), '"Hi," she says.');
  assert.equal(stripSpeakerPrefix("## Mara's diary\nDay one.", "Mara"), "## Mara's diary\nDay one.");
});

test("hidden roleplay turns report their kind", () => {
  const meta = (kind: string) => ({ custom: { roleplay: { kind, speakerId: "a" } } });
  assert.equal(hiddenRoleplayTurnKind({ role: "user", metadata: meta("opening") }), "opening");
  assert.equal(hiddenRoleplayTurnKind({ role: "user", metadata: meta("turn") }), "turn");
  assert.equal(hiddenRoleplayTurnKind({ role: "assistant", metadata: meta("turn") }), null);
  assert.equal(hiddenRoleplayTurnKind({ role: "user", metadata: {} }), null);
});
