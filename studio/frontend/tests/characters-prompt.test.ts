// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import {
  buildCharacterPrompt,
  nudgeText,
  shapeReplayMessage,
  stripSpeakerPrefix,
  type PromptCharacter,
} from "../src/features/characters/prompt/build.ts";
import { activateLorebook, parseKeyList } from "../src/features/characters/prompt/lorebook.ts";
import { applyMacros } from "../src/features/characters/prompt/macros.ts";
import { tidyRoleplayDisplay } from "../src/features/characters/prompt/display.ts";
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
  assert.equal(tidyRoleplayDisplay(text), "Look.\n\n\n```\nHP: 10\n```\n\n<details>\nkept\n</details>");
  assert.equal(tidyRoleplayDisplay("a <b>bold</b> line"), "a <b>bold</b> line");
});

test("review fixes: opening retries keep the first member, notes skip a trailing reply, code keeps its tags", () => {
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
  assert.equal(tidyRoleplayDisplay(code), "```html\n<template>\n</template>\n```\n\nok\n");
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
  tidyRoleplayDisplay(`<a${"b".repeat(100_000)}`);
  assert.ok(Date.now() - started < 1000);
  assert.equal(applyMacros("a{{// note }}b{{ char }}{{//open", ctx), "abBo{{//open");
  assert.equal(applyMacros("{{roll: 2d1 + 3 }}", ctx), "5");
  assert.equal(tidyRoleplayDisplay("<stats>\nok\n</stats>"), "\nok\n");
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
  assert.equal(solo.named, false);
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
  assert.equal(fired.fired.filter((f) => String(f.id).startsWith("r")).length, 128);
  assert.ok(fired.fired.some((f) => f.id === "word"));
  assert.ok(elapsed < 5_000, `took ${elapsed} ms`);
});
