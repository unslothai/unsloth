// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { mergeCharacters } from "../src/features/characters/meta.ts";
import type { Character } from "../src/features/characters/types.ts";

const read = (path: string) => readFileSync(new URL(`../src/${path}`, import.meta.url), "utf8");

function character(id: string, updatedAt = 1): Character {
  return {
    id, name: id, description: "", personality: "", scenario: "", firstMes: "", alternateGreetings: [],
    mesExample: "", systemPrompt: "", postHistoryInstructions: "", creatorNotes: "", creator: "",
    characterVersion: "", tags: [], talkativeness: 0.5, favorite: false, depthPrompt: null, characterBook: null,
    extensions: {}, hasAvatar: false, avatarType: null, avatarRev: 0, createdAt: 1, updatedAt, lastChatAt: null,
    chatCount: 0, cardSpec: null,
  };
}

test("a library refetch keeps unchanged records, and the whole map when nothing changed", () => {
  const previous = { a: character("a"), b: character("b") };
  assert.equal(mergeCharacters(previous, [character("a"), character("b")]), previous);
  const edited = mergeCharacters(previous, [character("a"), character("b", 2)]);
  assert.notEqual(edited, previous);
  assert.equal(edited.a, previous.a);
  assert.equal(edited.b.updatedAt, 2);
  assert.deepEqual(Object.keys(mergeCharacters(previous, [character("a")])), ["a"]);
});

test("the sidebar loads the character mark lazily, off the startup chunk", () => {
  const sidebar = read("components/app-sidebar.tsx");
  assert.doesNotMatch(sidebar, /^import [^;]*from "@\/features\/characters/m);
  assert.match(sidebar, /lazy\(\(\) =>\s*import\("@\/features\/characters\/sidebar-mark"\)/);
});

test("the library is fetched for the sidebar only once a character chat exists", () => {
  const mark = read("features/characters/sidebar-mark.tsx");
  assert.match(mark, /refreshIndex\(\)\s*\.then\(/);
  assert.match(mark, /Object\.keys\(store\.chatIndex\)\.length > 0\)/);
});

test("a send re-reads the cast and personas, not only the scene", () => {
  const runtime = read("features/characters/chat-runtime.ts");
  assert.match(runtime, /if \(args\.revalidate\) await revalidateParticipants\(scene\);/);
  assert.match(runtime, /getCharacter\(member\.characterId\)/);
});

test("a reply subscribes to its own speaker, not the whole library", () => {
  const parts = read("features/characters/chat/message-parts.tsx");
  assert.doesNotMatch(parts, /useCharactersStore\(\(s\) => s\.characters\)/);
});

test("a scene text field keeps what is being typed when an earlier save answers", () => {
  const panel = read("features/characters/chat/roleplay-panel.tsx");
  const field = panel.slice(panel.indexOf("function SavedTextarea"));
  assert.match(field, /if \(!editing\.current\) setDraft\(value\);/);
  assert.match(field, /onFocus=\{\(\) => \{\s*editing\.current = true;/);
  assert.match(field, /onBlur=\{\(\) => \{\s*editing\.current = false;/);
});

test("the chat page loads the character panel and picker only when they open", () => {
  const sheet = read("features/chat/chat-settings-sheet.tsx");
  assert.doesNotMatch(sheet, /^import [^;]*from "@\/features\/characters\/chat\/roleplay-panel"/m);
  assert.match(sheet, /lazy\(\(\) =>\s*import\("@\/features\/characters\/chat\/roleplay-panel"\)/);
  const thread = read("components/assistant-ui/thread.tsx");
  assert.doesNotMatch(thread, /^import [^;]*from "@\/features\/characters\/chat\/start-picker"/m);
  assert.match(thread, /lazy\(\(\) =>\s*import\("@\/features\/characters\/chat\/start-picker"\)/);
});
