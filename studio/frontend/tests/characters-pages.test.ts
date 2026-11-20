// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { isCharacterChat } from "../src/features/characters/chat-gate.ts";
import {
  addTags,
  blankDraft,
  clampNumber,
  MAX_TAGS,
  permanentPromptText,
  toSavePayload,
} from "../src/features/characters/draft.ts";
import {
  filterCharacters,
  groupTags,
  liveSelection,
  sortCharacters,
} from "../src/features/characters/library-filters.ts";
import type { Character } from "../src/features/characters/types.ts";

const read = (path: string) => readFileSync(new URL(`../src/${path}`, import.meta.url), "utf8");

function character(id: string, extra: Partial<Character> = {}): Character {
  return {
    id, name: id, description: "", personality: "", scenario: "", firstMes: "", alternateGreetings: [],
    mesExample: "", systemPrompt: "", postHistoryInstructions: "", creatorNotes: "", creator: "",
    characterVersion: "", tags: [], talkativeness: 0.5, favorite: false, depthPrompt: null, characterBook: null,
    extensions: {}, hasAvatar: false, avatarType: null, avatarRev: 0, createdAt: 1, updatedAt: 1, lastChatAt: null,
    chatCount: 0, cardSpec: null, ...extra,
  };
}

test("tags dedupe within one paste and across letter case, and respect the server's limits", () => {
  assert.deepEqual(addTags([], "a, a"), ["a"]);
  assert.deepEqual(addTags(["Fantasy"], "fantasy, Sci-fi"), ["Fantasy", "Sci-fi"]);
  assert.equal(addTags([], "x".repeat(150))[0].length, 100);
  const full = Array.from({ length: MAX_TAGS }, (_, i) => `t${i}`);
  assert.equal(addTags(full, "one more").length, MAX_TAGS);
  const same = ["a"];
  assert.equal(addTags(same, " , A "), same);
});

test("a cleared number field falls back instead of snapping mid-edit, and clamps when left", () => {
  assert.equal(clampNumber(Number.NaN, 64, undefined, 1024), 1024);
  assert.equal(clampNumber(2, 64), 64);
  assert.equal(clampNumber(642, 64), 642);
  assert.equal(clampNumber(250, 0, 100), 100);
  assert.equal(clampNumber(-3), -3);
  const field = read("features/characters/components/clamped-number-input.tsx");
  assert.match(field, /onBlur=\{\(\) => \{/);
  assert.doesNotMatch(read("features/characters/components/editor/lorebook-editor.tsx"), /Number\(event\.target\.value\) \|\|/);
});

test("the save body clears an empty note and book and drops blank alternate greetings", () => {
  const draft = { ...blankDraft(), name: " Mara ", alternateGreetings: ["hi", "  "], depthPrompt: { prompt: " ", depth: 4, role: "system" as const } };
  const { fields, clear } = toSavePayload(draft);
  assert.equal(fields.name, "Mara");
  assert.deepEqual(fields.alternateGreetings, ["hi"]);
  assert.equal("depthPrompt" in fields, false);
  assert.deepEqual(clear, ["depthPrompt", "characterBook"]);
});

test("the token estimate counts every field sent each turn", () => {
  const draft = { ...blankDraft(), description: "d", postHistoryInstructions: "p", depthPrompt: { prompt: "n", depth: 4, role: "system" as const } };
  assert.equal(permanentPromptText(draft), "d\np\nn");
});

test("library filters group tags by case and keep only live selections", () => {
  const list = [
    character("a", { tags: ["Fantasy"], favorite: true }),
    character("b", { tags: ["fantasy", "Noir"], createdAt: 5 }),
    character("c", { tags: ["fantasy"], creator: "Ann" }),
  ];
  assert.deepEqual(groupTags(list)[0], { key: "fantasy", count: 3, label: "fantasy" });
  assert.deepEqual(filterCharacters(list, { query: "", favoritesOnly: false, tag: "noir" }).map((c) => c.id), ["b"]);
  assert.deepEqual(filterCharacters(list, { query: "ann", favoritesOnly: false, tag: null }).map((c) => c.id), ["c"]);
  assert.deepEqual(filterCharacters(list, { query: "", favoritesOnly: true, tag: null }).map((c) => c.id), ["a"]);
  assert.deepEqual(sortCharacters(list, "newest").map((c) => c.id)[0], "b");
  assert.deepEqual(liveSelection(["a", "gone", "c"], { a: list[0], c: list[2] }), ["a", "c"]);
});

test("ordinary chats skip the scene fetch once the chat index has loaded", () => {
  const state = { chatIndex: { rp: ["a"] }, indexLoaded: false, scenes: { plain: null } };
  assert.equal(isCharacterChat("new", false)(state), true);
  assert.equal(isCharacterChat("plain", false)(state), false);
  assert.equal(isCharacterChat(null, true)(state), false);
  const loaded = { ...state, indexLoaded: true };
  assert.equal(isCharacterChat("new", false)(loaded), false);
  assert.equal(isCharacterChat("new", true)(loaded), true);
  assert.equal(isCharacterChat("rp", false)(loaded), true);
  assert.match(read("features/characters/store.ts"), /set\(\{ chatIndex, indexLoaded: true \}\)/);
});

test("import warnings reach the user and a failed duplicate says what failed", () => {
  const actions = read("features/characters/actions.ts");
  assert.match(actions, /if \(result\.warnings\.length\) \{\s*toast\.warning\(/);
  assert.match(actions, /toast\.error\(t\("characters\.duplicateFailed"\)/);
  assert.match(actions, /openFilePicker\(CARD_ACCEPT/);
});

test("a failed personas load offers a retry instead of a skeleton forever", () => {
  const list = read("features/characters/components/personas-list.tsx");
  assert.match(list, /const loading = status === "idle" \|\| status === "loading";/);
  assert.match(list, /status === "error" && personas\.length === 0 \? \(\s*<LoadFailed/);
});

test("saving then failing to open the chat keeps the save and leaves no unsaved-changes prompt", () => {
  const hook = read("features/characters/hooks/use-character-draft.ts");
  assert.match(hook, /setBaseline\(JSON\.stringify\(draft\)\);/);
  const chatFailure = hook.slice(hook.indexOf("const opened = await openChat"));
  assert.match(chatFailure, /if \(!opened\) profile\(\);/);
  assert.doesNotMatch(chatFailure, /saveFailed/);
  const actions = read("features/characters/actions.ts");
  assert.match(actions.slice(actions.indexOf("export function useOpenCharacterChat")), /toast\.error\(t\("characters\.chat\.startFailed"\)/);
});

test("a new picture revision replaces the old object URL instead of leaking it", () => {
  const images = read("features/characters/images.ts");
  assert.match(images, /new BlobUrlCache\(/);
  assert.match(images, /cache\.set\(key, url, blob\.size\)/);
  assert.doesNotMatch(images, /const resolved = new Map/);
});

test("the sidebar re-reads the chat index only when another tab changes chats", () => {
  const mark = read("features/characters/sidebar-mark.tsx");
  assert.doesNotMatch(mark, /subscribeChatHistoryUpdated/);
  assert.match(mark, /event\.key === CHAT_HISTORY_REVISION_KEY\) refreshSoon\(\)/);
  assert.match(mark, /store\.status === "idle" \|\| store\.status === "error"/);
});

test("the profile resets per character, says not found only on a 404, and greets the persona", () => {
  assert.match(read("app/routes/characters.$characterId.tsx"), /<CharacterProfilePage key=\{characterId\}/);
  const profile = read("features/characters/pages/character-profile-page.tsx");
  assert.match(profile, /isNotFound\(error\) && setMissing\(true\)/);
  assert.match(profile, /resolvePersona\(null, personas, accountName\)\.name/);
  assert.doesNotMatch(profile, /user: "\{\{user\}\}"/);
});

test("scene writes from a previous account are not applied", () => {
  const store = read("features/characters/store.ts");
  const patch = store.slice(store.indexOf("patchScene: (threadId, patch) =>"));
  assert.match(patch, /const started = epoch;\s*const scene = await updateChatScene\(threadId, patch\);\s*if \(started !== epoch\) return scene;/);
});
