// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Read aloud with an Unsloth speech model while chatting: the speech model and the chat
// model stay loaded together when Keep multiple models loaded allows it, and the settings say so.

import assert from "node:assert/strict";
import { readdirSync } from "node:fs";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const slotHook = readSrc("features/audio/hooks/use-audio-model-slot.ts");
const chatAdapter = readSrc("features/chat/api/chat-adapter.ts");
const chatApi = readSrc("features/chat/api/chat-api.ts");
const audioPage = readSrc("features/audio/audio-page.tsx");
const audioApi = readSrc("features/audio/api.ts");
const chatRuntime = readSrc("features/chat/hooks/use-chat-model-runtime.ts");
const voiceTab = readSrc("features/settings/tabs/voice-tab.tsx");
const en = readSrc("i18n/locales/en.ts");

test("a speech load from Audio keeps the chat model loaded", () => {
  // Without it, When they fit replaced the chat model and Always refused the load.
  assert.match(
    slotHook,
    /const res = await loadModel\([\s\S]*?alongside: true,[\s\S]*?runtime: "tts",/,
  );
  // The speech model it replaces is unloaded just before /load goes out: as a third model, Always
  // refuses a load that fits only in its place, and a load cancelled earlier keeps it.
  assert.match(
    slotHook,
    /audioModelToReplace\(statusRef\.current, loadId \|\| repoId\)[\s\S]*?const replacePrevious = previous\s*\? \(\) =>\s*unloadModel\(\{\s*model_path: audioSlotRoutingId\(previous\) \?\? previous,\s*\}\)[\s\S]*?const res = await loadModel\([\s\S]*?beforeRequest: replacePrevious,/,
  );
  assert.doesNotMatch(slotHook, /await unloadModel\(\{\s*model_path: audioSlotRoutingId\(previous\)/);
  // A refused unload stops the switch: swallowed, the busy old model kept answering Read aloud.
  assert.match(
    slotHook,
    /const replacePrevious = previous\s*\? \(\) =>\s*unloadModel\(\{\s*model_path: audioSlotRoutingId\(previous\) \?\? previous,\s*\}\)\s*: undefined;/,
  );
  assert.match(
    chatApi,
    /if \(!preparedToken\.proceed\)[\s\S]*?if \(options\?\.signal\?\.aborted\)[\s\S]*?await options\?\.beforeRequest\?\.\(\);\s*options\?\.onRequestStart\?\.\(\);/,
  );
  // Kept beside the chat model, the load leaves running chats alone, as Chat's keep-others load does.
  assert.match(
    slotHook,
    /const keepsChat = useChatRuntimeStore\.getState\(\)\.keepModelsLoaded;\s*const stopDecision = keepsChat\s*\? \{\s*proceed: true,\s*forceCancelActive: false,/,
  );
  // Its queue stops run only when the load can replace the chat model.
  assert.match(
    slotHook,
    /if \(!keepsChat\) \{\s*cancelPreStreamRunReservations\(stopDecision\.preStreamRunTokens\);\s*requestLocalPromptQueueStop\(stopDecision\.promptQueueThreadIds\);\s*\}/,
  );
  // Ejecting a speech model with the chat model kept beside it (in a slot, or active with the chat
  // model in a slot) asks nothing, stops no queue and unloads it by its routing id.
  assert.match(
    slotHook,
    /const slotModel = audioSlotRoutingId\(activeModel\);\s*const slotEject =\s*slotModel !== null \|\| \(statusRef\.current\?\.serving\?\.length \?\? 0\) > 1;/,
  );
  assert.match(
    slotHook,
    /if \(!slotEject\) \{\s*cancelPreStreamRunReservations\(stopDecision\.preStreamRunTokens\);\s*requestLocalPromptQueueStop\(stopDecision\.promptQueueThreadIds\);\s*\}\s*await unloadModel\(\{\s*model_path: slotModel \?\? activeModel,[\s\S]*?if \(!slotEject\) requestLocalPromptQueueStop\(\);/,
  );
  // What the load unloaded is shown, and the other pages hear about it.
  assert.match(slotHook, /const evicted = evictedSlotsFrom\(res\);/);
  assert.match(slotHook, /resyncEvicted\(evicted, "audio"\);/);
});

test("under Off, the speech load says it unloads the chat model and offers to keep both", () => {
  assert.match(
    slotHook,
    /const replacesChat =\s*!useChatRuntimeStore\.getState\(\)\.keepModelsLoaded &&/,
  );
  assert.match(
    slotHook,
    /"Keep multiple models loaded is Off, so this unloads the chat model\."/,
  );
  assert.match(
    slotHook,
    // Owner only: a managed account cannot save the installation's mode.
    /action:\s*replacesChat && isAccountOwner\(\)\s*\? \{ label: "Keep both loaded", onClick: \(\) => void keepBothLoaded\(\) \}/,
  );
  assert.match(slotHook, /updateMultiModelMode\(\s*"fit",/);
});

test("the chat model's automatic loads keep the speech model loaded", () => {
  const loads =
    chatAdapter.match(
      /const loadResp = await loadModel\(\{[\s\S]*?\}\)/g,
    ) ?? [];
  assert.equal(loads.length, 2);
  for (const load of loads) assert.match(load, /alongside: true\b/);
  // Silent before: say what they unloaded to make room.
  assert.equal(chatAdapter.match(/reportEvicted\(loadResp\);/g)?.length, 2);
  assert.match(
    chatAdapter,
    /toastEvicted\(slots\);\s*resyncEvicted\(slots, "chat"\);/,
  );
});

test("Keep multiple models loaded is read from every status, not only after Settings", () => {
  assert.match(
    chatRuntime,
    /syncKeepModelsLoaded\(statusRes\.multi_model_mode\);/,
  );
  assert.match(audioPage, /syncKeepModelsLoaded\(next\.multi_model_mode\);/);
});

test("the Audio page shows and runs its own speech model, not the chat model", () => {
  assert.match(
    audioPage,
    /const view = await describeAudioSlot\(next, \(model\) =>\s*getInferenceStatus\(undefined, model\),?\s*\);[\s\S]*?setAudioSlotModel\(view\.slotModel, view\.slotLabel\);\s*setStatus\(view\.status\);/,
  );
  // Requests that name no model reach the active one, which is the chat model here.
  assert.match(
    audioApi,
    /return audioSlotModel \? \{ \.\.\.body, model: audioSlotModel \} : body;/,
  );
  assert.match(
    audioApi,
    /withAudioSlotModel\(\{\s*messages: \[\{ role: "user", content: text \}\],/,
  );
  assert.match(
    audioApi,
    /JSON\.stringify\(withAudioSlotModel\(buildAudioRunBody\(request\)\)\)/,
  );
});

test("Settings shows the loaded speech model and, under Off, how to keep both", () => {
  assert.match(
    voiceTab,
    /loadedTtsModel\s*\? t\("settings\.voice\.readAloud\.modelLoaded", \{\s*model: audioModelLabel\(loadedTtsModel\),/,
  );
  assert.match(
    voiceTab,
    /\{multiModelMode === "off" && isOwner \? \(\s*<SettingsRow\s*label=\{t\("settings\.voice\.readAloud\.keepBothLabel"\)\}/,
  );
  assert.match(voiceTab, /await updateMultiModelMode\(\s*"fit",/);
  assert.match(en, /modelLoaded: "\{model\} is loaded",/);
  assert.match(en, /keepBothAction: "Keep both loaded",/);
});

test("read-aloud and memory copy is clear in every language", () => {
  // The engine is a choice, not an action, and the row no longer states the bug as policy.
  assert.match(en, /engineStudio: "Unsloth model",/);
  assert.match(
    en,
    /multiModelDescription: "Loading a chat, speech, image or video model unloads the other ones\.",/,
  );
  assert.match(
    en,
    /multiModelHint: "Covers chat, speech, image and video models\./,
  );
  const dir = new URL("../src/i18n/locales/", import.meta.url);
  for (const file of readdirSync(dir)) {
    const locale = readSrc(`i18n/locales/${file}`);
    assert.doesNotMatch(
      locale,
      /replaces the chat model|load one in Speak/i,
      file,
    );
    const readAloud = locale.slice(locale.indexOf("readAloud: {"));
    const block = readAloud.slice(0, readAloud.indexOf("\n      },"));
    for (const key of [
      "modelLoaded",
      "keepBothLabel",
      "keepBothDescription",
      "keepBothAction",
    ]) {
      assert.match(block, new RegExp(`\\b${key}:`), `${file} ${key}`);
    }
    assert.match(block, /modelLoaded:[^\n]*\{model\}/, file);
    // "Unsloth", never "Studio" alone or "Unsloth Studio": Desktop shows these strings too
    // (tests/studio/test_tauri_branding_contract.py).
    const engine = block.match(/engineStudio: "([^"]*)"/)?.[1] ?? "";
    assert.match(engine, /Unsloth/, file);
    assert.doesNotMatch(engine, /Studio/, file);
  }
});
