// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The Images pill on a local or Unsloth Studio-tools model renders on the image or video
// model loaded in Unsloth Studio (generate_image / generate_video); a hosted image model
// keeps the provider's own image_generation.

import assert from "node:assert/strict";
import { register } from "node:module";
import test from "node:test";

import { installLocalStorageFake, readSrc } from "./helpers/kit.ts";

const { store: localStorageFake } = installLocalStorageFake();
localStorageFake.set("unsloth_chat_settings_imported_to_studio_db", "true");
register("./thread-sampling-resolver.mjs", import.meta.url);
await import("./helpers/store-stubs/settings-http.ts");
await import("./helpers/store-stubs/chat-history-storage.ts");

type MediaState = {
  supportsTools: boolean;
  supportsBuiltinImageGeneration: boolean;
  localImageModelReady: boolean;
  localVideoModelReady: boolean;
  imageToolsEnabled: boolean;
};

const store = await import("../src/features/chat/stores/chat-runtime-store.ts");
const { localImageTool, imageToolAvailable, localImagePillCount, localMediaToolsForTurn } =
  store as unknown as {
    localImageTool: (s: MediaState) => boolean;
    imageToolAvailable: (s: MediaState) => boolean;
    localImagePillCount: (s: MediaState) => number;
    localMediaToolsForTurn: (s: MediaState) => string[];
  };
const runtime = store.useChatRuntimeStore as unknown as {
  getState: () => MediaState & {
    setImageToolsEnabled: (on: boolean) => void;
    setLocalMediaModelsReady: (ready: { image: boolean; video: boolean }) => void;
  };
  setState: (patch: Record<string, unknown>) => void;
};

const LOCAL: MediaState = {
  supportsTools: true,
  supportsBuiltinImageGeneration: false,
  localImageModelReady: true,
  localVideoModelReady: false,
  imageToolsEnabled: true,
};
const HOSTED: MediaState = { ...LOCAL, supportsBuiltinImageGeneration: true };

test("a hosted image model keeps its builtin; a local turn sends the loaded media models' tools", () => {
  assert.deepEqual(localMediaToolsForTurn(HOSTED), []);
  assert.deepEqual(localMediaToolsForTurn({ ...HOSTED, localVideoModelReady: true }), []);
  assert.deepEqual(localMediaToolsForTurn(LOCAL), ["generate_image"]);
  assert.deepEqual(localMediaToolsForTurn({ ...LOCAL, localVideoModelReady: true }), [
    "generate_image",
    "generate_video",
  ]);
  assert.deepEqual(
    localMediaToolsForTurn({ ...LOCAL, localImageModelReady: false, localVideoModelReady: true }),
    ["generate_video"],
  );
  // Off, no Unsloth Studio tools, or nothing loaded: nothing is sent.
  assert.deepEqual(localMediaToolsForTurn({ ...LOCAL, imageToolsEnabled: false }), []);
  assert.deepEqual(localMediaToolsForTurn({ ...LOCAL, supportsTools: false }), []);
  assert.deepEqual(localMediaToolsForTurn({ ...LOCAL, localImageModelReady: false }), []);
  for (const state of [LOCAL, HOSTED, { ...LOCAL, localVideoModelReady: true }]) {
    assert.ok(!localMediaToolsForTurn(state).includes("image_generation"));
  }
});

test("the pill shows for a hosted image model or a tools model with a media model loaded", () => {
  assert.equal(imageToolAvailable(HOSTED), true);
  assert.equal(imageToolAvailable({ ...HOSTED, supportsTools: false }), true);
  assert.equal(imageToolAvailable(LOCAL), true);
  assert.equal(imageToolAvailable({ ...LOCAL, imageToolsEnabled: false }), true);
  assert.equal(imageToolAvailable({ ...LOCAL, supportsTools: false }), false);
  assert.equal(imageToolAvailable({ ...LOCAL, localImageModelReady: false }), false);
  assert.equal(localImageTool(HOSTED), false, "hosted wins over the local tool");
});

test("the local pill takes composer room only while on; the hosted pill is counted apart", () => {
  assert.equal(localImagePillCount({ ...LOCAL, imageToolsEnabled: false }), 0);
  assert.equal(localImagePillCount(LOCAL), 1);
  assert.equal(localImagePillCount(HOSTED), 0);
  // Loading an image model with the pill off keeps the shared composer at 4 pills (labels stay).
  const sharedComposer = readSrc("features/chat/shared-composer.tsx");
  assert.match(
    sharedComposer,
    /4 \+\s*\(supportsBuiltinImageGeneration \? 1 : 0\) \+\s*localImagePills \+/,
  );
  const thread = readSrc("components/assistant-ui/thread.tsx");
  assert.match(thread, /\(supportsBuiltinImageGeneration \? 1 : 0\) \+\s*localImagePills \+/);
  assert.ok(thread.includes("useChatRuntimeStore(localImagePillCount)"));
});

test("both composers show the pill through the shared selector", () => {
  const sharedComposer = readSrc("features/chat/shared-composer.tsx");
  assert.ok(
    sharedComposer.includes(
      "const showImagePill = supportsBuiltinImageGeneration || localImageToolHere;",
    ),
  );
  assert.match(sharedComposer, /localImageTool\(\{ \.\.\.s, supportsBuiltinImageGeneration \}\)/);
  assert.ok(!sharedComposer.includes("No local tool runtime fallback"));
  // Gemini's image mode stays hosted-only.
  assert.match(
    sharedComposer,
    /imageModeDisablesCode =\s*isExternalGemini &&\s*imageToolsEnabled &&\s*modelLoaded &&\s*supportsBuiltinImageGeneration;/,
  );
  const thread = readSrc("components/assistant-ui/thread.tsx");
  // ImagesToggle and the + menu.
  assert.equal(thread.split("useChatRuntimeStore(imageToolAvailable)").length - 1, 2);
  assert.ok(thread.includes("if (!imageToolShown) {"));
  assert.ok(thread.includes("{imageToolShown && ("));
});

test("readiness changes never clear the stored Images preference (D3)", () => {
  const state = () => runtime.getState();
  runtime.setState({ supportsTools: true, supportsBuiltinImageGeneration: false });
  state().setLocalMediaModelsReady({ image: true, video: false });
  state().setImageToolsEnabled(true);
  state().setLocalMediaModelsReady({ image: false, video: false });
  assert.equal(state().imageToolsEnabled, true);
  assert.equal(state().localImageModelReady, false);
  state().setLocalMediaModelsReady({ image: true, video: true });
  assert.equal(state().imageToolsEnabled, true);
  assert.deepEqual(localMediaToolsForTurn(state()), ["generate_image", "generate_video"]);

  // The clamps key on capability (hosted builtin or Unsloth Studio tools), never on readiness.
  const storeSource = readSrc("features/chat/stores/chat-runtime-store.ts");
  assert.match(
    storeSource,
    /imageToolsEnabled:\s*state\.supportsBuiltinImageGeneration \|\| state\.supportsTools,/,
  );
  const chatPage = readSrc("features/chat/chat-page.tsx");
  const clamps = chatPage.match(
    /imageToolsEnabled:\s*supportsBuiltinImageGeneration \|\| supportsStudioToolsHere\s*\?/g,
  );
  assert.equal(clamps?.length, 2);
  assert.ok(!/imageToolsEnabled:[^,]*localImageModelReady/.test(storeSource + chatPage));
});

test("chat-page refreshes readiness from both status routes and the lifecycle events", () => {
  const chatPage = readSrc("features/chat/chat-page.tsx");
  for (const needle of [
    "getDiffusionStatus(current.signal)",
    "getVideoStatus(current.signal)",
    "Promise.allSettled(",
    'imageStatus.workflows.includes("txt2img")',
    "subscribeModelLifecycle(",
    'subscribeModelEjected("image", refresh)',
    'subscribeModelEjected("video", refresh)',
  ]) {
    assert.ok(chatPage.includes(needle), needle);
  }
});

test("every adapter tool gate opens for, and every tool list carries, the local media tools", () => {
  const adapter = readSrc("features/chat/api/chat-adapter.ts");
  assert.equal(adapter.split("...localMediaTools,").length - 1, 3, "token count, external, local lists");
  assert.equal(adapter.split("localMediaTools.length > 0 ||").length - 1, 2, "external and local gates");
  assert.ok(adapter.includes("localMediaTools.length === 0"), "token-count early exit");
  assert.match(adapter, /images:\s*imageGenerationEnabledForThisTurn \|\| localMediaTools\.length > 0,/);
  // The hosted name only ever follows the hosted flag.
  assert.match(
    adapter,
    /\.\.\.\(imageGenerationEnabledForThisTurn\s*\?\s*\["image_generation"\]\s*:\s*\[\]\)/,
  );
});
