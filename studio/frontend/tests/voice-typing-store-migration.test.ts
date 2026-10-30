// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The voice-typing preferences are persisted, so every rework of their shape has to
// bring older installs along: the master switch becomes opt-in, the old clipboard
// flag becomes the copy-once preference, and removed keys disappear.

import assert from "node:assert/strict";
import { register } from "node:module";
import test from "node:test";

import { installLocalStorageFake } from "./helpers/kit.ts";

// Resolves the store's extensionless relative imports and re-evaluates them per
// "?bust=N", so each case starts from its own persisted payload.
register("./helpers/tauri-core-resolver.mjs", import.meta.url);

const { store: localStorage } = installLocalStorageFake();

const STORAGE_KEY = "unsloth_voice_typing";

// A file:// URL keeps the query string; a relative specifier would be normalised by
// the resolver and the module would come back from the cache.
const STORE_URL = new URL(
  "../src/features/voice-typing/voice-typing-store.ts",
  import.meta.url,
).href;

interface VoiceTypingState {
  enabled: boolean;
  shortcut: string;
  copyLastTranscript: boolean;
  micWarmSeconds: number;
  soundVolume: number;
  pasteMethod: string;
  hudPosition: string;
  muteWhileRecording: boolean;
  lastTranscript: string;
  hudEnabled?: unknown;
  hudShowTranscript?: unknown;
  keepOnClipboard?: unknown;
  setEnabled: (enabled: boolean) => void;
  setLastTranscript: (text: string) => void;
}

let generation = 0;

async function load(saved?: {
  state: Record<string, unknown>;
  version: number;
}) {
  if (saved) {
    localStorage.set(STORAGE_KEY, JSON.stringify(saved));
  } else {
    localStorage.delete(STORAGE_KEY);
  }
  generation += 1;
  const module = (await import(`${STORE_URL}?bust=${generation}`)) as {
    useVoiceTypingStore: {
      getState: () => VoiceTypingState;
      getInitialState: () => VoiceTypingState;
    };
  };
  return module.useVoiceTypingStore;
}

test("voice typing is off by default", async () => {
  const store = await load();
  assert.equal(store.getInitialState().enabled, false);
  assert.equal(store.getState().enabled, false);
  assert.equal(store.getState().copyLastTranscript, false);
  assert.equal(store.getState().micWarmSeconds, 15);
});

test("an older enabled preference is reset to off", async () => {
  const store = await load({
    version: 2,
    state: {
      enabled: true,
      shortcut: "Ctrl+Shift+Space",
      mode: "hybrid",
      hudEnabled: true,
      hudShowTranscript: true,
    },
  });
  assert.equal(store.getState().enabled, false, "a global hook stays opt-in");
  assert.equal(store.getState().hudEnabled, undefined);
  assert.equal(store.getState().hudShowTranscript, undefined);
  assert.equal(store.getState().shortcut, "Ctrl+Shift+Space");
});

test("the old clipboard flag becomes the copy-once preference", async () => {
  const kept = await load({
    version: 2,
    state: { enabled: true, keepOnClipboard: true },
  });
  assert.equal(kept.getState().copyLastTranscript, true);
  assert.equal(kept.getState().keepOnClipboard, undefined);

  const restored = await load({
    version: 2,
    state: { enabled: false, keepOnClipboard: false },
  });
  assert.equal(restored.getState().copyLastTranscript, false);
});

test("a current payload keeps its choices and its -1 warm setting", async () => {
  const store = await load({
    version: 3,
    state: {
      enabled: true,
      copyLastTranscript: true,
      micWarmSeconds: -1,
      soundVolume: 0.8,
    },
  });
  assert.equal(
    store.getState().enabled,
    true,
    "a version 3 choice is respected",
  );
  assert.equal(store.getState().copyLastTranscript, true);
  assert.equal(store.getState().micWarmSeconds, -1, "Always stays Always");
  assert.equal(store.getState().soundVolume, 0.8);
});

test("out-of-range values are repaired instead of shown", async () => {
  const store = await load({
    version: 3,
    state: {
      enabled: false,
      micWarmSeconds: -7,
      soundVolume: 4,
      pasteMethod: "bogus",
      hudPosition: "middle",
      muteWhileRecording: "yes",
    },
  });
  assert.equal(store.getState().micWarmSeconds, 15);
  assert.equal(store.getState().soundVolume, 1);
  assert.equal(store.getState().pasteMethod, "ctrl_v");
  assert.equal(store.getState().hudPosition, "bottom");
  assert.equal(store.getState().muteWhileRecording, false);
});

test("new writes carry the current schema version", async () => {
  const store = await load();
  store.getState().setEnabled(true);
  store.getState().setLastTranscript("Transient session text");
  const saved = JSON.parse(localStorage.get(STORAGE_KEY) ?? "{}") as {
    version?: number;
    state?: Record<string, unknown>;
  };
  assert.equal(saved.version, 3);
  assert.equal(saved.state?.enabled, true);
  assert.ok(!("hudEnabled" in (saved.state ?? {})));
  assert.ok(!("keepOnClipboard" in (saved.state ?? {})));
  assert.ok(!("lastTranscript" in (saved.state ?? {})));
});
