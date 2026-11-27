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
  autoStopDelayMs: number;
  idleStopMs: number;
  insertSubmitKey: string;
  pasteMethod: string;
  hudPosition: string;
  muteWhileRecording: boolean;
  llmPostProcessEnabled: boolean;
  llmPostProcessModel: string;
  llmPostProcessPrompt: string;
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
  assert.equal(store.getState().micWarmSeconds, 0);
  assert.equal(store.getState().llmPostProcessEnabled, false);
  assert.equal(store.getState().llmPostProcessModel, "");
  assert.ok(store.getState().llmPostProcessPrompt.includes("${output}"));
  assert.equal(store.getState().idleStopMs, 6000);
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

test("a current payload keeps its choices and forces micWarmSeconds off", async () => {
  const store = await load({
    version: 3,
    state: {
      enabled: true,
      copyLastTranscript: true,
      micWarmSeconds: -1,
      soundVolume: 0.8,
      idleStopMs: 120_000,
      submitAfterInsert: true,
    },
  });
  assert.equal(
    store.getState().enabled,
    true,
    "a version 3 choice is respected",
  );
  assert.equal(store.getState().copyLastTranscript, true);
  assert.equal(store.getState().micWarmSeconds, 0, "micWarmSeconds is always 0");
  assert.equal(store.getState().soundVolume, 1, "feedback volume is fixed at 100%");
  assert.equal(store.getState().idleStopMs, 120_000);
  assert.equal(
    store.getState().insertSubmitKey,
    "enter",
    "the old Enter switch migrates to the insert-key choice",
  );
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
      autoStopDelayMs: -1,
      idleStopMs: -5,
    },
  });
  assert.equal(store.getState().micWarmSeconds, 0);
  assert.equal(store.getState().soundVolume, 1);
  assert.equal(store.getState().pasteMethod, "ctrl_v");
  assert.equal(store.getState().hudPosition, "bottom");
  assert.equal(store.getState().muteWhileRecording, false);
  assert.equal(store.getState().autoStopDelayMs, 2500);
  assert.equal(store.getState().idleStopMs, 6000);
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

test("default dictation dictionary has 1000 curated base words plus user name words", async () => {
  const voiceSettingsModule = (await import(
    new URL(
      "../src/features/settings/stores/voice-settings-store.ts",
      import.meta.url,
    ).href
  )) as {
    DEFAULT_DICTATION_DICTIONARY_WORDS: readonly string[];
    extractUserNameWords: (name?: string) => string[];
    buildDefaultDictationDictionary: (name?: string) => string[];
    applyDictationDictionary: (
      text: string,
      dictionary?: string[],
      options?: { snippets?: boolean },
    ) => string;
  };

  assert.equal(
    voiceSettingsModule.DEFAULT_DICTATION_DICTIONARY_WORDS.length,
    1000,
    "exactly 1000 quality base words excluding user name",
  );
  assert.ok(
    voiceSettingsModule.DEFAULT_DICTATION_DICTIONARY_WORDS.includes("Unsloth"),
    "includes Unsloth",
  );

  const nameWords = voiceSettingsModule.extractUserNameWords("Fahad Khan");
  assert.deepEqual(nameWords, ["Fahad", "Khan"]);

  const combined = voiceSettingsModule.buildDefaultDictationDictionary("Fahad Khan");
  assert.equal(combined.length, 1000);
  assert.equal(combined[0], "Fahad");
  assert.equal(combined[1], "Khan");
  assert.equal(combined[2], "Unsloth");
});

test("custom spellings match split words and win the final pass", async () => {
  const settingsModule = (await import(
    new URL(
      "../src/features/settings/stores/voice-settings-store.ts",
      import.meta.url,
    ).href
  )) as {
    applyDictationDictionary: (
      text: string,
      dictionary?: string[],
      options?: { snippets?: boolean },
    ) => string;
  };
  const apply = settingsModule.applyDictationDictionary;
  assert.equal(
    apply("i use un sloth daily", ["Unsloth"], { snippets: false }),
    "i use Unsloth daily",
  );
  assert.equal(
    apply("i use UNSLOTH daily", ["Unsloth"], { snippets: false }),
    "i use Unsloth daily",
  );
  assert.equal(
    apply("send it k8s now", ["k8s -> Kubernetes"], { snippets: true }),
    "send it Kubernetes now",
  );
  assert.equal(
    apply("send it k8s now", ["k8s -> Kubernetes"], { snippets: false }),
    "send it k8s now",
    "the final spelling pass leaves snippets alone",
  );
});

test("turning history off keeps past dictations so they can still be managed", async () => {
  const settingsModule = (await import(
    new URL(
      "../src/features/settings/stores/voice-settings-store.ts",
      import.meta.url,
    ).href
  )) as {
    useVoiceSettingsStore: {
      getState: () => {
        recentDictations: { id: string; text: string }[];
        setMaxRecentDictations: (value: number) => void;
        addRecentDictation: (text: string) => void;
      };
    };
  };
  const store = settingsModule.useVoiceSettingsStore.getState();
  store.setMaxRecentDictations(5);
  settingsModule.useVoiceSettingsStore
    .getState()
    .addRecentDictation("a saved dictation");
  const before =
    settingsModule.useVoiceSettingsStore.getState().recentDictations.length;
  settingsModule.useVoiceSettingsStore.getState().setMaxRecentDictations(0);
  assert.equal(
    settingsModule.useVoiceSettingsStore.getState().recentDictations.length,
    before,
    "off stops new entries but does not delete the old ones",
  );
});

test("changing the profile name replaces the old name words in the default list", async () => {
  const settingsModule = (await import(
    new URL(
      "../src/features/settings/stores/voice-settings-store.ts",
      import.meta.url,
    ).href
  )) as {
    useVoiceSettingsStore: {
      getState: () => {
        collections: { isDefault?: boolean; words: string[] }[];
        syncUserNameToDictionary: (
          displayName?: string,
          nickname?: string,
        ) => void;
      };
    };
  };
  const store = settingsModule.useVoiceSettingsStore.getState();
  const defaultWords = () => {
    const collection = settingsModule.useVoiceSettingsStore
      .getState()
      .collections.find((entry) => entry.isDefault);
    assert.ok(collection);
    return collection.words;
  };
  store.syncUserNameToDictionary("Fahad Khan");
  assert.ok(defaultWords().includes("Fahad"));
  store.syncUserNameToDictionary("Ali Khan");
  assert.ok(!defaultWords().includes("Fahad"));
  assert.ok(defaultWords().includes("Ali"));
});
