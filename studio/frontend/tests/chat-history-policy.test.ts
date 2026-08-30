// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  clearPersistedChatDrafts,
  isChatHistoryDisabled,
  stripPersistedDictationHistory,
} from "../src/lib/chat-history-policy.ts";
import {
  composerDraftKey,
  composerPasteDraftKey,
  readComposerDraft,
  readPasteDraft,
  writeComposerDraft,
  writePasteDraft,
} from "../src/features/chat/utils/composer-draft.ts";

const deleteThreadMessage = readFileSync(
  new URL(
    "../src/features/chat/utils/delete-thread-message.ts",
    import.meta.url,
  ),
  "utf8",
);
const dataTab = readFileSync(
  new URL("../src/features/settings/tabs/data-tab.tsx", import.meta.url),
  "utf8",
);
const chatAdapter = readFileSync(
  new URL("../src/features/chat/api/chat-adapter.ts", import.meta.url),
  "utf8",
);
const chatPage = readFileSync(
  new URL("../src/features/chat/chat-page.tsx", import.meta.url),
  "utf8",
);
const voiceSettingsStore = readFileSync(
  new URL(
    "../src/features/settings/stores/voice-settings-store.ts",
    import.meta.url,
  ),
  "utf8",
);
const audioApi = readFileSync(
  new URL("../src/features/audio/api.ts", import.meta.url),
  "utf8",
);
const audioPage = readFileSync(
  new URL("../src/features/audio/audio-page.tsx", import.meta.url),
  "utf8",
);
const speechAdapter = readFileSync(
  new URL(
    "../src/features/chat/adapters/studio-speech-synthesis-adapter.ts",
    import.meta.url,
  ),
  "utf8",
);
const projectsRoute = readFileSync(
  new URL("../src/app/routes/projects.tsx", import.meta.url),
  "utf8",
);
const rootRoute = readFileSync(
  new URL("../src/app/routes/__root.tsx", import.meta.url),
  "utf8",
);
const appSidebar = readFileSync(
  new URL("../src/components/app-sidebar.tsx", import.meta.url),
  "utf8",
);
const assistantThread = readFileSync(
  new URL("../src/components/assistant-ui/thread.tsx", import.meta.url),
  "utf8",
);
const sharedComposer = readFileSync(
  new URL("../src/features/chat/shared-composer.tsx", import.meta.url),
  "utf8",
);
const directRepositorySyncGuard =
  /syncExportedRepositoryToBackend[\s\S]*?if \(isThreadIncognito\(remoteId\)\) return;/;
const enabledClearButton = /disabled=\{!historyDisabled && count === 0\}/;
const clearButtonLabel = /Clear stored chat data/;

test("operator policy ignores and removes browser draft content", () => {
  const values = new Map<string, string>();
  const attributes = new Map([["data-unsloth-no-chat-history", "true"]]);
  const originalDocument = Object.getOwnPropertyDescriptor(
    globalThis,
    "document",
  );
  const originalWindow = Object.getOwnPropertyDescriptor(globalThis, "window");

  Object.defineProperty(globalThis, "document", {
    configurable: true,
    value: {
      documentElement: {
        getAttribute: (name: string) => attributes.get(name) ?? null,
      },
    },
  });
  Object.defineProperty(globalThis, "window", {
    configurable: true,
    value: {
      localStorage: {
        get length() {
          return values.size;
        },
        getItem: (key: string) => values.get(key) ?? null,
        key: (index: number) => [...values.keys()][index] ?? null,
        setItem: (key: string, value: string) => values.set(key, value),
        removeItem: (key: string) => values.delete(key),
      },
    },
  });
  try {
    const draftKey = composerDraftKey("thread-1");
    const pasteKey = composerPasteDraftKey("thread-1");
    values.set(draftKey, "old draft");
    values.set(pasteKey, JSON.stringify(["old paste"]));
    values.set("chat-draft:thread-2", "another old draft");
    values.set("chat-draft-pastes:thread-3", JSON.stringify(["another paste"]));
    values.set("unsloth_chat_auto_continue_leases", "old lease");
    values.set("voice-settings", "keep me");

    assert.equal(isChatHistoryDisabled(), true);
    assert.equal(values.has("chat-draft:thread-2"), false);
    assert.equal(values.has("chat-draft-pastes:thread-3"), false);
    assert.equal(values.has("unsloth_chat_auto_continue_leases"), false);
    assert.equal(values.get("voice-settings"), "keep me");
    assert.equal(readComposerDraft(draftKey), null);
    assert.deepEqual(readPasteDraft(pasteKey), []);

    writeComposerDraft(draftKey, "new draft");
    writePasteDraft(pasteKey, ["new paste"]);
    assert.equal(values.has(draftKey), false);
    assert.equal(values.has(pasteKey), false);

    values.set("chat-draft:late-thread", "late draft");
    clearPersistedChatDrafts();
    assert.equal(values.has("chat-draft:late-thread"), false);
    assert.equal(values.get("voice-settings"), "keep me");
  } finally {
    if (originalDocument) {
      Object.defineProperty(globalThis, "document", originalDocument);
    } else {
      Reflect.deleteProperty(globalThis, "document");
    }
    if (originalWindow) {
      Object.defineProperty(globalThis, "window", originalWindow);
    } else {
      Reflect.deleteProperty(globalThis, "window");
    }
  }
});

test("desktop document-start policy does not depend on a parsed root element", () => {
  const originalDocument = Object.getOwnPropertyDescriptor(
    globalThis,
    "document",
  );
  const originalWindow = Object.getOwnPropertyDescriptor(globalThis, "window");
  Object.defineProperty(globalThis, "document", {
    configurable: true,
    value: { documentElement: null },
  });
  Object.defineProperty(globalThis, "window", {
    configurable: true,
    value: Object.defineProperty({}, "__UNSLOTH_NO_CHAT_HISTORY__", {
      value: true,
    }),
  });
  try {
    assert.equal(isChatHistoryDisabled(), true);
  } finally {
    if (originalDocument) {
      Object.defineProperty(globalThis, "document", originalDocument);
    } else {
      Reflect.deleteProperty(globalThis, "document");
    }
    if (originalWindow) {
      Object.defineProperty(globalThis, "window", originalWindow);
    } else {
      Reflect.deleteProperty(globalThis, "window");
    }
  }
});

test("operator policy removes persisted dictations but keeps voice preferences", () => {
  const originalWindow = Object.getOwnPropertyDescriptor(globalThis, "window");
  Object.defineProperty(globalThis, "window", {
    configurable: true,
    value: Object.defineProperty({}, "__UNSLOTH_NO_CHAT_HISTORY__", {
      value: true,
    }),
  });
  try {
    const scrubbed = JSON.parse(
      stripPersistedDictationHistory(
        JSON.stringify({
          state: {
            dictationLanguage: "ja-JP",
            recentDictations: [{ text: "private prompt", chatId: "thread-1" }],
          },
          version: 1,
        }),
      ),
    ) as {
      state: { dictationLanguage: string; recentDictations: unknown[] };
      version: number;
    };
    assert.equal(scrubbed.state.dictationLanguage, "ja-JP");
    assert.deepEqual(scrubbed.state.recentDictations, []);
    assert.equal(scrubbed.version, 1);
    assert.match(
      voiceSettingsStore,
      /getItem:[\s\S]*?stripPersistedDictationHistory\(value\)/,
    );
    assert.match(
      voiceSettingsStore,
      /addRecentDictation:[\s\S]*?if \(isChatHistoryDisabled\(\)\) return state;/,
    );
  } finally {
    if (originalWindow) {
      Object.defineProperty(globalThis, "window", originalWindow);
    } else {
      Reflect.deleteProperty(globalThis, "window");
    }
  }
});

test("temporary threads bypass direct repository synchronization", () => {
  assert.match(deleteThreadMessage, directRepositorySyncGuard);
});

test("operator mode keeps destructive stored-history cleanup available", () => {
  assert.match(dataTab, enabledClearButton);
  assert.match(dataTab, clearButtonLabel);
});

test("operator mode separates standalone audio persistence from chat speech", () => {
  assert.match(audioApi, /\/api\/inference\/audio\/generate\/gallery/);
  assert.match(speechAdapter, /\/api\/inference\/audio\/generate/);
  assert.doesNotMatch(
    speechAdapter,
    /\/api\/inference\/audio\/generate\/gallery/,
  );
});

test("operator mode keeps hidden audio cleanup reachable and confirmed", () => {
  assert.match(audioPage, /clips\.length > 0 \|\| historyDisabled/);
  assert.match(audioPage, /onClick=\{\(\) => setClearConfirmOpen\(true\)\}/);
  assert.match(
    audioPage,
    /<AlertDialogTitle>Clear stored audio\?<\/AlertDialogTitle>/,
  );
  assert.match(
    audioPage,
    /including\s+clips hidden by the no-chat-history policy/,
  );
  assert.match(audioPage, /void handleClearGallery\(\)/);
  assert.match(audioPage, /disabled=\{clearingGallery\}/);
});

test("operator mode cannot adopt a deep-linked project", () => {
  assert.match(
    chatAdapter,
    /resolveProjectId[\s\S]*?if \(isChatHistoryDisabled\(\)\) return null;/,
  );
  assert.match(
    chatPage,
    /if \(historyDisabled\) \{[\s\S]*?setActiveProjectId\(null\);[\s\S]*?replace: true/,
  );
});

test("operator mode redirects and removes project entry points", () => {
  assert.match(
    projectsRoute,
    /if \(isChatHistoryDisabled\(\)\) \{\s*throw redirect\(\{ to: "\/chat" \}\)/,
  );
  assert.match(
    rootRoute,
    /"switchToProjects"[\s\S]*?enabled: routeShortcutEnabled && !historyDisabled/,
  );
  assert.match(appSidebar, /!historyDisabled \|\| item\.id !== "projects"/);
  assert.match(
    assistantThread,
    /const visiblePlusItems = PLUS_MENU_ORDER\.filter\([\s\S]*?!historyDisabled \|\| id !== "projects"/,
  );
  assert.match(
    sharedComposer,
    /const visiblePlusItems = PLUS_MENU_ORDER\.filter\([\s\S]*?!historyDisabled \|\| id !== "projects"/,
  );
});
