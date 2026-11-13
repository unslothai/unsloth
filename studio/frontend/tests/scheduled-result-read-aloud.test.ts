// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import * as zustand from "zustand";
import {
  decideAutoRead,
  type AutoReadMessage,
} from "../src/features/chat/utils/auto-read-aloud.ts";
import type * as Unseen from "../src/features/scheduled-tasks/unseen.ts";
import { installLocalStorageFake } from "./helpers/kit.ts";
import { loadWithStubs } from "./helpers/module-stubs.ts";

const { fireWindowEvent } = installLocalStorageFake();

function unseenStore() {
  return loadWithStubs<typeof Unseen>(
    new URL("../src/features/scheduled-tasks/unseen.ts", import.meta.url),
    {
      zustand,
      "@/features/auth": {
        AUTH_SESSION_CLEARED_EVENT: "out",
        AUTH_SESSION_STORED_EVENT: "in",
        getAuthToken: () => "token",
      },
      "@/lib/account-transition": { browserAccountKey: () => "owner" },
      "./results": { UNSEEN_KEY_PREFIX: `unseen-${Math.random()}:` },
    },
  ).useUnseenScheduledRuns;
}

function openedResult() {
  const store = unseenStore();
  store.getState().markFinished([{ taskId: "task", threadId: "result" }], false);
  store.getState().openThread("result", true);
  return store;
}

const scheduledReply = (custom: unknown = { scheduledTask: { taskId: "task", runId: "run" } }) =>
  ({
    id: "answer",
    role: "assistant",
    status: { type: "complete" },
    content: [{ type: "text", text: "Your scheduled summary." }],
    metadata: { custom },
  }) as AutoReadMessage;

/** Renders AutoReadAloud with hand-run effects against a fake thread. */
function player(store: ReturnType<typeof unseenStore>) {
  const state = {
    active: true,
    threadListItem: { id: "local", remoteId: "result" },
    thread: { isLoading: false, messages: [] as AutoReadMessage[], speech: null as unknown },
    voice: { ttsEnabled: true, autoReadAloud: true },
  };
  const spoken: string[] = [];
  const thread = {
    getState: () => state.thread,
    message: ({ id }: { id: string }) => ({ speak: () => spoken.push(id) }),
  };
  const refs: { current: unknown }[] = [];
  const effects: { deps: unknown[]; cleanup?: unknown }[] = [];
  let index = 0;
  const component = loadWithStubs<{ AutoReadAloud: () => void }>(
    new URL("../src/components/assistant-ui/auto-read-aloud.tsx", import.meta.url),
    {
      "@/features/chat": { decideAutoRead, useChatActive: () => state.active },
      "@/features/settings": { useVoiceSettingsStore: { getState: () => state.voice } },
      "@/features/scheduled-tasks": {
        useUnseenScheduledRuns: Object.assign(
          (select: (s: ReturnType<typeof store.getState>) => unknown) => select(store.getState()),
          { getState: store.getState },
        ),
      },
      "@assistant-ui/react": {
        useAui: () => ({ thread: () => thread, threads: () => ({}) }),
        useAuiState: (select: (s: typeof state) => unknown) => select(state),
        useAuiEvent: () => undefined,
      },
      react: {
        useRef: (initial: unknown) => (refs[index++] ??= { current: initial }),
        useCallback: (callback: unknown) => callback,
        useEffect: (effect: () => unknown, deps: unknown[]) => {
          const i = index++;
          const previous = effects[i];
          if (previous && deps.every((value, d) => Object.is(value, previous.deps[d]))) return;
          if (typeof previous?.cleanup === "function") previous.cleanup();
          effects[i] = { deps, cleanup: effect() };
        },
      },
    },
  );
  const render = () => {
    index = 0;
    component.AutoReadAloud();
  };
  return { state, spoken, render };
}

test("an opened unread result survives the Recents-to-router handoff and plays once", () => {
  const store = openedResult(); // Recents, before it clears the unread marker.
  store.getState().openThread("result", true); // The router sees the same opening.
  assert.equal(store.getState().takeReadAloud("result"), true);
  store.getState().openThread("result", true);
  assert.equal(store.getState().takeReadAloud("result"), false);
});

test("read results, automatic read aloud off, leaving, and signing out drop the playback", () => {
  const store = unseenStore();
  store.getState().markFinished([{ taskId: "task", threadId: "result" }], false);
  store.getState().openThread("result", false);
  store.getState().openThread("result", true);
  assert.equal(store.getState().takeReadAloud("result"), false);

  for (const leave of [
    (s: ReturnType<typeof unseenStore>) => s.getState().openThread(null, true),
    () => fireWindowEvent("out", {}),
  ]) {
    const opened = openedResult();
    leave(opened);
    assert.equal(opened.getState().takeReadAloud("result"), false);
  }
});

test("playback waits for history, needs the matching visible chat, and speaks once", () => {
  const p = player(openedResult());
  p.state.thread.isLoading = true;
  p.state.thread.messages = [scheduledReply()];
  p.render();
  p.state.thread.isLoading = false;
  p.state.active = false;
  p.render();
  p.state.active = true;
  p.state.threadListItem.remoteId = "other";
  p.render();
  assert.deepEqual(p.spoken, []);
  p.state.threadListItem.remoteId = "result";
  p.render();
  p.render();
  assert.deepEqual(p.spoken, ["answer"]);
});

test("an ordinary reply or disabled voice settings consume the opening silently", () => {
  for (const kind of ["ordinary", "auto-off", "tts-off"]) {
    const store = openedResult();
    const p = player(store);
    p.state.thread.messages = [scheduledReply(kind === "ordinary" ? {} : undefined)];
    if (kind === "auto-off") p.state.voice.autoReadAloud = false;
    if (kind === "tts-off") p.state.voice.ttsEnabled = false;
    p.render();
    assert.deepEqual(p.spoken, [], kind);
    assert.equal(store.getState().pendingReadAloudThreadId, null, kind);
  }
});
