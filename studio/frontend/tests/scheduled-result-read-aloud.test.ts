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

function setup() {
  const globals = globalThis as Record<string, unknown>;
  const saved = {
    window: globals.window,
    document: globals.document,
    localStorage: globals.localStorage,
  };
  const { fireWindowEvent } = installLocalStorageFake();
  const load = () =>
    loadWithStubs<typeof Unseen>(
      new URL("../src/features/scheduled-tasks/unseen.ts", import.meta.url),
      {
        zustand,
        "@/features/auth": {
          AUTH_SESSION_CLEARED_EVENT: "out",
          AUTH_SESSION_STORED_EVENT: "in",
          getAuthToken: () => "token",
        },
        "@/lib/account-transition": { browserAccountKey: () => "owner" },
        "./results": { UNSEEN_KEY_PREFIX: "unseen:" },
      },
    ).useUnseenScheduledRuns;
  const store = load();
  return { store, load, fireWindowEvent, restore: () => Object.assign(globals, saved) };
}

function scheduledReply(): {
  -readonly [K in keyof AutoReadMessage]: AutoReadMessage[K];
} {
  return {
    id: "answer",
    role: "assistant",
    status: { type: "complete" },
    content: [{ type: "text", text: "Your scheduled summary." }],
    metadata: { custom: { scheduledTask: { taskId: "task", runId: "run" } } },
  };
}

function player(store: ReturnType<typeof setup>["store"]) {
  const state = {
    active: true,
    threadListItem: { id: "local-thread", remoteId: "result" },
    thread: {
      isLoading: true,
      messages: [] as AutoReadMessage[],
      speech: null as null | { messageId: string },
    },
    voice: { ttsEnabled: true, autoReadAloud: true },
  };
  const spoken: string[] = [];
  const stopped: string[] = [];
  const events = new Map<string, () => void>();
  const refs: { current: unknown }[] = [];
  const effects: { deps: unknown[]; cleanup?: (() => void) | void }[] = [];
  let refIndex = 0;
  let effectIndex = 0;
  let pending: (() => void)[] = [];
  const thread = {
    getState: () => state.thread,
    message: ({ id }: { id: string }) => ({
      speak: () => {
        spoken.push(id);
        state.thread.speech = { messageId: id };
      },
    }),
    stopSpeaking: () => {
      if (state.thread.speech) stopped.push(state.thread.speech.messageId);
      state.thread.speech = null;
    },
  };
  const aui = {
    thread: () => thread,
    threads: () => ({
      __internal_getAssistantRuntime: () => ({
        threads: { getById: () => thread },
      }),
    }),
  };
  const useUnseenScheduledRuns = Object.assign(
    (select: (value: ReturnType<typeof store.getState>) => unknown) =>
      select(store.getState()),
    { getState: store.getState },
  );
  const component = loadWithStubs<{ AutoReadAloud: () => void }>(
    new URL(
      "../src/components/assistant-ui/auto-read-aloud.tsx",
      import.meta.url,
    ),
    {
      "@/features/chat": { decideAutoRead, useChatActive: () => state.active },
      "@/features/settings": {
        useVoiceSettingsStore: { getState: () => state.voice },
      },
      "@/features/scheduled-tasks": { useUnseenScheduledRuns },
      "@assistant-ui/react": {
        useAui: () => aui,
        useAuiState: (select: (value: typeof state) => unknown) =>
          select(state),
        useAuiEvent: (name: string, callback: () => void) =>
          events.set(name, callback),
      },
      react: {
        useRef: (initial: unknown) =>
          (refs[refIndex++] ??= { current: initial }),
        useCallback: (callback: unknown) => callback,
        useEffect: (effect: () => (() => void) | void, deps: unknown[]) => {
          const index = effectIndex++;
          const previous = effects[index];
          if (
            previous &&
            deps.every((value, i) => Object.is(value, previous.deps[i]))
          )
            return;
          pending.push(() => {
            previous?.cleanup?.();
            effects[index] = { deps, cleanup: effect() };
          });
        },
      },
    },
  );
  return {
    state,
    spoken,
    stopped,
    render: () => {
      refIndex = effectIndex = 0;
      pending = [];
      component.AutoReadAloud();
      for (const effect of pending) effect();
    },
    event: (name: string) => events.get(name)?.(),
    unmount: () => {
      for (const effect of effects) effect?.cleanup?.();
    },
  };
}

test("opening an unread scheduled result retains one playback through the route handoff", () => {
  const env = setup();
  try {
    const s = env.store.getState();
    s.markFinished([{ taskId: "task", threadId: "result" }], false);
    s.openThread("result", true); // Recents, before its unread marker is cleared.
    assert.deepEqual(env.load().getState().threads, []);
    s.openThread("result", true); // The router observes the same opening.
    assert.equal(s.takeReadAloud("result"), true);
    s.openThread("result", true);
    assert.equal(s.takeReadAloud("result"), false);
    assert.equal(env.load().getState().pendingReadAloudThreadId, null);
  } finally {
    env.restore();
  }
});

test("background completion, marking read, and opening with automatic TTS off stay silent", () => {
  const env = setup();
  try {
    const s = env.store.getState();
    for (const id of ["marked", "disabled", "unopened"])
      s.markFinished([{ taskId: id, threadId: id }], false);
    assert.equal(env.store.getState().pendingReadAloudThreadId, null);
    s.clearThread("marked");
    s.openThread("marked", true);
    assert.equal(s.takeReadAloud("marked"), false);
    s.openThread("disabled", false);
    s.openThread("disabled", true);
    assert.equal(s.takeReadAloud("disabled"), false);
    assert.deepEqual(env.load().getState().threads, ["unopened"]);
  } finally {
    env.restore();
  }
});

test("another tab cannot replay an already opened result", () => {
  const env = setup();
  try {
    env.store
      .getState()
      .markFinished([{ taskId: "task", threadId: "result" }], false);
    const other = env.load();
    env.store.getState().openThread("result", true);
    other.getState().openThread("result", true);
    assert.equal(other.getState().takeReadAloud("result"), false);
    assert.equal(env.store.getState().takeReadAloud("result"), true);
  } finally {
    env.restore();
  }
});

test("navigation away and account changes discard a playback still waiting on history", () => {
  const env = setup();
  try {
    for (const leave of [
      () => env.store.getState().openThread(null, true),
      () => env.fireWindowEvent("out", {}),
    ]) {
      const s = env.store.getState();
      s.markFinished([{ taskId: "task", threadId: "result" }], false);
      s.openThread("result", true);
      leave();
      assert.equal(s.takeReadAloud("result"), false);
    }
  } finally {
    env.restore();
  }
});

test("the actual auto-read component waits for history and speaks once without changing background playback", () => {
  const env = setup();
  try {
    const s = env.store.getState();
    s.markFinished([{ taskId: "task", threadId: "result" }], false);
    s.openThread("result", true);
    const p = player(env.store);
    p.state.thread.isLoading = false;
    p.render();
    assert.equal(env.store.getState().pendingReadAloudThreadId, "result");
    p.state.thread.isLoading = true;
    p.render();
    assert.deepEqual(p.spoken, []);
    p.state.thread.messages = [scheduledReply()];
    p.render();
    assert.deepEqual(p.spoken, []);
    p.state.thread.isLoading = false;
    p.render();
    p.render();
    assert.deepEqual(p.spoken, ["answer"]);
    p.state.active = false;
    p.render();
    assert.deepEqual(p.stopped, []);
    p.unmount();
    assert.deepEqual(p.stopped, ["answer"]);
    const revisit = player(env.store);
    revisit.state.thread = {
      isLoading: false,
      messages: [scheduledReply()],
      speech: null,
    };
    s.openThread("result", true);
    revisit.render();
    assert.deepEqual(revisit.spoken, []);
  } finally {
    env.restore();
  }
});

test("hidden chats and a different chat runtime cannot consume the opened result", () => {
  const env = setup();
  try {
    const s = env.store.getState();
    s.markFinished([{ taskId: "task", threadId: "result" }], false);
    s.openThread("result", true);
    const p = player(env.store);
    p.state.thread = {
      isLoading: false,
      messages: [scheduledReply()],
      speech: null,
    };
    p.state.active = false;
    p.render();
    p.state.active = true;
    p.state.threadListItem.remoteId = "other";
    p.render();
    assert.deepEqual(p.spoken, []);
    assert.equal(env.store.getState().pendingReadAloudThreadId, "result");
    p.state.threadListItem.remoteId = "result";
    p.render();
    assert.deepEqual(p.spoken, ["answer"]);
  } finally {
    env.restore();
  }
});

test("opening does not speak ordinary, incomplete, empty, disabled, or already speaking replies", () => {
  const cases = [
    "ordinary",
    "incomplete",
    "empty",
    "auto-off",
    "tts-off",
    "speaking",
  ];
  for (const kind of cases) {
    const env = setup();
    try {
      const s = env.store.getState();
      s.markFinished([{ taskId: "task", threadId: "result" }], false);
      s.openThread("result", true);
      const p = player(env.store);
      const message = scheduledReply();
      if (kind === "ordinary") delete message.metadata;
      if (kind === "incomplete") message.status = { type: "incomplete" };
      if (kind === "empty") message.content = [];
      p.state.thread = { isLoading: false, messages: [message], speech: null };
      if (kind === "auto-off") p.state.voice.autoReadAloud = false;
      if (kind === "tts-off") p.state.voice.ttsEnabled = false;
      if (kind === "speaking") p.state.thread.speech = { messageId: "manual" };
      p.render();
      assert.deepEqual(p.spoken, [], kind);
      assert.equal(s.takeReadAloud("result"), false, kind);
    } finally {
      env.restore();
    }
  }
});

test("normal live replies still auto-read at run end", () => {
  const env = setup();
  try {
    const p = player(env.store);
    const message = scheduledReply();
    delete message.metadata;
    p.state.thread = { isLoading: false, messages: [message], speech: null };
    p.render();
    assert.deepEqual(p.spoken, []);
    p.event("thread.runEnd");
    assert.deepEqual(p.spoken, ["answer"]);
    p.state.active = false;
    p.render();
    assert.deepEqual(p.stopped, []);
    p.unmount();
    assert.deepEqual(p.stopped, ["answer"]);
  } finally {
    env.restore();
  }
});
