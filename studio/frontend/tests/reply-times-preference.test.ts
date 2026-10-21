// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc, registerBundlerResolver } from "./helpers/kit.ts";
import { replyFinishedTimestamp } from "../src/lib/message-timestamp.ts";

registerBundlerResolver();
const { useChatPreferencesStore } = await import(
  "../src/features/chat/stores/chat-preferences-store.ts"
);

const STORE = readSrc("features/chat/stores/chat-preferences-store.ts");
const THREAD = readSrc("components/assistant-ui/thread.tsx");
const TIME = readSrc("components/assistant-ui/assistant-message-time.tsx");
const CHAT_TAB = readSrc("features/settings/tabs/chat-tab.tsx");

test("reply times are off for fresh and older saved preferences", () => {
  assert.equal(useChatPreferencesStore.getInitialState().showReplyTimes, false);
  assert.match(STORE, /showReplyTimes: saved\?\.showReplyTimes === true/);
});

test("Settings > Chat toggles the preference", () => {
  assert.match(CHAT_TAB, /t\("settings\.chat\.showReplyTimes"\)/);
  assert.match(CHAT_TAB, /onCheckedChange=\{setShowReplyTimes\}/);
});

test("the time leads the reply footer, outside the autohiding action bar", () => {
  const footer = THREAD.indexOf("aui-assistant-message-footer");
  assert.notEqual(footer, -1);
  const time = THREAD.indexOf("<AssistantMessageTime />", footer);
  const bar = THREAD.indexOf("<AssistantActionBar />", footer);
  assert.ok(time !== -1 && time < bar);
  const gate = TIME.slice(
    TIME.indexOf("export const AssistantMessageTime"),
    TIME.indexOf("const ReplyTime"),
  );
  assert.match(gate, /return showReplyTimes \? <ReplyTime \/> : null;/);
  assert.doesNotMatch(gate, /useAuiState/);
  assert.doesNotMatch(TIME, /setInterval|setTimeout|useEffect|components\/ui\/tooltip/);
  assert.doesNotMatch(TIME, /formatMessageDate|todayAt|yesterdayAt/);
});

test("prefers the finish time, falls back to creation, hides while running", () => {
  const createdAt = new Date("2026-10-13T23:00:00Z");
  const finishedAt = Date.parse("2026-10-13T23:55:01Z");
  const done = {
    createdAt,
    status: { type: "complete" },
    metadata: { custom: { responseDetails: { finishedAt } } },
  };
  assert.equal(replyFinishedTimestamp(done), finishedAt);
  assert.equal(
    replyFinishedTimestamp({ createdAt, status: { type: "complete" } }),
    createdAt.getTime(),
  );
  assert.equal(
    replyFinishedTimestamp({ ...done, status: { type: "running" } }),
    undefined,
  );
  // A stopped reply records no responseDetails, only how long it streamed.
  const stoppedAt = Date.parse("2026-10-14T04:12:00Z");
  assert.equal(
    replyFinishedTimestamp({
      createdAt,
      status: { type: "incomplete" },
      metadata: {
        custom: {
          timing: {
            streamStartTime: createdAt.getTime(),
            totalStreamTime: stoppedAt - createdAt.getTime(),
          },
        },
      },
    }),
    stoppedAt,
  );
  const researchDone = Date.parse("2026-10-14T07:30:00Z");
  const research = (completedAt: number | null) => ({
    createdAt,
    status: { type: "complete" },
    metadata: { custom: { researchRun: { status: "completed", completedAt } } },
  });
  assert.equal(replyFinishedTimestamp(research(researchDone)), researchDone);
  // A run still going after a reload is restored as a complete message.
  assert.equal(replyFinishedTimestamp(research(null)), undefined);
  assert.equal(
    replyFinishedTimestamp({
      createdAt,
      metadata: {
        custom: {
          createdAtEstimated: true,
          responseDetails: { finishedAt: Number.NaN },
        },
      },
    }),
    undefined,
  );
});
