// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A chat waiting on an Allow / Deny prompt is marked in the sidebar, and on Desktop the window asks
// for attention. app-sidebar.tsx is .tsx, which node's type stripping cannot import, so its wiring
// is pinned from source.

import assert from "node:assert/strict";
import { register } from "node:module";
import test, { mock } from "node:test";

import { atDefaultUiScale, readSrc } from "./helpers/kit.ts";

const control: { attention?: (number | null | undefined)[] } = {};
const page = { visibilityState: "visible", focused: true };

Object.assign(globalThis, {
  __TAURI_WINDOW_STUB__: control,
  // lib/api-base reads this to decide it is running inside the desktop app.
  window: { __TAURI_INTERNALS__: {}, location: { protocol: "tauri:" } },
  document: {
    get visibilityState() {
      return page.visibilityState;
    },
    hasFocus: () => page.focused,
  },
});

register("./helpers/tauri-window-resolver.mjs", import.meta.url);

mock.timers.enable({ apis: ["setTimeout", "Date"], now: 0 });

const {
  APPROVAL_SETTLE_MS,
  syncApprovalWaiting,
  useApprovalWaitingStore,
  watchApprovalWaiting,
} = await import("../src/features/chat/utils/approval-waiting.ts");

const INFORMATIONAL = 2;

function waiting(): string[] {
  return [...useApprovalWaitingStore.getState().threadIds].sort();
}

function parked(entries: Record<string, string>) {
  return Object.fromEntries(
    Object.entries(entries).map(([id, threadId]) => [id, { threadId }]),
  );
}

// Loaded once up front, so the module's own dynamic import is a cache hit in every case.
await import("@tauri-apps/api/window");

// The attention request is a dynamic import, so wait on the calls rather than a turn count. A case
// expecting none still waits a while, so a call that should not happen has time to land. Real
// time, since setTimeout and Date are mocked.
async function attention(
  expected: number,
): Promise<(number | null | undefined)[]> {
  const deadline = performance.now() + (expected === 0 ? 250 : 10_000);
  while (performance.now() < deadline) {
    if (expected > 0 && (control.attention?.length ?? 0) >= expected) break;
    await new Promise((resolve) => setImmediate(resolve));
  }
  return control.attention ?? [];
}

function reset(background = false) {
  syncApprovalWaiting({});
  control.attention = [];
  page.visibilityState = background ? "hidden" : "visible";
  page.focused = !background;
}

test("a parked call marks its chat only once it has waited out the settle window", () => {
  reset();
  syncApprovalWaiting(parked({ "s:a:1": "chat-a" }));
  assert.deepEqual(waiting(), []);
  mock.timers.tick(APPROVAL_SETTLE_MS - 1);
  assert.deepEqual(waiting(), []);
  mock.timers.tick(1);
  assert.deepEqual(waiting(), ["chat-a"]);
  syncApprovalWaiting({});
  assert.deepEqual(waiting(), [], "an answered call clears at once");
});

test("a call answered inside the settle window never marks its chat or asks for attention", async () => {
  reset(true);
  // "Always allow": the card answers in one request, well inside the window.
  syncApprovalWaiting(parked({ "s:a:1": "chat-a" }));
  mock.timers.tick(50);
  syncApprovalWaiting({});
  mock.timers.tick(APPROVAL_SETTLE_MS * 2);
  assert.deepEqual(waiting(), []);
  assert.deepEqual(await attention(0), []);
});

test("each chat settles on its own clock, and a later call does not delay an earlier one", () => {
  reset();
  syncApprovalWaiting(parked({ "s:a:1": "chat-a" }));
  mock.timers.tick(1_000);
  syncApprovalWaiting(parked({ "s:a:1": "chat-a", "s:b:1": "chat-b" }));
  mock.timers.tick(APPROVAL_SETTLE_MS - 1_000);
  assert.deepEqual(waiting(), ["chat-a"]);
  mock.timers.tick(1_000);
  assert.deepEqual(waiting(), ["chat-a", "chat-b"]);
  syncApprovalWaiting(parked({ "s:b:1": "chat-b" }));
  assert.deepEqual(waiting(), ["chat-b"]);
  reset();
});

test("the store keeps one Set while membership is unchanged, so rows do not re-render", () => {
  reset();
  syncApprovalWaiting(parked({ "s:a:1": "chat-a" }));
  mock.timers.tick(APPROVAL_SETTLE_MS);
  const first = useApprovalWaitingStore.getState().threadIds;
  syncApprovalWaiting(parked({ "s:a:1": "chat-a", "s:a:2": "chat-a" }));
  mock.timers.tick(APPROVAL_SETTLE_MS);
  assert.equal(useApprovalWaitingStore.getState().threadIds, first);
  reset();
});

test("in the background, Desktop asks for attention once per settled call and withdraws it after", async () => {
  reset(true);
  syncApprovalWaiting(parked({ "s:a:1": "chat-a" }));
  assert.deepEqual(await attention(0), [], "nothing before the settle window");
  mock.timers.tick(APPROVAL_SETTLE_MS);
  assert.deepEqual(await attention(1), [INFORMATIONAL]);
  syncApprovalWaiting({});
  assert.deepEqual(await attention(2), [INFORMATIONAL, null]);
});

test("a focused window is not asked for attention, and has nothing to withdraw", async () => {
  reset(false);
  syncApprovalWaiting(parked({ "s:a:1": "chat-a" }));
  mock.timers.tick(APPROVAL_SETTLE_MS);
  assert.deepEqual(waiting(), ["chat-a"], "the sidebar is still marked");
  syncApprovalWaiting({});
  assert.deepEqual(await attention(0), []);
});

test("a run with no saved chat yet asks for attention without marking a row", async () => {
  reset(true);
  syncApprovalWaiting(parked({ "_default:1": "" }));
  mock.timers.tick(APPROVAL_SETTLE_MS);
  assert.deepEqual(waiting(), []);
  assert.deepEqual(await attention(1), [INFORMATIONAL]);
  reset();
});

test("watching follows the store's changes and stopping clears the marks", () => {
  reset();
  type State = { toolConfirmations: Record<string, { threadId: string }> };
  let state: State = { toolConfirmations: {} };
  const listeners = new Set<(next: State, previous: State) => void>();
  const source = {
    getState: () => state,
    subscribe: (listener: (next: State, previous: State) => void) => {
      listeners.add(listener);
      return () => void listeners.delete(listener);
    },
  };
  const set = (next: State) => {
    const previous = state;
    state = next;
    for (const listener of listeners) listener(state, previous);
  };

  const stop = watchApprovalWaiting(source);
  set({ toolConfirmations: parked({ "s:a:1": "chat-a" }) });
  mock.timers.tick(APPROVAL_SETTLE_MS);
  assert.deepEqual(waiting(), ["chat-a"]);
  stop();
  assert.equal(listeners.size, 0);
  assert.deepEqual(waiting(), []);
});

test("both writers record which chat a parked call belongs to", () => {
  const adapter = readSrc("features/chat/api/chat-adapter.ts");
  assert.match(
    adapter,
    /\.setToolConfirmation\(\s*id,\s*approvalId,\s*sandboxSessionId \?\? "",\s*toolConfirmationScopeId,\s*resolvedThreadId \?\? "",/,
  );
  const provider = readSrc("features/chat/runtime-provider.tsx");
  assert.match(
    provider,
    /\.setToolConfirmation\(\s*partId,\s*approvalId,\s*sessionId,\s*`\$\{sessionId \|\| "_default"\}:\$\{threadId\}`,\s*threadId,\s*\)/,
  );
});

test("the sidebar watches once and swaps the row spinner for the approval dot", () => {
  const sidebar = atDefaultUiScale(readSrc("components/app-sidebar.tsx"));
  assert.match(
    sidebar,
    /useEffect\(\(\) => watchApprovalWaiting\(useChatRuntimeStore\), \[\]\);/,
  );
  assert.match(
    sidebar,
    /const awaitingApproval =\s*isGenerating &&\s*threadIds\.some\(\(threadId\) => approvalWaitingThreadIds\.has\(threadId\)\);/,
  );
  // Inside the spinner's own anchor, so it takes the same column, padding and hover fade.
  const anchor = sidebar.indexOf("{showWorkSpinner ? (");
  const dot = sidebar.indexOf("{awaitingApproval ? (\n", anchor);
  const spinner = sidebar.indexOf('data-testid="chat-row-spinner"', anchor);
  const anchorEnd = sidebar.indexOf(") : null}", anchor);
  assert.ok(anchor > 0 && anchor < dot && dot < spinner && spinner < anchorEnd);
  assert.match(
    sidebar.slice(dot, spinner),
    /<ChatRowApprovalDot\s+label=\{translate\("shell\.navigation\.chatAwaitingApproval"\)\}\s+\/>/,
  );
  const component =
    /function ChatRowApprovalDot\([\s\S]*?\n\}\n/.exec(sidebar)?.[0] ?? "";
  // Announced like the spinner it replaces, not motion-only.
  assert.match(component, /role="status"\s+aria-label=\{label\}/);
  assert.match(
    component,
    /size-2 rounded-full bg-primary motion-safe:animate-pulse/,
  );
});
