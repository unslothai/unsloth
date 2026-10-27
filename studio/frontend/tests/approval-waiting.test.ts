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
  returnToChatThreadId,
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
    /const confirmationThreadKey = liveThreadKey\(serverCancel\);\s*useChatRuntimeStore\s*\.getState\(\)\s*\.setToolConfirmation\(\s*id,\s*approvalId,\s*sandboxSessionId \?\? "",\s*toolConfirmationScopeId,\s*confirmationThreadKey === "__default" \? "" : confirmationThreadKey,/,
  );
  const provider = readSrc("features/chat/runtime-provider.tsx");
  assert.match(
    provider,
    /\.setToolConfirmation\(\s*partId,\s*approvalId,\s*sessionId,\s*`\$\{sessionId \|\| "_default"\}:\$\{threadId\}`,\s*threadId,\s*\)/,
  );
});

test("the sidebar watches once and swaps the row spinner for the approval mark", () => {
  const sidebar = atDefaultUiScale(readSrc("components/app-sidebar.tsx"));
  assert.match(
    sidebar,
    /useEffect\(\(\) => watchApprovalWaiting\(useChatRuntimeStore\), \[\]\);/,
  );
  assert.match(
    sidebar,
    /const awaitingApproval =\s*isGenerating &&\s*threadIds\.some\(\(threadId\) => approvalWaitingElsewhere\.has\(threadId\)\);/,
  );
  // Inside the spinner's own anchor, so it takes the same column, padding and hover fade.
  const anchor = sidebar.indexOf("{showWorkSpinner ? (");
  const mark = sidebar.indexOf("{awaitingApproval ? (\n", anchor);
  const spinner = sidebar.indexOf('data-testid="chat-row-spinner"', anchor);
  const anchorEnd = sidebar.indexOf(") : null}", anchor);
  assert.ok(anchor > 0 && anchor < mark && mark < spinner && spinner < anchorEnd);
  assert.match(
    sidebar.slice(mark, spinner),
    /<ApprovalMark testId="chat-row-approval" className="size-3\.5" \/>/,
  );
});

test("the approval mark is a static amber disc with a glyph, apart from the unread dot", () => {
  const sidebar = readSrc("components/app-sidebar.tsx");
  const component = /function ApprovalMark\([\s\S]*?\n\}\n/.exec(sidebar)?.[0] ?? "";
  assert.ok(component, "ApprovalMark not found");
  assert.match(component, /<svg[^>]*aria-hidden="true"/);
  // Amber in both themes, never the accent the unread dot uses.
  assert.match(component, /<circle cx="7" cy="7" r="7" className="fill-amber-700 dark:fill-amber-400" \/>/);
  assert.doesNotMatch(component, /primary/);
  assert.match(component, /<path\s+d="M7 3\.6v4\.1"[\s\S]*?className="stroke-white dark:stroke-neutral-900"/);
  assert.match(component, /<circle cx="7" cy="10\.3" r="1" className="fill-white dark:fill-neutral-900" \/>/);
  // Static: motion is the spinner's cue for work, and the wait can last an hour.
  assert.doesNotMatch(component, /animate-/);
});

test("a waiting or unread row says so in its button's name, and a waiting one is not busy", () => {
  const sidebar = readSrc("components/app-sidebar.tsx");
  const button = sidebar.indexOf('data-testid="recent-thread"');
  const buttonEnd = sidebar.indexOf("</SidebarMenuButton>", button);
  assert.ok(button > 0 && buttonEnd > button);
  const row = sidebar.slice(button, buttonEnd);
  assert.match(row, /aria-busy=\{\(isGenerating && !awaitingApproval\) \|\| undefined\}/);
  assert.match(
    row,
    /\{awaitingApproval \? \(\s*<span className="sr-only"> \{translate\("shell\.navigation\.chatAwaitingApproval"\)\}<\/span>\s*\) : hasUnreadActivity \? \(\s*<span className="sr-only"> \{translate\("shell\.navigation\.chatUnread"\)\}<\/span>\s*\) : null\}/,
  );
});

test("the New Chat item shows the mark instead of its spinner while a chat waits", () => {
  const sidebar = readSrc("components/app-sidebar.tsx");
  assert.match(
    sidebar,
    /const anyChatWaiting = useMemo\(\s*\(\) => \[\.\.\.approvalWaitingElsewhere\]\.some\(\(id\) => runningByThreadId\[id\]\),/,
  );
  // On every route, since the waiting chat's own row may not be on screen.
  assert.match(
    sidebar,
    /waiting=\{\s*anyChatWaiting\s*\? t\("shell\.navigation\.chatAwaitingApproval"\)\s*: undefined\s*\}/,
  );
  const navItem = /function NavItem\([\s\S]*?\n\}\n/.exec(sidebar)?.[0] ?? "";
  assert.ok(navItem, "NavItem not found");
  assert.match(navItem, /data-spinner=\{spinner && !waiting \? "true" : undefined\}/);
  assert.match(
    navItem,
    /\{waiting \? \(\s*<>\s*<span className="sr-only"> \{waiting\}<\/span>[\s\S]*?<ApprovalMark className="ml-auto mr-1\.5 size-3\.5 group-data-\[collapsible=icon\]:hidden" \/>\s*<\/>\s*\) : \(\s*spinner && \(/,
  );
  assert.match(
    navItem,
    /\{waiting \? \(\s*<ApprovalMark\s+testId="nav-approval"\s+className="pointer-events-none absolute right-1 top-1 hidden size-2\.5 group-data-\[collapsible=icon\]:block"\s+\/>\s*\) : \(\s*spinner && \(\s*<Spinner className="pointer-events-none absolute right-1 top-1/,
  );
});

test("the open chat is left out of the waiting marks: its card is on screen", () => {
  const sidebar = readSrc("components/app-sidebar.tsx");
  const owner =
    /const approvalWaitingElsewhere = useMemo\(\(\) => \{[\s\S]*?\n  \}, \[[^\]]*\]\);/.exec(sidebar)?.[0] ?? "";
  assert.ok(owner, "approvalWaitingElsewhere not found");
  // Only the chat route has an open chat; elsewhere every waiting chat is marked.
  assert.match(owner, /if \(!activeThreadId\) return approvalWaitingThreadIds;/);
  // A compare chat is open as its pair id while its panes wait under their own ids.
  assert.match(owner, /const open = allChatItems\.find\(\(item\) => item\.id === activeThreadId\);/);
  assert.match(owner, /new Set\(open \? getSidebarItemThreadIds\(open\) : \[activeThreadId\]\)/);
  assert.match(owner, /\.filter\(\(id\) => !openIds\.has\(id\)\)/);
  assert.match(owner, /\[approvalWaitingThreadIds, activeThreadId, allChatItems\]/);
  // Defined after the open chat is known, and the only reader of the raw store besides it.
  assert.ok(sidebar.indexOf("const activeThreadId = isChatRoute") < sidebar.indexOf("const approvalWaitingElsewhere"));
  const readers = sidebar.match(/approvalWaitingThreadIds/g) ?? [];
  assert.equal(readers.length, 4, "a reader skips the open-chat exclusion");
});

test("Return to Chat opens the newest waiting chat while one waits, else the newest running one", () => {
  const running = { "chat-a": true, "chat-b": true, "chat-c": false };
  assert.equal(returnToChatThreadId(running, new Set()), "chat-b");
  // The older chat waits while the newer one generates: the row's mark is chat-a's.
  assert.equal(returnToChatThreadId(running, new Set(["chat-a"])), "chat-a");
  assert.equal(returnToChatThreadId(running, new Set(["chat-a", "chat-b"])), "chat-b");
  // A stale wait on a chat that stopped running does not count.
  assert.equal(returnToChatThreadId(running, new Set(["chat-c"])), "chat-b");
  assert.equal(returnToChatThreadId({}, new Set(["chat-a"])), null);
});

test("Return to Chat targets a waiting chat whenever the row shows the mark", () => {
  const sidebar = readSrc("components/app-sidebar.tsx");
  const target =
    /const runningTarget = useMemo\(\(\) => \{[\s\S]*?\n  \}, \[[^\]]*\]\);/.exec(sidebar)?.[0] ?? "";
  assert.ok(target, "runningTarget not found");
  assert.match(target, /returnToChatThreadId\(runningByThreadId, approvalWaitingElsewhere\)/);
  assert.match(target, /\[runningByThreadId, approvalWaitingElsewhere, allChatItems\]/);
});

test("a first turn parked before its chat had an id marks the row once the run is adopted", () => {
  reset();
  syncApprovalWaiting(parked({ "s:_default:1": "" }));
  mock.timers.tick(APPROVAL_SETTLE_MS);
  assert.deepEqual(waiting(), []);
  syncApprovalWaiting(parked({ "s:_default:1": "chat-new" }));
  assert.deepEqual(waiting(), ["chat-new"]);
  reset();
});

test("adopting a first turn hands its unkeyed parked calls the chat id, and only those", async () => {
  const { adoptFirstTurnConfirmations } = await import(
    "../src/features/chat/utils/adopt-first-turn-confirmations.ts"
  );
  const before = {
    a: { threadId: "", approvalId: "x" },
    b: { threadId: "chat-b", approvalId: "y" },
  };
  assert.deepEqual(adoptFirstTurnConfirmations(before, "chat-new"), {
    a: { threadId: "chat-new", approvalId: "x" },
    b: { threadId: "chat-b", approvalId: "y" },
  });
  assert.equal(before.a.threadId, "");
  assert.equal(adoptFirstTurnConfirmations({ b: before.b }, "chat-new"), null);

  const store = readSrc("features/chat/stores/chat-runtime-store.ts");
  const adopt = store.slice(
    store.indexOf("adoptDefaultThreadRun: (threadId) =>"),
    store.indexOf("runKeyForOwner: (fallbackKey, owner) =>"),
  );
  assert.match(
    adopt,
    /if \(moved\.runningByThreadId\) \{[\s\S]*adoptFirstTurnConfirmations\([\s\S]*state\.toolConfirmations,[\s\S]*moved\.toolConfirmations = confirmations/,
  );
});
