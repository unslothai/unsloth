// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import ts from "typescript";
import { getThreadState, LocalThreadRuntimeCore } from "@assistant-ui/core/internal";
import {
  liveThreadRunning,
  registerLiveThreadView,
  subscribeLiveThreadViews,
} from "../src/features/chat/utils/live-thread-head.ts";
import { deriveContextUsageBar } from "../src/features/chat/lib/context-usage-bar-state.ts";
import {
  hasPreStreamRunReservation,
  releasePreStreamRunReservation,
  reservePreStreamRun,
  subscribePreStreamRunReservations,
} from "../src/features/chat/utils/pre-stream-run-reservation.ts";
import {
  compactionDividerCount,
  compactionDividerRows,
  compactionToolCallDividers,
} from "../src/components/assistant-ui/message-derived.ts";
import { ggufCompactionRequestFields } from "../src/features/chat/utils/auto-compaction.ts";
import {
  appendCompactionMarker,
  type CompactionMarker,
  compactionAnchorToolCallId,
  type ContextTruncation,
} from "../src/features/chat/utils/context-truncation.ts";
import {
  isCompactCommand,
  requestThreadCompaction,
  takeThreadCompactionRequest,
} from "../src/features/chat/stores/compact-request-store.ts";

import { readSrc } from "./helpers/kit.ts";

const fit = (
  boundary: number,
  extra: Partial<ContextTruncation> = {},
): ContextTruncation => ({
  dropped_messages: boundary,
  boundary_messages: boundary,
  fits: true,
  ...extra,
});

test("/compact is the command typed alone, in any case", () => {
  assert.equal(isCompactCommand("/compact"), true);
  assert.equal(isCompactCommand("  /COMPACT \n"), true);
  assert.equal(isCompactCommand("/compact the chat please"), false);
  assert.equal(isCompactCommand("compact"), false);
});

test("a compaction request applies to the thread's next request only", () => {
  requestThreadCompaction("t1");
  requestThreadCompaction("t1");
  assert.equal(takeThreadCompactionRequest("t2"), false);
  assert.equal(takeThreadCompactionRequest(undefined), false);
  assert.equal(takeThreadCompactionRequest("t1"), true);
  assert.equal(takeThreadCompactionRequest("t1"), false);
});

test("compact now asks a GGUF or MLX model to compact, even with auto-compaction off", () => {
  const asked = { context_overflow: "truncate_oldest", compact_context: true };
  for (const autoCompactEnabled of [true, false]) {
    assert.deepEqual(
      ggufCompactionRequestFields({ isGguf: true, autoCompactEnabled, compactNow: true }),
      asked,
    );
    assert.deepEqual(
      ggufCompactionRequestFields({
        isGguf: false,
        isMlx: true,
        autoCompactEnabled,
        compactNow: true,
      }),
      asked,
    );
  }
  assert.deepEqual(
    ggufCompactionRequestFields({ isGguf: false, autoCompactEnabled: true, compactNow: true }),
    {},
  );
});

test("a tool loop's replayed cut is not a new marker, a new reset is", () => {
  let markers: CompactionMarker[] | undefined;
  markers = appendCompactionMarker(markers, fit(0, { dropped_messages: 0 }), null);
  assert.equal(markers, undefined);
  markers = appendCompactionMarker(markers, fit(4, { checkpoint_started: true }), null);
  markers = appendCompactionMarker(markers, fit(4), "call-1");
  markers = appendCompactionMarker(markers, fit(4, { checkpoint_started: true }), "call-2");
  assert.deepEqual(
    markers?.map((marker) => marker.afterToolCallId),
    [null, "call-2"],
  );
});

const toolCall = (toolCallId: string) => ({ type: "tool-call", toolCallId });

test("a compaction inside a tool loop is drawn after the call it followed", () => {
  const midLoop = fit(0, { dropped_messages: 6, checkpoint_started: true });
  const messages = [
    { id: "u1", role: "user" },
    {
      id: "a1",
      role: "assistant",
      parts: [toolCall("call-1"), toolCall("call-2"), toolCall("call-3")],
      metadata: {
        custom: {
          contextTruncation: midLoop,
          compactionMarkers: [{ afterToolCallId: "call-2", truncation: midLoop }],
        },
      },
    },
  ];

  assert.equal(compactionDividerRows(messages).size, 0);
  assert.deepEqual(compactionToolCallDividers(messages).get("a1"), {
    afterToolCallId: "call-2",
    truncation: midLoop,
  });
  assert.equal(compactionDividerCount(messages), 1);
});

test("a request that compacted before replying keeps its divider above the prompt", () => {
  const atStart = fit(4, { checkpoint_started: true, manual: true });
  const inLoop = fit(4, { dropped_messages: 3, checkpoint_started: true });
  const messages = [
    { id: "u1", role: "user" },
    { id: "a1", role: "assistant" },
    { id: "u2", role: "user" },
    {
      id: "a2",
      role: "assistant",
      parts: [toolCall("call-9")],
      metadata: {
        custom: {
          contextTruncation: inLoop,
          compactionMarkers: [
            { afterToolCallId: null, truncation: atStart },
            { afterToolCallId: "call-9", truncation: inLoop },
          ],
        },
      },
    },
  ];

  assert.deepEqual([...compactionDividerRows(messages)], [["u2", atStart]]);
  assert.deepEqual(
    [...compactionToolCallDividers(messages)].map(([id, m]) => [id, m.afterToolCallId, m.truncation]),
    [["a2", "call-9", inLoop]],
  );
  assert.equal(compactionDividerCount(messages), 2);
});

test("a marker naming a call the reply no longer shows falls back to the prompt", () => {
  const midLoop = fit(0, { dropped_messages: 2, checkpoint_started: true });
  const messages = [
    { id: "u1", role: "user" },
    {
      id: "a1",
      role: "assistant",
      parts: [],
      metadata: {
        custom: {
          contextTruncation: midLoop,
          compactionMarkers: [{ afterToolCallId: "gone", truncation: midLoop }],
        },
      },
    },
  ];

  assert.deepEqual([...compactionDividerRows(messages)], [["u1", midLoop]]);
  assert.equal(compactionToolCallDividers(messages).size, 0);
});

test("the composer reads /compact before any send path, and only for GGUF or MLX", async () => {
  const thread = await readSrc("components/assistant-ui/thread.tsx");
  const submit = thread.slice(thread.indexOf("const handleSubmit = useCallback("));
  const command = submit.indexOf("isCompactCommand(composerText)");
  assert.ok(command > 0);
  assert.ok(command < submit.indexOf("if (disabled || shouldBlockSend())"));
  assert.match(submit, /requestThreadCompaction\(referenceThreadId\)/);
  assert.match(submit, /resumesThought\(\{/);
  assert.match(submit, /!runtime\.supportsTools \|\|/);
  assert.match(submit, /runtime\.incognito \|\| isThreadIncognito\(referenceThreadId\)/);
  const adapter = await readSrc("features/chat/api/chat-adapter.ts");
  assert.match(adapter, /takeThreadCompactionRequest\(resolvedThreadId\) &&/);
  assert.match(adapter, /compactNow,\n\s+\}\),/);
  // Only a reset started by this request counts; a replayed manual epoch is not one.
  assert.match(
    adapter,
    /marker\.truncation\.manual === true &&\s+marker\.truncation\.checkpoint_started === true/,
  );
  assert.match(adapter, /compactNow &&\n\s+!compactionMarkers\?\.some\(/);
  assert.match(adapter, /"This chat was not compacted"/);
});

test("a fit that recalled anchors before the recall it already emitted", () => {
  const recalled = fit(4, { checkpoint_started: true, recalled_chunks: 3 });
  const search = { toolCallId: "recall", toolName: "search_conversation" };
  const terminal = { toolCallId: "call-1", toolName: "terminal" };
  assert.equal(compactionAnchorToolCallId([search], recalled), null);
  assert.equal(compactionAnchorToolCallId([terminal, search], recalled), "call-1");
  assert.equal(compactionAnchorToolCallId([terminal, search], fit(4)), "recall");
  assert.equal(compactionAnchorToolCallId([], fit(4)), null);
});

test("a reply that reset on many tool rounds draws only its last reset", () => {
  const reset = (dropped: number) =>
    fit(0, { dropped_messages: dropped, checkpoint_started: true });
  const first = reset(4);
  const last = reset(2);
  const messages = [
    { id: "u1", role: "user" },
    {
      id: "a1",
      role: "assistant",
      parts: [toolCall("c1"), toolCall("c2"), toolCall("c3")],
      metadata: {
        custom: {
          contextTruncation: last,
          compactionMarkers: [
            { afterToolCallId: "c1", truncation: first },
            { afterToolCallId: "c2", truncation: reset(3) },
            { afterToolCallId: "c3", truncation: last },
          ],
        },
      },
    },
  ];

  assert.deepEqual(
    [...compactionToolCallDividers(messages)].map(([id, m]) => [id, m.afterToolCallId, m.truncation]),
    [["a1", "c3", last]],
  );
  assert.equal(compactionDividerCount(messages), 1);
});

test("two replies reusing a tool call id keep their own dividers", () => {
  const reset = (dropped: number) =>
    fit(0, { dropped_messages: dropped, checkpoint_started: true });
  const reply = (id: string, truncation: ContextTruncation) => ({
    id,
    role: "assistant",
    parts: [toolCall("tool_call_0")],
    metadata: {
      custom: {
        contextTruncation: truncation,
        compactionMarkers: [{ afterToolCallId: "tool_call_0", truncation }],
      },
    },
  });
  const first = reset(4);
  const second = reset(5);
  const messages = [
    { id: "u1", role: "user" },
    reply("a1", first),
    { id: "u2", role: "user" },
    reply("a2", second),
  ];

  const dividers = compactionToolCallDividers(messages);
  assert.equal(dividers.get("a1")?.truncation, first);
  assert.equal(dividers.get("a2")?.truncation, second);
  assert.equal(compactionDividerCount(messages), 2);
});

function menuCallback(name: string, deps: Record<string, unknown>, argument: number | null = 0) {
  const source = ts.createSourceFile(
    "menu.tsx",
    readSrc("features/chat/components/chat-header-menu.tsx"),
    ts.ScriptTarget.Latest,
    true,
    ts.ScriptKind.TSX,
  );
  let expression: ts.Expression | undefined;
  function visit(node: ts.Node) {
    if (ts.isVariableDeclaration(node) && node.name.getText(source) === name) {
      const value = node.initializer;
      expression = value && ts.isCallExpression(value) && argument !== null ? value.arguments[argument] : value;
    }
    ts.forEachChild(node, visit);
  }
  visit(source);
  assert.ok(expression, `menu callback ${name} exists`);
  const code = ts.transpileModule(`return (${expression.getText(source)});`, {
    compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.None },
  }).outputText;
  return new Function(...Object.keys(deps), code)(...Object.values(deps));
}

function compactMenu(threadId: string) {
  const runtime = {
    modelLoading: false,
    runningByThreadId: {} as Record<string, boolean>,
    loadedIsGguf: true,
    loadedIsMlx: false,
    supportsTools: true,
    activeGgufVariant: "Q4_K_M",
    activeNativePathToken: null,
    params: { checkpoint: "local-model" },
  };
  const menu = { item: { id: threadId, type: "single" } };
  const notices: string[] = [];
  const branch = { ids: ["user", "reply"] as string[] | null };
  let supported = true;
  const click = menuCallback("compact", {
    menu,
    useChatRuntimeStore: { getState: () => runtime },
    liveThreadBranch: (id: string) => id === threadId ? branch.ids : null,
    resumesThought: (target: unknown) => {
      assert.deepEqual(target, {
        loadedIsGguf: runtime.loadedIsGguf,
        loadedIsMlx: runtime.loadedIsMlx,
        activeGgufVariant: runtime.activeGgufVariant,
        activeNativePathToken: runtime.activeNativePathToken,
        checkpoint: runtime.params.checkpoint,
      });
      return supported;
    },
    requestThreadCompaction,
    hasPreStreamRunReservation,
    liveThreadRunning,
    toast: { info: (text: string) => notices.push(text) },
    t: (key: string) => key,
  }) as () => void;
  return { click, runtime, menu, branch, notices, unsupported: () => { supported = false; } };
}

test("the compact button queues the same one-shot request for the selected chat", () => {
  for (const id of ["menu-chat-1", "menu-chat-2"]) {
    const menu = compactMenu(id);
    menu.click();
    menu.click();
    assert.deepEqual(menu.notices, ["chatMenu.compactQueued", "chatMenu.compactQueued"]);
    assert.equal(takeThreadCompactionRequest("another-chat"), false);
    assert.equal(takeThreadCompactionRequest(id), true);
    assert.equal(takeThreadCompactionRequest(id), false);
  }
});

test("the compact button does not queue empty, hidden, unsupported or busy chats", () => {
  for (const blocked of ["empty", "hidden", "unsupported", "running", "loading", "compare"]) {
    const menu = compactMenu(`menu-${blocked}`);
    if (blocked === "empty") menu.branch.ids = [];
    if (blocked === "hidden") menu.branch.ids = null;
    if (blocked === "unsupported") menu.unsupported();
    if (blocked === "running") menu.runtime.runningByThreadId[menu.menu.item.id] = true;
    if (blocked === "loading") menu.runtime.modelLoading = true;
    if (blocked === "compare") menu.menu.item.type = "compare";
    menu.click();
    assert.equal(takeThreadCompactionRequest(menu.menu.item.id), false, blocked);
    assert.deepEqual(menu.notices, blocked === "empty" || blocked === "hidden"
      ? ["chatMenu.compactEmpty"]
      : blocked === "unsupported" ? ["chatMenu.compactUnsupported"] : [], blocked);
  }
});

test("the compact button requires tool capability for both local backends", () => {
  for (const mlx of [false, true]) {
    const menu = compactMenu(`tools-${mlx}`);
    menu.runtime.loadedIsGguf = !mlx;
    menu.runtime.loadedIsMlx = mlx;
    menu.runtime.supportsTools = false;
    menu.click();
    assert.equal(takeThreadCompactionRequest(menu.menu.item.id), false);
    assert.deepEqual(menu.notices, ["chatMenu.compactUnsupported"]);
    menu.runtime.supportsTools = true;
    menu.click();
    assert.equal(takeThreadCompactionRequest(menu.menu.item.id), true);
    assert.deepEqual(menu.notices, ["chatMenu.compactUnsupported", "chatMenu.compactQueued"]);
  }
});

test("the compact button waits for its accepted send's preflight to release", () => {
  for (const [id, siblingId] of [["preflight-a", "preflight-b"], ["preflight-b", "preflight-a"]]) {
    const menu = compactMenu(id);
    const isBusy = () => {
      const preStreamRunActive = menuCallback("preStreamRunActive", {
        menu: menu.menu,
        hasPreStreamRunReservation,
        subscribePreStreamRunReservations,
        useSyncExternalStore: (
          subscribe: typeof subscribePreStreamRunReservations,
          snapshot: () => boolean,
          serverSnapshot: () => boolean,
        ) => {
          assert.equal(subscribe, subscribePreStreamRunReservations);
          assert.equal(serverSnapshot(), false);
          return snapshot();
        },
      }, null);
      return menuCallback("busy", {
        menu: menu.menu,
        preStreamRunActive,
        liveRunActive: false,
        useChatRuntimeStore: (selector: (state: typeof menu.runtime) => boolean) => selector(menu.runtime),
      }, null) as boolean;
    };
    assert.equal(isBusy(), false);
    const token = reservePreStreamRun([id]);
    const sibling = reservePreStreamRun([siblingId]);
    assert.ok(token);
    assert.ok(sibling);
    try {
      assert.equal(isBusy(), true);
      menu.click();
      assert.equal(takeThreadCompactionRequest(id), false);
      assert.deepEqual(menu.notices, []);
      releasePreStreamRunReservation(token);
      assert.equal(isBusy(), false);
      menu.click();
      assert.equal(takeThreadCompactionRequest(id), true);
      assert.equal(takeThreadCompactionRequest(siblingId), false);
      assert.deepEqual(menu.notices, ["chatMenu.compactQueued"]);
    } finally {
      releasePreStreamRunReservation(token);
      releasePreStreamRunReservation(sibling);
      takeThreadCompactionRequest(id);
      takeThreadCompactionRequest(siblingId);
    }
  }
});

test("the compact button waits for a directly issued assistant run", async () => {
  for (const [id, siblingId] of [["direct-a", "direct-b"], ["direct-b", "direct-a"]]) {
    let finish!: () => void;
    const gate = new Promise<void>((resolve) => { finish = resolve; });
    const core = new LocalThreadRuntimeCore({ getModelContext: () => ({}) }, {
      adapters: { chatModel: { run: async () => {
        await gate;
        return { content: [{ type: "text", text: "continued" }], status: { type: "complete", reason: "stop" } };
      } } },
    });
    const item = { id: `local-${id}`, remoteId: id, externalId: undefined, title: undefined, status: "regular" as const, isMain: true };
    const main = { id: item.id };
    const view = {
      subscribe: (listener: () => void) => core.subscribe(listener),
      threads: () => ({ getState: () => ({ mainThreadId: main.id }) }),
      threadListItem: () => ({ getState: () => item }),
      thread: () => ({ getState: () => getThreadState(core, item) }),
    };
    let notifications = 0;
    const unsubscribe = subscribeLiveThreadViews(() => { notifications++; });
    const unregister = registerLiveThreadView(view);
    assert.equal(notifications, 1);
    const menu = compactMenu(id);
    const started = core.startRun({ parentId: null, sourceId: null, runConfig: {} });
    try {
      assert.equal(liveThreadRunning(id), true);
      assert.equal(liveThreadRunning(siblingId), false);
      main.id = "other-selected-chat";
      assert.equal(liveThreadRunning(id), false);
      main.id = item.id;
      assert.equal(hasPreStreamRunReservation([id]), false);
      assert.deepEqual(menu.runtime.runningByThreadId, {});
      menu.click();
      assert.equal(takeThreadCompactionRequest(id), false);
      assert.deepEqual(menu.notices, []);
      const sibling = compactMenu(siblingId);
      sibling.click();
      assert.equal(takeThreadCompactionRequest(siblingId), true);
      const isBusy = () => {
        const liveRunActive = menuCallback("liveRunActive", {
        menu: menu.menu, liveThreadRunning, subscribeLiveThreadViews,
        useSyncExternalStore: (subscribe: typeof subscribeLiveThreadViews, snapshot: () => boolean, serverSnapshot: () => boolean) => {
          assert.equal(subscribe, subscribeLiveThreadViews);
          assert.equal(serverSnapshot(), false);
          return snapshot();
        },
      }, null);
        return menuCallback("busy", {
        menu: menu.menu, preStreamRunActive: false, liveRunActive,
        useChatRuntimeStore: (selector: (state: typeof menu.runtime) => boolean) => selector(menu.runtime),
        }, null);
      };
      assert.equal(isBusy(), true);
      const beforeFinish = notifications;
      finish();
      await started;
      assert.ok(notifications > beforeFinish);
      assert.equal(liveThreadRunning(id), false);
      assert.equal(isBusy(), false);
      menu.click();
      assert.equal(takeThreadCompactionRequest(id), true);
      item.remoteId = siblingId;
      const otherStarted = core.startRun({ parentId: null, sourceId: null, runConfig: {} });
      assert.equal(liveThreadRunning(id), false);
      assert.equal(liveThreadRunning(siblingId), true);
      const beforeUnmount = notifications;
      unregister();
      assert.ok(notifications > beforeUnmount);
      assert.equal(liveThreadRunning(siblingId), false);
      const afterUnmount = notifications;
      await otherStarted;
      assert.equal(notifications, afterUnmount);
    } finally {
      finish();
      await started;
      unregister();
      unsubscribe();
      takeThreadCompactionRequest(id);
    }
  }
});

test("the compact menu ring follows the selected chat's current context usage", () => {
  const select = menuCallback("usedPercent", {
    menu: { item: { id: "ring-chat" } },
    deriveContextUsageBar,
  }) as (state: object) => number | null;
  const state = {
    activeThreadId: "ring-chat",
    contextUsage: { contextTokens: 780, totalTokens: 900 },
    contextUsageByThreadId: { "ring-chat": { totalTokens: 200 } },
    loadedContextLength: 1000 as number | null,
  };
  assert.equal(select(state), 78);
  state.contextUsage.contextTokens = 180;
  assert.equal(select(state), 18);
  state.contextUsage.contextTokens = 980;
  assert.equal(select(state), 98);
  state.activeThreadId = "another-chat";
  assert.equal(select(state), 20);
  state.contextUsageByThreadId = {} as typeof state.contextUsageByThreadId;
  assert.equal(select(state), null);
  state.contextUsageByThreadId = { "ring-chat": { totalTokens: 200 } };
  state.loadedContextLength = null;
  assert.equal(select(state), null);
  state.loadedContextLength = 1000;
  const priced = { ...state, params: { checkpoint: "unsloth/gguf-model" } };
  priced.contextUsageByThreadId = {
    "ring-chat": { totalTokens: 200, modelId: "unsloth/gguf-model" },
  } as typeof state.contextUsageByThreadId;
  assert.equal(select(priced), 20);
  priced.contextUsageByThreadId = {
    "ring-chat": { totalTokens: 200, modelId: "unsloth/other-gguf" },
  } as typeof state.contextUsageByThreadId;
  assert.equal(select(priced), null);
});

test("the compact menu item binds its action, live ring and busy state", () => {
  const source = ts.createSourceFile(
    "menu.tsx",
    readSrc("features/chat/components/chat-header-menu.tsx"),
    ts.ScriptTarget.Latest,
    true,
    ts.ScriptKind.TSX,
  );
  const items: ts.JsxElement[] = [];
  const rings: ts.JsxSelfClosingElement[] = [];
  function visit(node: ts.Node) {
    if (ts.isJsxElement(node) && node.openingElement.tagName.getText(source) === "Item" &&
      node.getText(source).includes('t("chatMenu.compact")')) items.push(node);
    ts.forEachChild(node, visit);
  }
  visit(source);
  assert.equal(items.length, 1);
  const item = items[0];
  function expression(tag: ts.JsxOpeningLikeElement, name: string) {
    const attribute = tag.attributes.properties.find((prop) =>
      ts.isJsxAttribute(prop) && prop.name.getText(source) === name,
    ) as ts.JsxAttribute | undefined;
    assert.ok(attribute?.initializer && ts.isJsxExpression(attribute.initializer));
    assert.ok(attribute.initializer.expression);
    return attribute.initializer.expression.getText(source);
  }
  assert.equal(expression(item.openingElement, "onSelect"), "compact");
  const disabled = new Function("menu", "busy", `return (${expression(item.openingElement, "disabled")});`);
  assert.equal(disabled({ item: { type: "single" } }, false), false);
  assert.equal(disabled({ item: { type: "single" } }, true), true);
  assert.equal(disabled({ item: { type: "compare" } }, false), true);
  function findRing(node: ts.Node) {
    if (ts.isJsxSelfClosingElement(node) && node.tagName.getText(source) === "UsageRing") rings.push(node);
    ts.forEachChild(node, findRing);
  }
  findRing(item);
  assert.equal(rings.length, 1);
  assert.equal(expression(rings[0], "percent"), "usedPercent");
});
