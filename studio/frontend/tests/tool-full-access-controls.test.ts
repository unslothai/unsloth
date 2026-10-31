// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { loadWithStubs, stubJsxRuntime, type StubElement } from "./helpers/module-stubs.ts";
import { hasChatFullAccess, withChatFullAccess, type ChatFullAccessGrant } from "../src/features/chat/utils/chat-full-access.ts";

type Node = StubElement;
function descendants(node: unknown): Node[] {
  if (Array.isArray(node)) return node.flatMap(descendants);
  if (!node || typeof node !== "object" || !("props" in node)) return [];
  const element = node as Node;
  return [element, ...descendants(element.props.children)];
}

function harness({ owner = true, disclosure = false, grantedThread = "", scopedFull = false } = {}) {
  const requests: unknown[][] = [], grants: ChatFullAccessGrant[] = [];
  const values: unknown[] = [];
  let index = 0, epoch = 0;
  let effects: Array<() => void> = [];
  let answer: () => Promise<boolean> = async () => true;
  const confirmation = {
    approvalId: "approval-a", sessionId: "project-shared", threadId: "chat-a",
    runId: "run-a", autoAllowKey: "project-shared:chat-a",
    ...(disclosure ? { imageDisclosure: { size_bytes: 1, server: "remote", tool: "view", destination: "outside" } } : {}),
  };
  const state = {
    toolConfirmations: { call: confirmation } as Record<string, typeof confirmation>,
    permissionMode: scopedFull ? "full" : "ask",
    chatFullAccessPreviousMode: scopedFull ? "ask" : null,
    bypassPermissions: scopedFull,
    confirmToolCalls: !scopedFull,
    deepResearchEnabled: false,
    chatPermissionChoiceVersion: 0,
    chatFullAccessGrants: grantedThread ? { other: { runId: "other", threadId: grantedThread, sessionId: "project-shared" } } : {},
    alwaysAllowToolsBySession: new Map(),
    allowToolAlways: () => {},
    grantChatFullAccess: (grant: ChatFullAccessGrant) => { grants.push(grant); },
    clearToolConfirmation: (id: string) => { delete state.toolConfirmations[id]; },
  };
  const store = Object.assign((selector: (s: typeof state) => unknown) => selector(state), { getState: () => state });
  const module = loadWithStubs<{ ToolConfirmationControls: (props: unknown) => Node }>(
    new URL("../src/components/assistant-ui/tool-confirmation-controls.tsx", import.meta.url), {
      "react/jsx-runtime": stubJsxRuntime(),
      react: {
        useCallback: (fn: unknown) => fn,
        useEffect: (effect: () => void) => { effects.push(effect); },
        useState: (initial: unknown) => {
          const slot = index++;
          if (!(slot in values)) values[slot] = initial;
          return [values[slot], (next: unknown) => { values[slot] = next; }];
        },
      },
      "@/components/ui/button": { Button: "Button" },
      "@/components/ui/alert-dialog": { AlertDialog: "Dialog" },
      "@/features/auth": { getAuthSessionEpoch: () => epoch },
      "@/features/auth/account-session": { useFullAccessAllowed: () => owner },
      "@/features/chat/permission-mode-select": { FullAccessConfirmContent: "FullAccessWarning" },
      "@/features/chat/utils/chat-full-access": { hasChatFullAccess, withChatFullAccess },
      "@/features/chat/api/chat-api": {
        resolveToolConfirmation: async (...args: unknown[]) => { requests.push(args); return answer(); },
        ToolApprovalGoneError: class Gone extends Error {},
      },
      "@/features/chat/stores/chat-runtime-store": { useChatRuntimeStore: store },
      "@/features/settings": {
        COMPOSER_INPUT_SELECTOR: "textarea", isSurfaceBackgrounded: () => false, useShortcut: () => {},
      },
      "@/features/chat": {
        useChatActive: () => true,
        useChatNavigationStore: (selector: (s: unknown) => unknown) => selector({ selectionActive: false }),
      },
    },
  );
  const render = () => {
    index = 0;
    effects = [];
    const tree = module.ToolConfirmationControls({ toolCallId: "call", toolName: "terminal", result: undefined, status: { type: "running" } });
    return descendants(tree);
  };
  const get = (nodes: Node[], type: string, label?: string) => {
    const element = nodes.find((node) => node.type === type && (label === undefined || node.props.children === label));
    assert.ok(element, `missing ${label ?? type}`);
    return element;
  };
  return {
    render, get, requests, grants, state,
    setAnswer: (next: () => Promise<boolean>) => { answer = next; },
    changeAccount: () => { epoch += 1; },
    runEffects: () => { for (const effect of effects) effect(); },
  };
}

async function settle() { await new Promise((resolve) => setTimeout(resolve, 0)); }
function click(node: Node, prop = "onClick") { (node.props[prop] as () => void)(); }

test("the shortcut opens the existing warning and cancel grants nothing", async () => {
  const h = harness();
  click(h.get(h.render(), "Button", "Full Access"));
  const nodes = h.render();
  assert.equal(h.get(nodes, "Dialog").props.open, true);
  assert.equal(h.requests.length, 0);
  click(h.get(nodes, "FullAccessWarning"), "onClose");
  await settle();
  assert.equal(h.get(h.render(), "Dialog").props.open, false);
  assert.deepEqual(h.requests, []);
  assert.deepEqual(h.grants, []);
});

test("Turn on upgrades the parked call before recording a grant for that chat", async () => {
  const h = harness();
  click(h.get(h.render(), "Button", "Full Access"));
  click(h.get(h.render(), "FullAccessWarning"), "onConfirm");
  assert.deepEqual(h.grants, [], "a pending POST is not a grant");
  await settle();
  assert.deepEqual(h.requests, [["project-shared", "approval-a", "allow", true]]);
  assert.deepEqual(h.grants, [{ runId: "run-a", sessionId: "project-shared", threadId: "chat-a" }]);
});

test("server failure or an unacknowledged legacy upgrade never records Full access", async () => {
  for (const answer of [async () => false, async () => { throw new Error("offline"); }]) {
    const h = harness();
    h.setAnswer(answer);
    click(h.get(h.render(), "Button", "Full Access"));
    click(h.get(h.render(), "FullAccessWarning"), "onConfirm");
    await settle();
    assert.deepEqual(h.grants, []);
    assert.ok(h.state.toolConfirmations.call);
  }
});

test("a delayed acknowledgment cannot grant the next signed-in account", async () => {
  const h = harness();
  let finish!: (ok: boolean) => void;
  h.setAnswer(() => new Promise((resolve) => { finish = resolve; }));
  click(h.get(h.render(), "Button", "Full Access"));
  click(h.get(h.render(), "FullAccessWarning"), "onConfirm");
  h.changeAccount();
  finish(true);
  await settle();
  assert.deepEqual(h.grants, []);
});

test("restricted accounts and explicit image disclosure never offer the shortcut", () => {
  for (const options of [{ owner: false }, { disclosure: true }]) {
    const h = harness(options);
    const nodes = h.render();
    assert.ok(!nodes.some((node) => node.type === "Button" && node.props.children === "Full Access"));
  }
});

test("already parked calls upgrade only within the confirmed chat", async () => {
  const own = harness({ grantedThread: "chat-a" });
  own.render(); own.runEffects(); await settle();
  assert.deepEqual(own.requests, [["project-shared", "approval-a", "allow", true]]);
  const other = harness({ grantedThread: "chat-b" });
  other.render(); other.runEffects(); await settle();
  assert.deepEqual(other.requests, []);
  const image = harness({ grantedThread: "chat-a", disclosure: true });
  image.render(); image.runEffects(); await settle();
  assert.deepEqual(image.requests, [], "Full access cannot auto-share an attached image");
});

test("a different chat's scoped Full Access does not hide this card's shortcut", () => {
  const h = harness({ grantedThread: "chat-b", scopedFull: true });
  assert.ok(h.get(h.render(), "Button", "Full Access"));
  assert.deepEqual(h.requests, []);
});


test("a newer explicit permission choice wins over a delayed Full Access acknowledgment", async () => {
  for (const mode of ["ask", "off"]) {
    const h = harness();
    let finish!: (ok: boolean) => void;
    h.setAnswer(() => new Promise((resolve) => { finish = resolve; }));
    click(h.get(h.render(), "Button", "Full Access"));
    click(h.get(h.render(), "FullAccessWarning"), "onConfirm");
    h.state.chatPermissionChoiceVersion += 1;
    h.state.permissionMode = mode;
    finish(true);
    await settle();
    assert.deepEqual(h.grants, []);
    assert.equal(h.state.permissionMode, mode);
  }
});
